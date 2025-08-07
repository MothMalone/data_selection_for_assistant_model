#!/usr/bin/env python3
"""
Evaluation Pipeline for DialogSum Summarization

This script evaluates fine-tuned LLaMA models on DialogSum test set using ROUGE and BERTScore metrics.
Revamped to use the preferred evaluation format with proper wandb integration.
"""

import os
import sys
import json
import time
import logging
import argparse
import random
import socket
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from itertools import chain
from datasets import Dataset, load_dataset

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, set_seed
from torch.utils.data import DataLoader
import pandas as pd
from peft import PeftModel
from tqdm import tqdm
import wandb
import evaluate
from evaluate import load

# Add paths for imports (if needed for score module)
try:
    sys.path.append("/storage/nammt/KD-SLM/Multi-Level-OT/llm_distillation")
    sys.path.append(f"{os.getenv('HOME')}/Multi-Level-OT/llm_distillation")
    sys.path.append(f"{os.getenv('HOME')}/Multi-Level-OT")
    import score
    HAS_SCORE_MODULE = True
except ImportError:
    # Fallback to evaluate library
    import evaluate
    from evaluate import load
    HAS_SCORE_MODULE = False

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_device():
    """Get the best available device."""
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    return device


def set_seed_for_reproducibility(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    set_seed(seed)
    logger.info(f"Random seed set to {seed} for reproducibility")


def tokenization(items, tokenizer):
    """Tokenize items for batch processing."""
    return tokenizer(items["prompt"], padding='longest', truncation=True, max_length=512)


def create_dialogsum_prompt(dialogue: str) -> str:
    """Create prompt for DialogSum evaluation."""
    return f"Summarize the following dialogue:\n\n{dialogue}\n\nSummary:"


class SummarizationEvaluator:
    """Handles evaluation of summarization models."""

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.rouge_scorer = load("rouge")
        # Load BERTScore metric
        self.bertscore_scorer = evaluate.load("bertscore")

    def load_test_data(self, max_samples: int = None, seed: int = 42) -> Dataset:
        """Load and optionally subsample the DialogSum test dataset."""
        logger.info("Loading DialogSum test dataset...")
        test_dataset = load_dataset("knkarthick/dialogsum", split="test", cache_dir=self.cache_dir)

        if max_samples and len(test_dataset) > max_samples:
            test_dataset = test_dataset.shuffle(seed=seed).select(range(max_samples))
        
        logger.info(f"Loaded {len(test_dataset)} test samples.")
        return test_dataset

    def load_model_and_tokenizer(self, model_path: str, base_model: str):
        """Load a fine-tuned model and its tokenizer."""
        device = get_device()
        logger.info(f"Loading model from {model_path} on device: {device}")

        # Load base model with 4-bit quantization
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        # Set padding side to left for decoder-only models as recommended
        tokenizer.padding_side = 'left'

        # Load LoRA adapter
        model = PeftModel.from_pretrained(base, model_path)
        model.eval()

        logger.info("Model and tokenizer loaded successfully.")
        return model, tokenizer

    def generate_summaries(self, model, tokenizer, dataset, batch_size: int = 16, max_new_tokens: int = 150):
        """Generate summaries for the entire dataset."""
        logger.info("Starting summary generation...")
        
        predictions = []
        references = []
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="Generating Summaries"):
            batch = dataset[i:i+batch_size]
            dialogues = batch['dialogue']
            summaries = batch['summary']
            
            prompts = [create_dialogsum_prompt(d) for d in dialogues]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
            
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Clean up the generated text to only get the summary
            # Handle cases where "Summary:" isn't in the output
            cleaned_preds = []
            for pred in decoded_preds:
                if "Summary:" in pred:
                    cleaned_preds.append(pred.split("Summary:")[1].strip())
                else:
                    # If "Summary:" is not found, use everything after the dialogue
                    # or just use the entire output as fallback
                    parts = pred.split("Summarize the following dialogue:")
                    if len(parts) > 1:
                        # Try to extract just the summary part after the dialogue
                        dialogue_and_summary = parts[1].strip()
                        dialogue_parts = dialogue_and_summary.split("\n\n")
                        if len(dialogue_parts) > 1:
                            # The last part should be the summary
                            cleaned_preds.append(dialogue_parts[-1].strip())
                        else:
                            # Fallback to using everything
                            cleaned_preds.append(dialogue_and_summary)
                    else:
                        # Just use the original output as a last resort
                        cleaned_preds.append(pred.strip())
                        
            predictions.extend(cleaned_preds)
            references.extend(summaries)

        logger.info(f"Generated {len(predictions)} summaries.")
        return predictions, references

    def compute_and_log_metrics(self, predictions, references, log_interval=10):
        """Compute ROUGE and BERTScore scores and log them incrementally to wandb."""
        logger.info("Computing and logging metrics...")
        
        all_rouge_scores = []
        all_bert_scores = []

        for i in tqdm(range(0, len(predictions), log_interval), desc="Calculating Metrics"):
            pred_chunk = predictions[i:i+log_interval]
            ref_chunk = references[i:i+log_interval]
            
            if not pred_chunk or not ref_chunk:
                continue

            # ROUGE scores
            rouge_scores = self.rouge_scorer.compute(predictions=pred_chunk, references=ref_chunk, use_stemmer=True)
            all_rouge_scores.append(rouge_scores)
            
            # BERTScore
            bert_scores = self.bertscore_scorer.compute(predictions=pred_chunk, references=ref_chunk, lang="en")
            all_bert_scores.append(bert_scores)
            
            # Aggregate scores up to the current point for logging
            agg_preds = predictions[:i+log_interval]
            agg_refs = references[:i+log_interval]
            agg_rouge_scores = self.rouge_scorer.compute(predictions=agg_preds, references=agg_refs, use_stemmer=True)
            agg_bert_scores = self.bertscore_scorer.compute(predictions=agg_preds, references=agg_refs, lang="en")

            log_data = {
                "eval/step": i + log_interval,
                "eval/bertscore_precision": np.mean(agg_bert_scores['precision']),
                "eval/bertscore_recall": np.mean(agg_bert_scores['recall']),
                "eval/bertscore_f1": np.mean(agg_bert_scores['f1']),
            }

            # Handle both object-based and direct float score formats for ROUGE
            try:
                # Try the object format first (some ROUGE implementations)
                log_data.update({
                    "eval/rouge1": agg_rouge_scores["rouge1"].mid.fmeasure,
                    "eval/rouge2": agg_rouge_scores["rouge2"].mid.fmeasure,
                    "eval/rougeL": agg_rouge_scores["rougeL"].mid.fmeasure,
                    "eval/rougeLsum": agg_rouge_scores["rougeLsum"].mid.fmeasure,
                })
            except AttributeError:
                # Fall back to direct float format
                log_data.update({
                    "eval/rouge1": agg_rouge_scores["rouge1"],
                    "eval/rouge2": agg_rouge_scores["rouge2"],
                    "eval/rougeL": agg_rouge_scores["rougeL"],
                    "eval/rougeLsum": agg_rouge_scores["rougeLsum"],
                })
            wandb.log(log_data)


        # Final overall scores
        final_rouge_scores = self.rouge_scorer.compute(predictions=predictions, references=references, use_stemmer=True)
        final_bert_scores = self.bertscore_scorer.compute(predictions=predictions, references=references, lang="en")
        
        logger.info(f"Final ROUGE scores: {final_rouge_scores}")
        logger.info(f"Final BERTScore: Precision: {np.mean(final_bert_scores['precision'])}, Recall: {np.mean(final_bert_scores['recall'])}, F1: {np.mean(final_bert_scores['f1'])}")

        # Prepare final results dictionary
        results = {}
        try:
            # Try the object format first for ROUGE
            results.update({k: v.mid.fmeasure for k, v in final_rouge_scores.items()})
        except AttributeError:
            # Fall back to direct float format
            results.update(final_rouge_scores)
        
        results.update({
            'bertscore_precision': np.mean(final_bert_scores['precision']),
            'bertscore_recall': np.mean(final_bert_scores['recall']),
            'bertscore_f1': np.mean(final_bert_scores['f1']),
        })

        return results


def load_training_results(results_file: str) -> Dict[str, Any]:
    """Load training results."""
    with open(results_file, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DialogSum Summarization Models")
    parser.add_argument("--model_dirs", type=str, nargs='+', required=True,
                        help="Directories containing fine-tuned models")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-13b-hf",
                        help="Base LLaMA model name")
    parser.add_argument("--max_test_samples", type=int, default=500,
                        help="Maximum number of test samples to evaluate (default: 500)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Output directory for evaluation results")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Cache directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--enable_wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="data-selection-experiments",
                        help="W&B project name")
    parser.add_argument("--wandb_run_group", type=str, default=None,
                        help="W&B run group name")
    parser.add_argument("--wandb_api_key", type=str, default=os.getenv("WANDB_API_KEY"),
                        help="Wandb API key for logging")

    args = parser.parse_args()

    # Set random seeds for reproducibility
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize evaluator
    evaluator = SummarizationEvaluator(cache_dir=args.cache_dir)

    # Load test data
    test_dataset = evaluator.load_test_data(
        max_samples=args.max_test_samples,
        seed=args.seed
    )

    # Evaluate each model
    all_results = {}

    for model_dir_str in args.model_dirs:
        model_dir = Path(model_dir_str)
        model_key = model_dir.name
        logger.info(f"\n--- Evaluating model: {model_key} ---")

        if args.enable_wandb:
            run_name = f"eval_{model_key}"
            wandb.init(
                project=args.wandb_project,
                group=args.wandb_run_group,
                name=run_name,
                config=vars(args),
                reinit=True
            )

        try:
            model, tokenizer = evaluator.load_model_and_tokenizer(
                model_path=str(model_dir),
                base_model=args.base_model
            )

            predictions, references = evaluator.generate_summaries(
                model, tokenizer, test_dataset, batch_size=args.batch_size
            )
            
            if args.enable_wandb:
                scores = evaluator.compute_and_log_metrics(predictions, references)
            else:
                rouge_scores = evaluator.rouge_scorer.compute(predictions=predictions, references=references, use_stemmer=True)
                bert_scores = evaluator.bertscore_scorer.compute(predictions=predictions, references=references, lang="en")
                
                # Handle different return formats for ROUGE
                rouge_scores = {
                    k: v.mid.fmeasure if hasattr(v, 'mid') else v 
                    for k, v in rouge_scores.items()
                }
                
                scores = {**rouge_scores, 'bertscore_f1': np.mean(bert_scores['f1'])}


            all_results[model_key] = scores

            # Log final results table to wandb
            if args.enable_wandb:
                df = pd.DataFrame({
                    "dialogue": [d['dialogue'] for d in test_dataset],
                    "reference": references,
                    "prediction": predictions
                })
                wandb.log({"evaluation_samples": wandb.Table(dataframe=df)})

        except Exception as e:
            logger.error(f"Failed to evaluate model {model_key}: {e}", exc_info=True)
        finally:
            if args.enable_wandb:
                wandb.finish()

    # Save combined results
    combined_results_file = output_dir / "all_evaluation_results.json"
    with open(combined_results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Create and save comparison summary
    comparison_summary = {}
    for model_key, scores in all_results.items():
        method_parts = model_key.split('_')
        
        # Extract the selection method (first part)
        selection_method = method_parts[0]
        
        # Try to extract the number of samples
        num_samples = 0
        for part in method_parts[1:]:
            if part.isdigit():  # If the part is just a number
                num_samples = int(part)
                break
            elif "samples" in part:  # If it contains "samples" like "10samples"
                try:
                    num_samples = int(part.replace("samples", ""))
                    break
                except ValueError:
                    pass
        
        comparison_summary[model_key] = {
            "selection_method": selection_method,
            "num_samples": num_samples,
            "rouge1": scores.get('rouge1', 0),
            "rouge2": scores.get('rouge2', 0),
            "rougeL": scores.get('rougeL', 0),
            "rougeLsum": scores.get('rougeLsum', 0),
            "bertscore_f1": scores.get('bertscore_f1', 0),
        }

    summary_file = output_dir / "comparison_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(comparison_summary, f, indent=4)

    logger.info(f"Comparison summary saved to {summary_file}")
    logger.info("Evaluation completed successfully!")

    # Print summary table
    print("\n" + "="*95)
    print("EVALUATION SUMMARY")
    print("="*95)
    print(f"{'Model':<40} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-Lsum':<12} {'BERTScore-F1':<15}")
    print("-" * 95)
    for model_key, summary in comparison_summary.items():
        print(f"{model_key:<40} {summary['rouge1']:<10.4f} {summary['rouge2']:<10.4f} {summary['rougeLsum']:<12.4f} {summary['bertscore_f1']:<15.4f}")
    print("="*95)


if __name__ == "__main__":
    main()
