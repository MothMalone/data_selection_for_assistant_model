#!/usr/bin/env python3
"""
Local Few-Shot Evaluation Pipeline for DialogSum Summarization

This script adapts the few-shot evaluation logic to run on a local machine.
It evaluates fine-tuned LLaMA models on the DialogSum test set using a few-shot prompting strategy.
"""

import os
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List

import torch
import wandb
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
from tqdm import tqdm
import pandas as pd
import evaluate

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Step 1: Port the Core Few-Shot Logic ---
# This is the core logic from your Modal script for building prompts.

DATASET_CONTEXT = {
    "summary_dialogue": "You're an expert at summarizing dialogues. You have to read the dialogue between two people and summarize it in no more than one sentence. The summary should be as short as possible, not re-explaining the dialogue in detail and using the person's name when implicitly mentioned."
}

FEW_SHOT_EXAMPLES = [
        {
            "context": "#Person1#: John, shall we go to Sun Store? I have decided to buy that Murrberry handbag. Anyway,I'm not carrying this one to Mary's wedding.\n#Person2#: But, Jane, why not rent one with Handbag Hire? Instead of $ 990,pay $ 50,and you have it for a whole week.\n#Person1#: Sounds great, but I never knew I can rent a handbag.\n#Person2#: Handbag Hire is a new business. It was founded two months ago. Its collection covers many designer handbags.\n#Person1#: So... for the price of one Murrberry, I can use a different bag each week for twenty weeks?\n#Person2#: Absolutely. And if you like one of them, you can choose to buy it at a discounted rate. Of course the price varies by age and condition. For example, a $ 1500 Murrberry bag can sell for just $750.\n#Person1#: Great, but how do I rent? By telephone? Or in person?\n#Person2#: Either. And more conveniently, it accepts online orders.\n#Person1#: I'll do it on line now. I still have one more question. Mary's wedding is next Saturday. There are only five days left. Do I have enough time?\n#Person2#: Don't worry. It promises that customers receive their orders by post within two days. Three more days to go.\n#Person1#: Oh, I'd better order one right now.",
            "answers": "Jane wants to buy that Murrberry handbag to carry to Mary's wedding, but John suggests renting one with Handbag Hire and tells her about the service in detail. Jane is pleased to have a try."
        },
        {
            "context": "#Person1#: The summers are so great here! Not hot at all. I love the cooling breezes, the clear air, all the greenery.\n#Person2#: This really has been a wonderful holiday for us. Shall we take a walk around the pond or into those woods for a while?\n#Person1#: Let's do both! Are we in a rush or anything\n#Person2#: No, not really. I had thought we'd stay in Hamburg tonight, but we can't unless we rush it. Let's stay in Bremen instead. Tomorrow we can have lunch in Hamburg, then check into a hostel in Copenhagen and have dinner there.\n#Person1#: Sounds fine to me. Whatever, let's enjoy this pond first.\n#Person2#: Sure. We can walk around to that path that leads into the woods there. Hey, look! There are some wild ducks over there in the reeds.\n#Person1#: I see them! Wow! How do you know they're wild?\n#Person2#: I used to go hunting with my uncle, that's how.\n#Person1#: They're neat. Now Let's take that path into the woods and see what we can see. . .",
            "answers": "#Person1# and #Person2# are enjoying a pond. #Person1# and #Person2# had planned to stay in Hamburg tonight, but they decide to stay in Bremen since they are not in a rush."
        },
        {
            "context": "#Person1#: Well Rebecca, is there anything else you need to know for now?\n#Person2#: I don't think so, Mr. Parsons. I think you have covered all the main points for me.\n#Person1#: Okay well listen, here is my business card with my mobile number. If any other questions spring to mind don't hesitate to contact me. Of course you can also call Miss Childs too.\n#Person2#: Great. Rmm, when can I expect to hear from you?\n#Person1#: Well, we are finishing the shortlist interviews tomorrow, so we will certainly have a decision made by early next week. Miss Childs will call you to discuss more on Monday or Tuesday. How does that so\n#Person2#: That sounds perfect. Thank you very much for taking the time to speak to me Mr. Parsons.\n#Person1#: The pleasure's all mine, Rebecca.\n#Person2#: I hope to hear from you very soon.\n#Person1#: Absolutely. Thanks for coming Rebecca. Goodbye.",
            "answers": "Mr. Parsons gives Rebecca his business card after the interview and tells Rebecca the decision will be made by early next week and Miss Childs will contact Rebecca."
        },
        {
            "context": "#Person1#: Trina, will you marry me?\n#Person2#: Yes! Yes! And yes! Jared, of course I'll marry you!\n#Person1#: Oh, Babe, I can't wait to spend the rest of my life with you! I can't wait for all the adventures we're going to have, for all the fights and the laughter. I can't wait to grow old and wrinkly with you.\n#Person2#: Oh, Jared! I can't wait for our wedding! I hope you don't mind, but I'Ve already chosen a date! Six months from now in the summer! Melissa saw you buying the ring last month so I'Ve had plenty of time to start planning!\n#Person1#: She what?\n#Person2#: Oh don't worry, sweetie, I didn't know when you were going to propose. It was still a nice surprise! As I was saying, I'Ve got it all planned out. There's almost nothing left to do! I wrote up our guest list and we will have roughly four hundred guests attending.\n#Person1#: Four hundred?\n#Person2#: No need to sweat it. My parents agreed to pay for most of the wedding, which is going to be low-budget anyway. So roughly four hundred people, which means that the hall at Northwoods Heights will be our reception venue. I thought it would be nice if we had the wedding at your parents'church and my uncle of course would be officiating. We'll meet with him soon for some pre-wedding counseling. The music for the wedding ceremony was a no-brainer. My step-sister and her string quartet will take care of that. My cousin will be the official photographer. I thought it would also be nice if his daughter could sing a solo. Did you know that she's going to be a professional opera singer?\n#Person1#: Ah. . .\n#Person2#: And then of course the ladies at the church would love to be our caterers for the banquet and we'll get the Youth Group to serve us. I was thinking that your friend's band could be our entertainment for the night. though they might have to tone it down a bit. Or we could hire a DJ. Your sister's husband could get us a discount with that company that does the decor at weddings. what's their name again? I was thinking that we could have an island paradise-themed wedding and our theme color would be a soothing blue like Aquamarine. And there will be a huge seashell on the wall behind the podium where we'll make our toasts! What do you think of small packages of drink mixes for our wedding favors? Who else am I missing? Oh, your uncle could be our florist and his wife could make our wedding cake!\n#Person1#: Wow.\n#Person2#: See? It's going to be wonderful! Oh this wedding is going to be everything I ever dreamed of.\n#Person1#: If I survive the next six months.",
            "answers": "Trina accepts Jared's proposal. Then, Jared is astonished to know that Trina already knew from Melissa who saw him buying the ring that he was planning this. Trina has chosen a date and has made a list of four hundred guests and she tells Jared about her arrangements in an ecstasy. Jared finds it hard to get through."
        },
        {
            "context": "#Person1#: Isabelle, you know I'm not interested in fame. \n#Person2#: Well, you don't seem to be interested in getting a real job, either. \n#Person1#: You know I'm interested in teaching. I'm looking for jazz students. . . \n#Person2#: Yeah, and every high school student in town is banging on your door, right? \n#Person1#: I know they're out there. I'll find them. \n#Person2#: You're such a dreamer! You think that you can spread the word of jazz in an underpass? ",
            "answers": "Isabelle thinks #Person1# is a dreamer because #Person1# doesn't do real things."
        },
        {
            "context": "#Person1#: Hi, Tony. You look unhappy. What's wrong? #Person2#: Oh, Steven, I made a big mistake. #Person1#: What happened? #Person2#: I really wish I hadn't done it. #Person1#: What on earth are you talking about? #Person2#: I got caught cheating. I feel so ashamed. The teacher saw me and told me I failed. #Person1#: What were you thinking? #Person2#: You know my father. If I fail, he'll kill me. I have to do well. #Person1#: But what you should do is study hard. #Person2#: I know. . . I know. . . it's all my fault. I feel awful that I didn't study, and I cheated, and I got caught. #Person1#: So long as you learn from your mistakes. ",
            "answers": "Tony got caught cheating and feels ashamed. Steven tells him to learn from it and study hard."
        },
        {
            "context": "#Person1#: Would you like to go to the party tonight? #Person2#: Whose party? #Person1#: Ruojia's. Don't you know that? Ruojia has got married. #Person2#: What! Is she really? I can't believe it! #Person1#: Yes. Yesterday. #Person2#: Good gracious. That's incredible! I feel so happy for her! #Person1#: Yes, me too. #Person2#: But how do you know that? #Person1#: I saw the news from her twitter. And she sent an email about it. #Person2#: What? I didn't receive it! #Person1#: Maybe you should check your email. #Person2#: Oh yes, I find it. Tonight at her home. Will you bring something? #Person1#: Yes, a pair of wineglasses and a card to wish her happy marriage. #Person2#: I will buy a tea set.",
            "answers": "#Person1# tells #Person2# that Ruojia is married and will have a party tonight. #Person2#'s surprised to know that. They will bring their gifts to bless her."
        }
]

def create_few_shot(number_few_shot: int):
    shot = []
    for i in range(min(number_few_shot, len(FEW_SHOT_EXAMPLES))):
        shot.append(
            f"Dialogue: {FEW_SHOT_EXAMPLES[i]['context']}\nSummary: {FEW_SHOT_EXAMPLES[i]['answers']}"
        )
    return shot

def create_request(context=""):
    return f"Dialogue: {context}\nSummary:"

def create_prompt(task: str = "summary_dialogue", few_shot: int = 3, context: str = ""):
    prompt = DATASET_CONTEXT.get(task, "") + "\n\n"
    request = create_request(context=context)
    if few_shot > 0:
        shot_examples = create_few_shot(few_shot)
        shot_text = '\n\n'.join(shot_examples)
        prompt += f"{shot_text}\n\n{request}"
    else:
        prompt += request
    return prompt

# --- End of Ported Logic ---


class FewShotEvaluator:
    """Handles few-shot evaluation of summarization models locally."""

    def __init__(self, cache_dir: str = "./cache", log_interval: int = 50):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.rouge_scorer = evaluate.load("rouge")
        self.bertscore_scorer = evaluate.load("bertscore")
        self.log_interval = log_interval

    def load_test_data(self, max_samples: int = None, seed: int = 42) -> Dataset:
        logger.info("Loading DialogSum test dataset...")
        test_dataset = load_dataset("knkarthick/dialogsum", split="test", cache_dir=self.cache_dir)
        if max_samples and len(test_dataset) > max_samples:
            test_dataset = test_dataset.shuffle(seed=seed).select(range(max_samples))
        logger.info(f"Loaded {len(test_dataset)} test samples.")
        return test_dataset

    def load_model_and_tokenizer(self, model_path: str, base_model: str):
        """Loads a PEFT model from a local path."""
        logger.info(f"Loading base model '{base_model}' with 4-bit quantization...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        logger.info(f"Loading LoRA adapter from '{model_path}'...")
        model = PeftModel.from_pretrained(base, model_path)
        model.eval()
        logger.info("Model and tokenizer loaded successfully.")
        return model, tokenizer

    def generate_summaries(self, model, tokenizer, dataset, batch_size: int, max_new_tokens: int, num_few_shots: int, output_dir: Path):
        """Generate summaries with incremental logging every log_interval samples."""
        logger.info(f"Starting summary generation with {num_few_shots} few-shot examples...")
        predictions, references = [], []
        
        # Create CSV file for incremental results
        incremental_results_file = output_dir / "incremental_results.csv"
        results_df = pd.DataFrame(columns=["num_samples", "rouge1", "rouge2", "rougeL", "rougeLsum", 
                                          "bertscore_precision", "bertscore_recall", "bertscore_f1"])
        
        # Process samples
        total_processed = 0
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="Generating Summaries"):
            batch = dataset[i:i+batch_size]
            dialogues = batch['dialogue']
            summaries = batch['summary']
            
            # Use the prompt creation logic
            prompts = [create_prompt(context=d, few_shot=num_few_shots) for d in dialogues]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
            
            # Decode only the newly generated tokens
            input_lengths = inputs.input_ids.shape[1]
            newly_generated_tokens = outputs[:, input_lengths:]
            decoded_preds = tokenizer.batch_decode(newly_generated_tokens, skip_special_tokens=True)
            
            cleaned_preds = [pred.strip() for pred in decoded_preds]
            predictions.extend(cleaned_preds)
            references.extend(summaries)
            
            # Update processed count
            total_processed += len(batch)
            
            # Log metrics every log_interval samples
            if total_processed % self.log_interval < batch_size:
                logger.info(f"Processed {total_processed} samples. Computing intermediate metrics...")
                
                # Compute metrics on samples processed so far
                metrics = self.compute_metrics(predictions, references)
                
                # Log to wandb
                wandb.log({
                    "samples_processed": total_processed,
                    "intermediate_rouge1": metrics["rouge1"],
                    "intermediate_rouge2": metrics["rouge2"],
                    "intermediate_rougeL": metrics["rougeL"],
                    "intermediate_rougeLsum": metrics["rougeLsum"],
                    "intermediate_bertscore_f1": metrics["bertscore_f1"],
                })
                
                # Add to CSV
                new_row = {
                    "num_samples": total_processed,
                    "rouge1": metrics["rouge1"],
                    "rouge2": metrics["rouge2"],
                    "rougeL": metrics["rougeL"],
                    "rougeLsum": metrics["rougeLsum"],
                    "bertscore_precision": metrics["bertscore_precision"],
                    "bertscore_recall": metrics["bertscore_recall"],
                    "bertscore_f1": metrics["bertscore_f1"]
                }
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                results_df.to_csv(incremental_results_file, index=False)
                
                # Also save predictions up to this point
                interim_preds_df = pd.DataFrame({
                    "dialogue": [d['dialogue'] for d in dataset[:total_processed]], 
                    "reference": references, 
                    "prediction": predictions
                })
                interim_preds_df.to_csv(output_dir / f"predictions_samples_{total_processed}.csv", index=False)
                
                logger.info(f"Intermediate results at {total_processed} samples: ROUGE-1={metrics['rouge1']:.4f}, BERTScore={metrics['bertscore_f1']:.4f}")

        logger.info(f"Generated {len(predictions)} summaries.")
        return predictions, references

    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute evaluation metrics between predictions and references."""
        rouge_scores = self.rouge_scorer.compute(predictions=predictions, references=references, use_stemmer=True)
        bert_scores = self.bertscore_scorer.compute(predictions=predictions, references=references, lang="en")
        
        results = {
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'rougeLsum': rouge_scores['rougeLsum'],
            'bertscore_precision': np.mean(bert_scores['precision']),
            'bertscore_recall': np.mean(bert_scores['recall']),
            'bertscore_f1': np.mean(bert_scores['f1']),
        }
        return results

def main():
    # --- Step 2: Use argparse for local execution ---
    parser = argparse.ArgumentParser(description="Local Few-Shot Evaluation for DialogSum")
    parser.add_argument("--model_path", type=str, required=True, help="Local directory containing the fine-tuned LoRA adapter")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-7b-hf", help="Base LLaMA model name")
    parser.add_argument("--num_few_shots", type=int, default=3, help="Number of few-shot examples to include in the prompt")
    parser.add_argument("--max_test_samples", type=int, default=500, help="Maximum number of test samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="./fewshot_results", help="Output directory for results")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--log_interval", type=int, default=50, help="Log metrics every N samples")
    parser.add_argument("--wandb_project", type=str, default="dialogsum-d2pruning-eval", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_api_key", type=str, default="445d8df72343591d6588f101349ad4752497ce62", help="W&B API key")
    args = parser.parse_args()

    # Initialize wandb
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or f"eval-{Path(args.model_path).name}-{args.num_few_shots}shot",
        config=vars(args)
    )

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # --- Step 3: Run the Evaluation Flow ---
    evaluator = FewShotEvaluator(cache_dir=args.cache_dir, log_interval=args.log_interval)
    test_dataset = evaluator.load_test_data(max_samples=args.max_test_samples, seed=args.seed)
    model, tokenizer = evaluator.load_model_and_tokenizer(model_path=args.model_path, base_model=args.base_model)

    predictions, references = evaluator.generate_summaries(
        model, tokenizer, test_dataset, 
        batch_size=args.batch_size, 
        max_new_tokens=128,  # Summaries are short
        num_few_shots=args.num_few_shots,
        output_dir=output_dir
    )

    # Compute final metrics
    scores = evaluator.compute_metrics(predictions, references)
    
    # Log final results to wandb
    wandb.log({
        "final_rouge1": scores["rouge1"],
        "final_rouge2": scores["rouge2"],
        "final_rougeL": scores["rougeL"],
        "final_rougeLsum": scores["rougeLsum"],
        "final_bertscore_precision": scores["bertscore_precision"],
        "final_bertscore_recall": scores["bertscore_recall"],
        "final_bertscore_f1": scores["bertscore_f1"],
    })
    
    # --- Step 4: Save Final Results Locally ---
    model_key = Path(args.model_path).name
    results_file = output_dir / f"results_{model_key}.json"
    with open(results_file, 'w') as f:
        json.dump(scores, f, indent=4)
    
    # Save final predictions
    df = pd.DataFrame({"dialogue": [d['dialogue'] for d in test_dataset], "reference": references, "prediction": predictions})
    df.to_csv(output_dir / f"predictions_{model_key}.csv", index=False)

    logger.info(f"Evaluation for '{model_key}' complete. Results saved to {results_file}")
    
    # --- Step 5: Print Summary Table ---
    print("\n" + "="*80)
    print(f"FEW-SHOT EVALUATION SUMMARY FOR: {model_key}")
    print("="*80)
    print(f"{'Metric':<20} {'Score'}")
    print("-"*80)
    for key, value in scores.items():
        print(f"{key:<20} {value:.4f}")
    print("="*80)
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()