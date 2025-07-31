#!/usr/bin/env python3
"""
Gradient Norm-based Sample Selection for DialogSum

This script selects the most informative samples using gradient norms from a teacher model.
Higher gradient norms indicate samples that are more difficult/informative for learning.
"""

import os
import json
import time
import logging
import argparse
from typing import List, Dict, Tuple, Any
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class GradientNormSampleSelector:
    """
    Efficient sample selection using gradient norms for knowledge distillation.
    Selects samples that are most informative for learning based on gradient magnitudes.
    """
    
    def __init__(self, 
                 teacher_model_name: str,
                 device: str = 'cuda',
                 batch_size: int = 4,
                 max_length: int = 512,
                 cache_dir: str = "./cache"):
        self.teacher_model_name = teacher_model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self):
        """Load the teacher model and tokenizer."""
        logger.info(f"Loading teacher model: {self.teacher_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.teacher_model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model based on type
        try:
            if "llama" in self.teacher_model_name.lower() or "gpt" in self.teacher_model_name.lower():
                self.teacher_model = AutoModelForCausalLM.from_pretrained(
                    self.teacher_model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float32,  # Use float32 for gradient computation
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # For seq2seq models like BART, T5
                self.teacher_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.teacher_model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    trust_remote_code=True
                )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Set to train mode for gradient computation
        self.teacher_model.train()
        
        # Enable gradient computation for all parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = True
            
        logger.info(f"Model loaded successfully on device: {self.device}")
    
    def compute_sample_gradient_norms(self, dataset) -> np.ndarray:
        """
        Compute gradient norms for each sample in the dataset.
        
        Args:
            dataset: HuggingFace dataset with dialogue and summary
        
        Returns:
            Array of gradient norms for each sample
        """
        gradient_norms = []
        
        logger.info(f"Computing gradient norms for {len(dataset)} samples...")
        
        # Process in batches
        for i in tqdm(range(0, len(dataset), self.batch_size), desc="Computing gradients"):
            batch_end = min(i + self.batch_size, len(dataset))
            batch = dataset.select(range(i, batch_end))
            
            batch_gradient_norms = self._compute_batch_gradient_norms(batch)
            gradient_norms.extend(batch_gradient_norms)
        
        return np.array(gradient_norms)
    
    def _compute_batch_gradient_norms(self, batch) -> List[float]:
        """Compute gradient norms for a single batch."""
        batch_norms = []
        
        for sample in batch:
            try:
                # Prepare input text (dialogue + summary for causal LM)
                if "llama" in self.teacher_model_name.lower() or "gpt" in self.teacher_model_name.lower():
                    # For causal LM: combine dialogue and summary
                    input_text = f"Dialogue: {sample['dialogue']}\nSummary: {sample['summary']}"
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        input_text,
                        return_tensors='pt',
                        max_length=self.max_length,
                        truncation=True,
                        padding=True
                    ).to(self.device)
                    
                    # Zero gradients
                    self.teacher_model.zero_grad()
                    
                    # Forward pass with labels for loss computation
                    outputs = self.teacher_model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss
                    
                else:
                    # For seq2seq models: dialogue as input, summary as target
                    inputs = self.tokenizer(
                        sample['dialogue'],
                        return_tensors='pt',
                        max_length=self.max_length,
                        truncation=True,
                        padding=True
                    ).to(self.device)
                    
                    targets = self.tokenizer(
                        sample['summary'],
                        return_tensors='pt',
                        max_length=self.max_length,
                        truncation=True,
                        padding=True
                    ).to(self.device)
                    
                    # Zero gradients
                    self.teacher_model.zero_grad()
                    
                    # Forward pass
                    outputs = self.teacher_model(**inputs, labels=targets['input_ids'])
                    loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Compute gradient norm
                grad_norm = self._compute_gradient_norm()
                batch_norms.append(grad_norm)
                
                # Clear gradients to save memory
                self.teacher_model.zero_grad()
                
            except Exception as e:
                logger.warning(f"Error computing gradient for sample: {e}")
                batch_norms.append(0.0)  # Default to 0 for failed computations
        
        return batch_norms
    
    def _compute_gradient_norm(self) -> float:
        """Compute the L2 norm of gradients across all parameters."""
        total_norm = 0.0
        param_count = 0
        
        for param in self.teacher_model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        # Return normalized gradient norm
        return (total_norm ** 0.5) / max(param_count, 1)
    
    def select_top_samples(self, 
                          dataset,
                          num_samples: int,
                          method: str = 'top_k') -> Tuple[List[int], np.ndarray]:
        """
        Select the most informative samples based on gradient norms.
        
        Args:
            dataset: Full dataset
            num_samples: Number of samples to select
            method: Selection method ('top_k', 'threshold', or 'stratified')
        
        Returns:
            Tuple of (selected_indices, gradient_norms)
        """
        # Compute gradient norms for all samples
        gradient_norms = self.compute_sample_gradient_norms(dataset)
        
        if method == 'top_k':
            return self._select_top_k(gradient_norms, num_samples)
        elif method == 'threshold':
            return self._select_by_threshold(gradient_norms, num_samples)
        elif method == 'stratified':
            return self._select_stratified(gradient_norms, num_samples)
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def _select_top_k(self, gradient_norms: np.ndarray, num_samples: int) -> Tuple[List[int], np.ndarray]:
        """Select top-k samples with highest gradient norms."""
        top_indices = np.argsort(gradient_norms)[-num_samples:].tolist()
        
        logger.info(f"Selected {len(top_indices)} samples with gradient norms "
                   f"ranging from {gradient_norms[top_indices].min():.4f} to {gradient_norms[top_indices].max():.4f}")
        
        return top_indices, gradient_norms
    
    def _select_by_threshold(self, gradient_norms: np.ndarray, num_samples: int) -> Tuple[List[int], np.ndarray]:
        """Select samples above a dynamic threshold to get target number."""
        # Find threshold that gives approximately the desired number
        sorted_indices = np.argsort(gradient_norms)
        threshold_idx = len(sorted_indices) - num_samples
        threshold = gradient_norms[sorted_indices[threshold_idx]]
        
        selected_indices = np.where(gradient_norms >= threshold)[0][:num_samples].tolist()
        
        logger.info(f"Selected {len(selected_indices)} samples above threshold {threshold:.4f}")
        
        return selected_indices, gradient_norms
    
    def _select_stratified(self, gradient_norms: np.ndarray, num_samples: int) -> Tuple[List[int], np.ndarray]:
        """Stratified selection across gradient norm ranges."""
        # Divide into quantiles and sample from each
        n_strata = 5
        k_per_stratum = num_samples // n_strata
        remaining = num_samples % n_strata
        
        selected_indices = []
        for i in range(n_strata):
            start_percentile = i * (100 / n_strata)
            end_percentile = (i + 1) * (100 / n_strata)
            
            start_val = np.percentile(gradient_norms, start_percentile)
            end_val = np.percentile(gradient_norms, end_percentile)
            
            stratum_indices = np.where((gradient_norms >= start_val) & 
                                     (gradient_norms <= end_val))[0]
            
            # Sample k_per_stratum from this stratum (+ 1 for first few strata if remainder)
            k_this_stratum = k_per_stratum + (1 if i < remaining else 0)
            
            if len(stratum_indices) > k_this_stratum:
                stratum_sample = np.random.choice(stratum_indices, k_this_stratum, replace=False)
            else:
                stratum_sample = stratum_indices
            
            selected_indices.extend(stratum_sample.tolist())
        
        logger.info(f"Stratified selection: {len(selected_indices)} samples across {n_strata} strata")
        
        return selected_indices[:num_samples], gradient_norms


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Gradient Norm-based Sample Selection")

    # Dataset parameters
    parser.add_argument("--dataset_name", type=str, default="knkarthick/dialogsum",
                        help="HuggingFace dataset name")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Directory for caching models and data")

    # Teacher model parameters
    parser.add_argument("--teacher_model", type=str, default="microsoft/deberta-xlarge-mnli",
                        help="Teacher model for gradient computation")
    parser.add_argument("--teacher_batch_size", type=int, default=4,
                        help="Batch size for teacher model inference")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for tokenization")

    # Selection parameters
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples to select")
    parser.add_argument("--selection_method", type=str, default="top_k",
                        choices=["top_k", "threshold", "stratified"],
                        help="Selection method to use")

    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./gradient_selection_results",
                        help="Directory to save results")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Specific output file name (optional)")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def save_selection_results(selected_indices: List[int],
                          gradient_norms: np.ndarray,
                          dataset,
                          selection_time: float,
                          args,
                          output_file: str):
    """Save selection results in the required JSON format."""

    # Get selected samples
    selected_samples = dataset.select(selected_indices)

    # Compute metadata
    all_dialogue_lengths = [len(item['dialogue']) for item in dataset]
    all_summary_lengths = [len(item['summary']) for item in dataset]

    # Prepare results dictionary
    results = {
        "selection_method": f"gradient_norm_{args.selection_method}",
        "num_samples": len(selected_indices),
        "selected_indices": selected_indices,
        "selection_time": selection_time,
        "embedding_model": args.teacher_model,  # Using teacher model as the "embedding" model
        "train_dataset_size": len(dataset),
        "selected_dialogues": [sample['dialogue'] for sample in selected_samples],
        "selected_summaries": [sample['summary'] for sample in selected_samples],
        "gradient_norms": gradient_norms[selected_indices].tolist(),
        "metadata": {
            "total_samples": len(dataset),
            "avg_dialogue_length": np.mean(all_dialogue_lengths),
            "avg_summary_length": np.mean(all_summary_lengths),
            "selected_avg_gradient_norm": float(np.mean(gradient_norms[selected_indices])),
            "all_avg_gradient_norm": float(np.mean(gradient_norms)),
            "gradient_norm_improvement": float(np.mean(gradient_norms[selected_indices]) / np.mean(gradient_norms))
        }
    }

    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")
    return results


def main():
    """Main execution function."""
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info("Gradient Norm-based Sample Selection")
    logger.info("=" * 60)
    logger.info(f"Teacher model: {args.teacher_model}")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Samples to select: {args.num_samples}")
    logger.info(f"Selection method: {args.selection_method}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(args.dataset_name, cache_dir=args.cache_dir)
    train_dataset = dataset["train"]
    logger.info(f"Dataset loaded: {len(train_dataset)} samples")

    # Initialize selector
    logger.info("Initializing gradient norm selector...")
    selector = GradientNormSampleSelector(
        teacher_model_name=args.teacher_model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=args.teacher_batch_size,
        max_length=args.max_length,
        cache_dir=args.cache_dir
    )

    # Perform selection
    logger.info("Starting gradient norm-based selection...")
    start_time = time.time()

    selected_indices, gradient_norms = selector.select_top_samples(
        dataset=train_dataset,
        num_samples=args.num_samples,
        method=args.selection_method
    )

    selection_time = time.time() - start_time

    # Prepare output file
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = os.path.join(
            args.output_dir,
            f"selection_results_gradient_norm_{args.selection_method}_{args.num_samples}.json"
        )

    # Save results
    results = save_selection_results(
        selected_indices=selected_indices,
        gradient_norms=gradient_norms,
        dataset=train_dataset,
        selection_time=selection_time,
        args=args,
        output_file=output_file
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("SELECTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(train_dataset)}")
    logger.info(f"Selected samples: {len(selected_indices)}")
    logger.info(f"Selection ratio: {len(selected_indices) / len(train_dataset):.3f}")
    logger.info(f"Selection time: {selection_time:.2f} seconds")
    logger.info(f"Selection method: {args.selection_method}")

    # Gradient statistics
    selected_grad_norms = gradient_norms[selected_indices]
    logger.info(f"All samples - Mean gradient norm: {np.mean(gradient_norms):.4f}")
    logger.info(f"Selected samples - Mean gradient norm: {np.mean(selected_grad_norms):.4f}")
    logger.info(f"Gradient norm improvement: {np.mean(selected_grad_norms) / np.mean(gradient_norms):.2f}x")

    logger.info(f"Results saved to: {output_file}")
    logger.info("Gradient norm selection completed successfully!")

    return results


if __name__ == "__main__":
    main()
