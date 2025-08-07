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

# Add visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
# Add imports for embedding visualization
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
                 batch_size: int = 16,
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
        """Load the teacher model and tokenizer, preparing it for gradient-based analysis."""
        logger.info(f"Loading teacher model: {self.teacher_model_name}")

        # --- Step 1: Load Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.teacher_model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer pad token set to EOS token.")

        # --- Step 2: Load the Model (Quantized or Float) ---
        is_causal_lm = "llama" in self.teacher_model_name.lower() or "gpt" in self.teacher_model_name.lower()

        try:
            if is_causal_lm:
                # Use Quantization + LoRA for large Causal LMs
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.teacher_model_name,
                    cache_dir=self.cache_dir,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info("Loaded base model with 4-bit quantization.")

                # Apply LoRA adapters
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM
                )
                self.teacher_model = get_peft_model(model, lora_config)
                logger.info("Applied LoRA adapters to the quantized model.")
                self.teacher_model.print_trainable_parameters()

            else:
                # For smaller Seq2Seq models, load in float16
                self.teacher_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.teacher_model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info(f"Loaded {self.teacher_model_name} as a float16 seq2seq model.")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # --- Step 3: Set Model to Training Mode ---
        # This is crucial. PEFT handles which params are trainable.
        self.teacher_model.train()
        
        logger.info(f"Model ready for gradient computation on device: {self.teacher_model.device}")
    
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
                # Prepare input based on model type
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
                    # For seq2seq models (BART, T5): dialogue as input, summary as target
                    inputs = self.tokenizer(
                        sample['dialogue'],
                        return_tensors='pt',
                        max_length=self.max_length,
                        truncation=True,
                        padding=True
                    ).to(self.device)

                    # For BART-mnli, we can also format as NLI: premise [SEP] hypothesis
                    if "mnli" in self.teacher_model_name.lower():
                        # Format as NLI task: dialogue entails summary
                        nli_input = f"{sample['dialogue']} </s> {sample['summary']}"
                        inputs = self.tokenizer(
                            nli_input,
                            return_tensors='pt',
                            max_length=self.max_length,
                            truncation=True,
                            padding=True
                        ).to(self.device)

                        # For NLI models, we can use the input as both source and target
                        # This measures how well the model can reconstruct the NLI pair
                        targets = inputs['input_ids'].clone()
                    else:
                        # Standard seq2seq: dialogue -> summary
                        targets = self.tokenizer(
                            sample['summary'],
                            return_tensors='pt',
                            max_length=self.max_length,
                            truncation=True,
                            padding=True
                        ).to(self.device)
                        targets = targets['input_ids']

                    # Zero gradients
                    self.teacher_model.zero_grad()

                    # Forward pass
                    outputs = self.teacher_model(**inputs, labels=targets)
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

    def visualize_gradient_norms(self, gradient_norms: np.ndarray, selected_indices: List[int], output_dir: str):
        """
        Generate visualizations of the gradient norms and selection.
        
        Args:
            gradient_norms: Array of gradient norms for all samples
            selected_indices: Indices of selected samples
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a mask for selected samples
        selected_mask = np.zeros(len(gradient_norms), dtype=bool)
        selected_mask[selected_indices] = True
        
        # 1. Histogram of all gradient norms
        plt.figure(figsize=(10, 6))
        sns.histplot(gradient_norms, bins=50, kde=True)
        plt.title('Distribution of Gradient Norms')
        plt.xlabel('Gradient Norm')
        plt.ylabel('Count')
        plt.axvline(x=np.mean(gradient_norms), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(gradient_norms):.4f}')
        plt.axvline(x=np.median(gradient_norms), color='g', linestyle='--', 
                    label=f'Median: {np.median(gradient_norms):.4f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_norm_distribution.png'), dpi=300)
        plt.close()
        
        # 2. Comparison of selected vs non-selected samples
        plt.figure(figsize=(10, 6))
        sns.kdeplot(gradient_norms[selected_mask], label='Selected', shade=True)
        sns.kdeplot(gradient_norms[~selected_mask], label='Not Selected', shade=True)
        plt.title('Distribution of Gradient Norms: Selected vs. Not Selected')
        plt.xlabel('Gradient Norm')
        plt.ylabel('Density')
        plt.axvline(x=np.mean(gradient_norms[selected_mask]), color='blue', linestyle='--', 
                    label=f'Selected Mean: {np.mean(gradient_norms[selected_mask]):.4f}')
        plt.axvline(x=np.mean(gradient_norms[~selected_mask]), color='orange', linestyle='--', 
                    label=f'Non-Selected Mean: {np.mean(gradient_norms[~selected_mask]):.4f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'selected_vs_nonselected.png'), dpi=300)
        plt.close()
        
        # 3. Scatter plot of gradient norms with selection highlighted
        plt.figure(figsize=(12, 6))
        x = np.arange(len(gradient_norms))
        plt.scatter(x[~selected_mask], gradient_norms[~selected_mask], 
                   alpha=0.5, s=10, label='Not Selected', color='gray')
        plt.scatter(x[selected_mask], gradient_norms[selected_mask], 
                   alpha=0.9, s=30, label='Selected', color='red')
        plt.title('Gradient Norms for Each Sample')
        plt.xlabel('Sample Index')
        plt.ylabel('Gradient Norm')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_norm_selection.png'), dpi=300)
        plt.close()
        
        # 4. Box plot comparison
        plt.figure(figsize=(8, 6))
        data = [
            gradient_norms[~selected_mask],
            gradient_norms[selected_mask]
        ]
        plt.boxplot(data, labels=['Not Selected', 'Selected'])
        plt.title('Gradient Norm Distribution Comparison')
        plt.ylabel('Gradient Norm')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_norm_boxplot.png'), dpi=300)
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")

    def compute_embeddings(self, dataset, subset_size=None):
        """
        Compute embeddings for dialogue samples.
        
        Args:
            dataset: Dataset containing dialogues
            subset_size: Optional size limit for large datasets
            
        Returns:
            Numpy array of embeddings
        """
        if subset_size and subset_size < len(dataset):
            # Use a subset for large datasets
            sample_indices = np.random.choice(len(dataset), subset_size, replace=False)
            compute_dataset = dataset.select(sample_indices)
            mapping = {i: idx for i, idx in enumerate(sample_indices)}
        else:
            compute_dataset = dataset
            mapping = {i: i for i in range(len(dataset))}
            
        logger.info(f"Computing embeddings for {len(compute_dataset)} samples...")
        embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(compute_dataset), self.batch_size), desc="Computing embeddings"):
            batch_end = min(i + self.batch_size, len(compute_dataset))
            batch = compute_dataset.select(range(i, batch_end))
            
            # Get embeddings for this batch
            batch_embeddings = self._compute_batch_embeddings(batch)
            embeddings.extend(batch_embeddings)
        
        # Convert to numpy array and map back to original indices if using subset
        embeddings_array = np.array(embeddings)
        return embeddings_array, mapping
    
    def _compute_batch_embeddings(self, batch):
        """
        Compute embeddings for a batch of samples.
        Uses the mean pooled hidden states as embeddings.
        """
        batch_embeddings = []
        
        # Store original model state
        was_training = self.teacher_model.training
        # Set to eval mode for embedding extraction
        self.teacher_model.eval()
        
        with torch.no_grad():
            for sample in batch:
                try:
                    # Process dialogue text to get embeddings
                    dialogue_text = sample['dialogue']
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        dialogue_text,
                        return_tensors='pt',
                        max_length=self.max_length,
                        truncation=True,
                        padding=True
                    ).to(self.device)
                    
                    # Get model outputs - depends on model type
                    if hasattr(self.teacher_model, 'get_encoder'):
                        # For encoder-decoder models
                        outputs = self.teacher_model.get_encoder()(**inputs)
                        hidden_states = outputs.last_hidden_state
                    else:
                        # For decoder-only models
                        outputs = self.teacher_model(**inputs, output_hidden_states=True)
                        # Use the last hidden state
                        hidden_states = outputs.hidden_states[-1]
                    
                    # Mean pooling to get sentence embedding
                    # Create attention mask
                    attention_mask = inputs['attention_token_type_ids'] if 'attention_token_type_ids' in inputs else inputs['attention_mask']
                    # Apply mask and compute mean
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    embedding = torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    
                    # Convert to numpy and add to batch
                    embedding = embedding.cpu().numpy()[0]  # Take first item as we have batch_size=1
                    batch_embeddings.append(embedding)
                    
                except Exception as e:
                    logger.warning(f"Error computing embedding for sample: {e}")
                    # Return zero vector as fallback
                    if hasattr(self.teacher_model, 'config') and hasattr(self.teacher_model.config, 'hidden_size'):
                        dim = self.teacher_model.config.hidden_size
                    else:
                        dim = 768  # Default dimension
                    batch_embeddings.append(np.zeros(dim))
        
        # Restore original model state
        if was_training:
            self.teacher_model.train()
            
        return batch_embeddings

    def visualize_embeddings(self, embeddings, selected_indices, gradient_norms=None, output_dir=None):
        """
        Visualize embeddings using dimensionality reduction techniques.
        
        Args:
            embeddings: Sample embeddings
            selected_indices: Indices of selected samples
            gradient_norms: Optional gradient norms for color coding
            output_dir: Directory to save visualizations
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Create a mask for selected samples
        all_indices = np.arange(len(embeddings))
        selected_mask = np.isin(all_indices, selected_indices)
        
        # Standardize embeddings
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)
        
        # Create visualization methods
        methods = {
            'PCA': PCA(n_components=2),
            # 'TSNE': TSNE(n_components=2, random_state=42)
        }
        
        for method_name, method in methods.items():
            logger.info(f"Computing {method_name} projection of embeddings...")
            projection = method.fit_transform(scaled_embeddings)
            
            # Create plots
            # 1. Basic plot showing selected vs non-selected
            plt.figure(figsize=(10, 8))
            plt.scatter(projection[~selected_mask, 0], projection[~selected_mask, 1], 
                       alpha=0.5, s=30, label='Not Selected', color='gray')
            plt.scatter(projection[selected_mask, 0], projection[selected_mask, 1], 
                       alpha=0.9, s=50, label='Selected', color='red')
            plt.title(f'{method_name} Projection of Dialogue Embeddings')
            plt.xlabel(f'{method_name} Component 1')
            plt.ylabel(f'{method_name} Component 2')
            plt.legend()
            plt.tight_layout()
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'embedding_{method_name.lower()}_selection.png'), dpi=300)
                plt.close()
            else:
                plt.show()
            
            # 2. Gradient norm colored plot (if provided)
            if gradient_norms is not None:
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(projection[:, 0], projection[:, 1], 
                           c=gradient_norms, alpha=0.7, s=40, cmap='viridis')
                plt.colorbar(scatter, label='Gradient Norm')
                # Highlight selected points
                plt.scatter(projection[selected_mask, 0], projection[selected_mask, 1], 
                          s=80, facecolors='none', edgecolors='red', linewidths=1.5,
                          label='Selected')
                plt.title(f'{method_name} Projection with Gradient Norm Coloring')
                plt.xlabel(f'{method_name} Component 1')
                plt.ylabel(f'{method_name} Component 2')
                plt.legend()
                plt.tight_layout()
                if output_dir:
                    plt.savefig(os.path.join(output_dir, f'embedding_{method_name.lower()}_gradient_colored.png'), dpi=300)
                    plt.close()
                else:
                    plt.show()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Gradient Norm-based Sample Selection")

    # Dataset parameters
    parser.add_argument("--dataset_name", type=str, default="knkarthick/dialogsum",
                        help="HuggingFace dataset name")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Directory for caching models and data")

    # Teacher model parameters
    parser.add_argument("--teacher_model", type=str, default="meta-llama/Llama-2-13b-hf",
                        help="Teacher model for gradient computation")
    parser.add_argument("--teacher_batch_size", type=int, default=16,
                        help="Batch size for teacher model inference")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for tokenization")

    # Selection parameters
    parser.add_argument("--num_samples", type=int, default=1250,
                        help="Number of samples to select")
    parser.add_argument("--selection_method", type=str, default="top_k",
                        choices=["top_k", "threshold", "stratified"],
                        help="Selection method to use")

    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./gradient_selection_results",
                        help="Directory to save results")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Specific output file name (optional)")

    # Add visualization parameter
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of the selection process")
    parser.add_argument("--viz_dir", type=str, default="./gradient_selection_viz",
                        help="Directory to save visualizations")

    # Add embedding visualization parameters
    parser.add_argument("--visualize_embeddings", action="store_true",
                        help="Generate visualizations of the embedding space")
    parser.add_argument("--embedding_subset", type=int, default=1000,
                        help="Number of samples to use for embedding visualization (for large datasets)")

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

    # Generate visualizations if requested
    if args.visualize:
        logger.info("Generating visualizations...")
        viz_dir = args.viz_dir
        selector.visualize_gradient_norms(gradient_norms, selected_indices, viz_dir)

    # Generate embedding visualizations if requested
    if args.visualize_embeddings:
        logger.info("Generating embedding visualizations...")
        
        # Compute embeddings (use subset for efficiency if specified)
        subset_size = min(args.embedding_subset, len(train_dataset)) if args.embedding_subset > 0 else None
        embeddings, mapping = selector.compute_embeddings(train_dataset, subset_size=subset_size)
        
        # Map selected indices to subset indices if using a subset
        if subset_size and subset_size < len(train_dataset):
            # Find which selected indices are in our subset
            selected_in_subset = []
            inverse_mapping = {v: k for k, v in mapping.items()}
            for idx in selected_indices:
                if idx in inverse_mapping:
                    selected_in_subset.append(inverse_mapping[idx])
        else:
            selected_in_subset = selected_indices
        
        # Visualize
        viz_dir = os.path.join(args.viz_dir, "embeddings")
        selector.visualize_embeddings(
            embeddings=embeddings,
            selected_indices=selected_in_subset,
            gradient_norms=gradient_norms if subset_size is None else gradient_norms[[mapping[i] for i in range(len(mapping))]],
            output_dir=viz_dir
        )

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
