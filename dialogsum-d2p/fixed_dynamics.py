import os
import torch
import json
import argparse
import logging
import numpy as np
import wandb
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Dict, List, Any, Tuple

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--dataset_name", type=str, default="knkarthick/dialogsum")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--output_dir", type=str, default="./dynamics")
    parser.add_argument("--batch_size", type=int, default=4, help="Effective batch size via gradient accumulation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_epochs", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=100, help="Limit samples for faster testing")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--wandb_project", type=str, default="dialogsum-d2pruning")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_api_key", type=str, default=None)

    return parser.parse_args()

def prepare_dataset_for_causal_lm(dataset, tokenizer, max_length=512):
    """Properly format and tokenize the dataset for causal LM training."""
    def tokenize_function(examples):
        # Format input with EOS token
        texts = [f"Dialogue: {d}\nSummary: {s}{tokenizer.eos_token}" 
                for d, s in zip(examples["dialogue"], examples["summary"])]
        
        # Tokenize without padding
        model_inputs = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
            add_special_tokens=True,
            padding=False  # Crucial: padding will be handled in collator
        )
        
        # Create labels by shifting inputs
        model_inputs["labels"] = [ids.copy() for ids in model_inputs["input_ids"]]
        return model_inputs
    
    # Create index column using the original indices
    if "idx" not in dataset.column_names:
        dataset = dataset.add_column("original_idx", range(len(dataset)))
    else:
        dataset = dataset.rename_column("idx", "original_idx")
    
    # Tokenize and remove original columns (keep original_idx)
    columns_to_remove = [col for col in dataset.column_names if col != "original_idx"]
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove,
        desc="Tokenizing dataset"
    )
    
    # Filter out empty sequences and add new index
    tokenized_dataset = tokenized_dataset.filter(
        lambda example: len(example["input_ids"]) > 1,
        desc="Filtering short sequences"
    )
    
    # Create a new sequential index for the filtered dataset
    tokenized_dataset = tokenized_dataset.add_column("idx", range(len(tokenized_dataset)))
    
    logger.info(f"Filtered dataset size: {len(tokenized_dataset)} samples")
    return tokenized_dataset

class D2PruningTracker:
    """Tracks training dynamics robustly, handling dataset shuffling."""
    def __init__(self, dataset_size: int, num_epochs: int):
        self.dataset_size = dataset_size
        self.num_epochs = num_epochs
        self.epoch_losses = np.full((num_epochs, dataset_size), np.nan)
        self.sample_correctness = np.full((num_epochs, dataset_size), np.nan)
        self.current_epoch = -1

    def on_epoch_start(self, epoch: int):
        self.current_epoch = epoch
        logger.info(f"Starting epoch {epoch+1}/{self.num_epochs}")
        
    def record_batch_info(self, sample_indices: List[int], losses: torch.Tensor, 
                      predictions_correct: torch.Tensor):
        if self.current_epoch < 0 or self.current_epoch >= self.num_epochs:
            return
            
        # Convert to numpy arrays
        losses_np = losses.cpu().numpy()
        correct_np = predictions_correct.cpu().numpy()
        
        # Record values for each valid index
        valid_count = 0
        for i, idx in enumerate(sample_indices):
            if 0 <= idx < self.dataset_size:
                self.epoch_losses[self.current_epoch, idx] = losses_np[i]
                self.sample_correctness[self.current_epoch, idx] = correct_np[i]
                valid_count += 1
            else:
                logger.warning(f"Invalid index found: {idx} (dataset size: {self.dataset_size})")
        
        if valid_count == 0:
            logger.error("No valid indices in batch!")

    def compute_final_dynamics(self) -> Dict[str, List[float]]:
        logger.info("Computing final training dynamics...")
        
        # Calculate forgetting events
        forgetting_events = np.zeros(self.dataset_size)
        for i in range(1, self.num_epochs):
            was_correct = self.sample_correctness[i-1, :] == 1
            is_now_incorrect = self.sample_correctness[i, :] == 0
            forgetting_events += (was_correct & is_now_incorrect).astype(int)

        # Calculate confidence using min-max normalization
        min_losses = np.nanmin(self.epoch_losses, axis=0)
        max_losses = np.nanmax(self.epoch_losses, axis=0)
        
        # Handle constant loss cases
        range_losses = np.where(max_losses > min_losses, max_losses - min_losses, 1.0)
        normalized_losses = (self.epoch_losses - min_losses) / range_losses
        confidence = 1.0 - normalized_losses
        
        # Calculate variance across epochs
        confidence_variance = np.nanvar(confidence, axis=0)
        confidence_variance = np.nan_to_num(confidence_variance, nan=0.0)
        
        # Final epoch losses
        final_losses = self.epoch_losses[-1, :]
        final_losses = np.nan_to_num(final_losses, nan=np.nanmean(final_losses))
        
        # Log statistics
        logger.info(f"Final losses - min: {np.min(final_losses):.4f}, max: {np.max(final_losses):.4f}, mean: {np.mean(final_losses):.4f}")
        logger.info(f"Forgetting events - total: {np.sum(forgetting_events)}, mean: {np.mean(forgetting_events):.2f}")
        logger.info(f"Confidence variance - min: {np.min(confidence_variance):.6f}, max: {np.max(confidence_variance):.6f}")
        
        return {
            'forgetting_events': forgetting_events.tolist(),
            'confidence_variance': confidence_variance.tolist(),
            'final_losses': final_losses.tolist(),
        }

class IndexPreservingCollator(DataCollatorForLanguageModeling):
    """Data collator that preserves sample indices during batching."""
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract indices
        indices = [ex.pop("idx") for ex in examples]
        
        # Collate other features
        batch = super().__call__(examples)
        
        # Add indices back to batch
        batch["idx"] = torch.tensor(indices, dtype=torch.long)
        return batch

class DynamicsTrainer(Trainer):
    """Custom trainer to track per-sample training dynamics."""
    def __init__(self, dynamics_tracker: D2PruningTracker, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamics_tracker = dynamics_tracker
        self.batch_counter = 0
        # Initialize loss scale based on your typical loss values
        self.loss_scale = 6.0  # ADD THIS LINE

    def compute_loss(self, model, inputs: Dict[str, torch.Tensor], 
                    return_outputs: bool = False, **kwargs):
        # Extract indices
        sample_indices = inputs.pop("idx").tolist()
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Shift for autoregressive loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        
        # Compute per-token loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        per_token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        ).view(shift_labels.shape)
        
        # Create mask for non-padding tokens
        mask = (shift_labels != self.tokenizer.pad_token_id).float()
        
        # Compute per-sample loss
        sample_losses = []
        predictions_correct = []
        valid_indices = []
        
        for i in range(shift_labels.size(0)):
            # Calculate valid tokens for this sample
            valid_tokens = mask[i].sum().item()
            
            if valid_tokens > 0:
                # Calculate sample loss
                loss_val = (per_token_losses[i] * mask[i]).sum() / valid_tokens

                if self.batch_counter < 5 and sample_losses:
                    logger.info(f"Batch {self.batch_counter} - Sample losses: {sample_losses_t.tolist()}")
                    logger.info(f"Batch {self.batch_counter} - Predictions correct: {predictions_correct_t.tolist()}")
                    logger.info(f"Batch {self.batch_counter} - Loss scale: {self.loss_scale:.4f}")
                        
                sample_losses.append(loss_val)
                
                # ADAPTIVE CORRECTNESS THRESHOLD (MODIFIED)
                # Use 1.5x current loss scale instead of fixed value
                is_correct = 1.0 if loss_val < (self.loss_scale * 1.5) else 0.0
                predictions_correct.append(is_correct)
                valid_indices.append(sample_indices[i])
        
        # Update loss scale if we have valid samples
        if sample_losses:
            sample_losses_t = torch.stack(sample_losses)
            predictions_correct_t = torch.tensor(predictions_correct, device=sample_losses_t.device)
            
            # Update loss scale with exponential moving average
            current_avg_loss = sample_losses_t.mean().item()
            self.loss_scale = 0.9 * self.loss_scale + 0.1 * current_avg_loss
            
            # Record dynamics for valid samples
            self.dynamics_tracker.record_batch_info(valid_indices, sample_losses_t, predictions_correct_t)
            
            # Compute overall loss for backprop
            loss = sample_losses_t.mean()
        else:
            # Handle case with no valid samples
            logger.warning(f"Batch {self.batch_counter} has no valid samples!")
            loss = torch.tensor(0.0, requires_grad=True)
        
        self.batch_counter += 1
        return (loss, outputs) if return_outputs else loss

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.dynamics_tracker.on_epoch_start(state.epoch)
        self.batch_counter = 0

def compute_real_training_dynamics(model, tokenizer, dataset, args):
    """Train the model and collect dynamics using a custom trainer."""
    logger.info("Starting REAL training to collect dynamics...")
    
    # Prepare dataset
    train_dataset = prepare_dataset_for_causal_lm(dataset, tokenizer, args.max_length)
    
    # Initialize tracker
    dynamics_tracker = D2PruningTracker(len(train_dataset), int(args.train_epochs))


    
    # Training configuration
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "temp_training"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.batch_size,
        gradient_checkpointing=True,
        num_train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,
        logging_dir="./logs",
        log_level="warning",
        disable_tqdm=False
    )
    
    # Custom data collator that preserves indices
    data_collator = IndexPreservingCollator(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Create trainer
    trainer = DynamicsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        dynamics_tracker=dynamics_tracker
    )

    # Add this right after creating the trainer
    logger.info("Added debug: Checking first batch")
    first_batch = next(iter(trainer.get_train_dataloader()))
    logger.info(f"First batch indices: {first_batch['idx']}")
    logger.info(f"First batch input shape: {first_batch['input_ids'].shape}")
    
    # Train and collect dynamics
    logger.info("Starting training...")
    trainer.train()
    
    # Compute final dynamics
    dynamics = dynamics_tracker.compute_final_dynamics()
    
    logger.info("Real training dynamics collection completed!")
    return dynamics

def compute_importance_scores(dynamics: Dict[str, List[float]]) -> np.ndarray:
    """Compute D2Pruning importance scores from training dynamics."""
    losses = np.array(dynamics['final_losses'])
    forgetting = np.array(dynamics['forgetting_events'])
    variance = np.array(dynamics['confidence_variance'])
    
    # Handle NaNs more robustly
    def safe_normalize(arr: np.ndarray) -> np.ndarray:
        # Replace NaNs with mean of finite values
        finite_mask = np.isfinite(arr)
        if np.any(finite_mask):
            mean_val = np.mean(arr[finite_mask])
        else:
            mean_val = 0.0
            
        arr = np.where(finite_mask, arr, mean_val)
        
        # Normalize only if there's variation
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-8:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    norm_losses = safe_normalize(losses)
    norm_forgetting = safe_normalize(forgetting)
    norm_variance = safe_normalize(variance)
    
    # Weighted combination
    weights = {'loss': 1.0, 'forgetting': 0.8, 'variance': 0.3}
    scores = (
        weights['loss'] * norm_losses +
        weights['forgetting'] * norm_forgetting +
        weights['variance'] * norm_variance
    )
    
    return safe_normalize(scores)

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Initialize WandB if API key provided
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"d2pruning-{args.model_name.split('/')[-1]}",
            config=vars(args)
        )

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train", cache_dir=args.cache_dir)
    if args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
        logger.info(f"Limited dataset to {args.max_samples} samples.")

    # Configure model quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        cache_dir=args.cache_dir,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare PEFT model
    logger.info("Preparing PEFT model for training")
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Compute training dynamics
    dynamics = compute_real_training_dynamics(model, tokenizer, dataset, args)
    
    # Compute importance scores
    importance_scores = compute_importance_scores(dynamics)
    
    # Save results
    results = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "training_dynamics": dynamics,
        "importance_scores": importance_scores.tolist(),
        "metadata": {
            "num_samples": len(dataset),
            "train_epochs": args.train_epochs,
            "learning_rate": args.learning_rate
        }
    }
    
    output_file = output_dir / f'd2pruning_dynamics_{args.model_name.split("/")[-1]}.json'
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {output_file}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    if args.wandb_api_key:
        wandb.finish()

if __name__ == "__main__":
    main()