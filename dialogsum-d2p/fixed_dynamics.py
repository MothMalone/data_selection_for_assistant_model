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
    DataCollatorForLanguageModeling
)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset_name", type=str, default="knkarthick/dialogsum")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--output_dir", type=str, default="./dynamics")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_epochs", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=1000, help="Limit samples for faster testing")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--wandb_project", type=str, default="dialogsum-d2pruning")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_api_key", type=str, default="your_wandb_api_key_here")

    args, _ = parser.parse_known_args()
    return args

def prepare_dataset_for_causal_lm(dataset, tokenizer, max_length=1024):
    """Properly format DialogSum for causal LM training."""
    original_columns = dataset.column_names

    def tokenize_function(examples):
        texts = [f"Dialogue: {d}\nSummary: {s}{tokenizer.eos_token}" for d, s in zip(examples["dialogue"], examples["summary"])]
        
        model_inputs = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )
        
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        if "idx" in examples:
             model_inputs["idx"] = examples["idx"]
             
        return model_inputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=original_columns
    )
    
    return tokenized_dataset

class D2PruningTracker:
    """Tracks training dynamics robustly, handling dataset shuffling."""
    
    def __init__(self, dataset_size, num_epochs):
        self.dataset_size = dataset_size
        self.num_epochs = num_epochs
        
        self.epoch_losses = np.full((num_epochs, dataset_size), np.nan)
        self.sample_correctness = np.full((num_epochs, dataset_size), np.nan)
        self.current_epoch = -1

    def on_epoch_start(self, epoch):
        self.current_epoch = int(epoch)
        
    def record_batch_info(self, sample_indices, losses, predictions_correct):
        """Record losses and correctness for a batch using original sample indices."""
        if self.current_epoch < 0 or self.current_epoch >= self.num_epochs:
            return
            
        losses_np = losses.cpu().numpy() if torch.is_tensor(losses) else np.array(losses)
        correct_np = predictions_correct.cpu().numpy() if torch.is_tensor(predictions_correct) else np.array(predictions_correct)
        
        self.epoch_losses[self.current_epoch, sample_indices] = losses_np
        self.sample_correctness[self.current_epoch, sample_indices] = correct_np

    def compute_final_dynamics(self):
        """Compute final training dynamics metrics."""
        was_correct = self.sample_correctness[:-1, :] == 1
        is_now_incorrect = self.sample_correctness[1:, :] == 0
        forgetting_events_per_epoch = was_correct & is_now_incorrect
        total_forgetting_events = np.sum(forgetting_events_per_epoch, axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            min_losses = np.nanmin(self.epoch_losses, axis=1, keepdims=True)
            max_losses = np.nanmax(self.epoch_losses, axis=1, keepdims=True)
            range_losses = max_losses - min_losses
            range_losses[range_losses == 0] = 1 
            
            normalized_losses = (self.epoch_losses - min_losses) / range_losses
            confidence_matrix = 1.0 - normalized_losses

        confidence_variance = np.nanvar(confidence_matrix, axis=0)
        total_forgetting_events = np.nan_to_num(total_forgetting_events, nan=0.0)
        confidence_variance = np.nan_to_num(confidence_variance, nan=0.0)
        final_losses = self.epoch_losses[-1, :]
        final_losses = np.nan_to_num(final_losses, nan=np.nanmean(final_losses))

        return {
            'forgetting_events': total_forgetting_events.tolist(),
            'confidence_variance': confidence_variance.tolist(),
            'final_losses': final_losses.tolist(),
        }

def compute_real_training_dynamics(model, tokenizer, dataset, args):
    """Train the model and collect dynamics using a custom trainer."""
    logger.info("Starting REAL training to collect dynamics...")
    
    indexed_dataset = dataset.add_column("idx", range(len(dataset)))
    train_dataset = prepare_dataset_for_causal_lm(indexed_dataset, tokenizer)
    
    num_epochs = int(args.train_epochs)
    dynamics_tracker = D2PruningTracker(len(dataset), num_epochs)
    
    # --- FIX IS HERE ---
    class DynamicsTrainer(Trainer):
        def __init__(self, **kwargs):
            # Pop the custom argument before calling the parent's init
            self.dynamics_tracker = kwargs.pop('dynamics_tracker')
            super().__init__(**kwargs)

        def compute_loss(self, model, inputs, return_outputs=False):
            sample_indices = inputs.pop("idx")
            labels = inputs.get("labels")
            
            outputs = model(**inputs)
            loss = outputs.loss
            
            if labels is not None:
                logits = outputs.get("logits")
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                losses = losses.view(shift_labels.size(0), -1)
                mask = (shift_labels != tokenizer.pad_token_id).float()
                sample_losses = (losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                
                loss_threshold = 2.5
                predictions_correct = (sample_losses < loss_threshold).float()
                
                self.dynamics_tracker.record_batch_info(sample_indices, sample_losses, predictions_correct)

            return (loss, outputs) if return_outputs else loss
    
        def on_epoch_begin(self, args, state, control, **kwargs):
            """Pass the epoch start signal to the tracker."""
            self.dynamics_tracker.on_epoch_start(state.epoch)
    # --- END OF FIX ---

    training_args = TrainingArguments(
        output_dir=args.output_dir + "/temp_training",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    
    trainer = DynamicsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        dynamics_tracker=dynamics_tracker
    )
    
    trainer.train()
    
    dynamics = dynamics_tracker.compute_final_dynamics()
    
    logger.info("Real training dynamics collection completed!")
    return dynamics

def compute_importance_scores(dynamics):
    """Compute D2Pruning importance scores from REAL training dynamics."""
    losses = np.array(dynamics['final_losses'])
    forgetting = np.array(dynamics['forgetting_events'])
    variance = np.array(dynamics['confidence_variance'])
    
    scores = np.zeros(len(losses))
    weights = {'loss': 1.0, 'forgetting': 0.8, 'variance': 0.3}
    
    def normalize(v):
        min_v, max_v = np.min(v), np.max(v)
        return (v - min_v) / (max_v - min_v) if max_v > min_v else np.zeros_like(v)

    if np.any(losses): scores += weights['loss'] * normalize(losses)
    if np.any(forgetting): scores += weights['forgetting'] * normalize(forgetting)
    if np.any(variance): scores += weights['variance'] * normalize(variance)
    
    return normalize(scores)

def main():
    args = parse_args()

    if args.wandb_api_key != "your_wandb_api_key_here":
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"real-dynamics-{args.model_name.split('/')[-1]}-{int(args.train_epochs)}epochs",
            config=vars(args)
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train", cache_dir=args.cache_dir)
    
    if args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
        logger.info(f"Limited dataset to {args.max_samples} samples.")

    logger.info(f"Loading model: {args.model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        cache_dir=args.cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    dynamics = compute_real_training_dynamics(model, tokenizer, dataset, args)
    
    importance_scores = compute_importance_scores(dynamics)

    if wandb.run and wandb.run.id:
        wandb.log({
            "avg_final_loss": np.mean(dynamics['final_losses']),
            "total_forgetting_events": np.sum(dynamics['forgetting_events']),
            "avg_confidence_variance": np.mean(dynamics['confidence_variance']),
        })

    results = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "training_dynamics": dynamics,
        "importance_scores": importance_scores.tolist(),
        "metadata": { "num_samples": len(dataset), "train_epochs": args.train_epochs, "learning_rate": args.learning_rate }
    }

    output_file = output_dir / f'real_training_dynamics_{args.model_name.split("/")[-1]}.json'
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Real training dynamics saved to {output_file}")
    
    if wandb.run and wandb.run.id:
        wandb.finish()

if __name__ == "__main__":
    main()