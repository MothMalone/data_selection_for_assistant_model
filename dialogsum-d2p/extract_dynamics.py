import os
import torch 
import json
import argparse
import logging
import numpy as np
import wandb
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--dataset_name", type=str, default="knkarthick/dialogsum")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--output_dir", type=str, default="./dynamics")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_epochs", type=int, default=3)
    parser.add_argument("--log_interval", type=int, default=50, help="Log every N steps")
    parser.add_argument("--wandb_project", type=str, default="dialogsum-d2pruning")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_api_key", type=str, default="445d8df72343591d6588f101349ad4752497ce62")

    return parser.parse_args()

def compute_loss_dynamics(model, tokenizer, dataset, batch_size=1, device="cuda"):
    """Compute per-sample loss values (difficulty metric)"""
    losses = []
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Computing losses"):
        batch = dataset[i:min(i + batch_size, len(dataset))]

        input_texts = [f"Dialogue: {d}\nSummary: {s}" for d, s in zip(batch["dialogue"], batch["summary"])]

        encoded = tokenizer(input_texts, padding=True, truncation=True,
                            return_tensors="pt", max_length=1024)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            batch_losses = outputs.loss.item()
            losses.append(batch_losses)

    return np.array(losses)
    
def compute_training_dynamics(model, tokenizer, dataset, num_epochs=3, batch_size=1, device="cuda", log_interval=50):
    """
    Track D2Pruning training dynamics over multiple epochs.
    This simulates the training process to collect metrics for sample difficulty.
    """
    data_size = len(dataset)
    dynamics = {
        'correctness': np.zeros((num_epochs, data_size)),
        'forgetting': np.zeros(data_size),
        'last_correctness': np.zeros(data_size),
        'accumulated_margin': np.zeros(data_size),
        'confidence': np.zeros((num_epochs, data_size)),
        'variance': np.zeros(data_size)
    }

    # Use different learning rates to simulate dynamics over time
    learning_rates = [1e-5, 5e-6, 2e-6]
    
    for epoch in range(num_epochs):
        logger.info(f"Computing dynamics for epoch {epoch+1}/{num_epochs}")
        
        epoch_loss = 0.0
        epoch_forgetting_events = 0
        epoch_confidence = 0.0
        step_count = 0
        
        for i in tqdm(range(0, data_size, batch_size)):
            batch = dataset[i:min(i + batch_size, data_size)]
            step_count += 1

            # Input is dialogue, target is summary (for seq2seq evaluation)
            input_texts = [d for d in batch["dialogue"]]
            target_texts = [s for s in batch["summary"]]

            # Tokenize inputs
            inputs = tokenizer(input_texts, padding=True, truncation=True,
                              return_tensors="pt", max_length=512).to(device)
            
            # Tokenize targets
            targets = tokenizer(target_texts, padding=True, truncation=True,
                              return_tensors="pt", max_length=128).to(device)
                                
            with torch.no_grad():
                # First get loss on dialogue->summary task
                dialogue_inputs = inputs.copy()
                outputs = model(**dialogue_inputs, labels=targets["input_ids"])
                
                batch_loss = outputs.loss.item()
                epoch_loss += batch_loss
                
                # In D2Pruning, confidence is inverse of loss
                # Scale by learning rate to simulate improvement over time
                lr_scale = learning_rates[min(epoch, len(learning_rates)-1)] * 1000
                confidence = 1.0 / (1.0 + batch_loss / lr_scale)
                epoch_confidence += confidence
                
                # Process each sample
                for j in range(len(batch)):
                    idx = i + j
                    if idx >= data_size:
                        break

                    # Store confidence for this epoch
                    dynamics['confidence'][epoch, idx] = confidence
                    
                    # D2Pruning correctness - if loss below threshold
                    threshold = 2.0 - (epoch * 0.3)
                    is_correct = batch_loss < threshold
                    dynamics['correctness'][epoch, idx] = int(is_correct)

                    # Track forgetting events - when correct → incorrect
                    if epoch > 0:
                        if dynamics['correctness'][epoch-1, idx] == 1 and is_correct == 0:
                            dynamics['forgetting'][idx] += 1
                            epoch_forgetting_events += 1

                    # Accumulate margin above baseline
                    dynamics['accumulated_margin'][idx] += max(0, confidence - (0.5 - epoch*0.1))

            # Log every log_interval steps with detailed metrics
            if i % log_interval == 0 and i > 0:
                # Calculate current averages
                avg_loss = epoch_loss / step_count
                avg_confidence = epoch_confidence / step_count
                avg_forgetting = np.mean(dynamics['forgetting'])
                avg_correctness = np.mean(dynamics['correctness'][epoch])
                
                # Log to console
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Step {i}/{data_size}, "
                           f"Avg loss: {avg_loss:.4f}, "
                           f"Avg forgetting: {avg_forgetting:.4f}, "
                           f"Avg confidence: {avg_confidence:.4f}, "
                           f"Avg correctness: {avg_correctness:.4f}")
                
                # Log to wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "step": i,
                    "avg_loss": avg_loss,
                    "avg_forgetting": avg_forgetting,
                    "avg_confidence": avg_confidence,
                    "avg_correctness": avg_correctness,
                    "global_step": epoch * data_size + i
                })

        # Log epoch summary
        avg_epoch_loss = epoch_loss / step_count
        avg_epoch_confidence = epoch_confidence / step_count
        
        logger.info(f"Epoch {epoch+1} summary - "
                   f"Avg loss: {avg_epoch_loss:.4f}, "
                   f"Forgetting events: {epoch_forgetting_events}, "
                   f"Avg confidence: {avg_epoch_confidence:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "epoch_avg_loss": avg_epoch_loss,
            "epoch_forgetting_events": epoch_forgetting_events,
            "epoch_avg_confidence": avg_epoch_confidence,
            "epoch_completed": epoch + 1
        })

    # Store last epoch correctness
    dynamics['last_correctness'] = dynamics['correctness'][-1]

    # Compute variance across epochs
    for i in range(data_size):
        dynamics['variance'][i] = np.var(dynamics['confidence'][:,i])        

    # Convert numpy arrays to lists for JSON serialization
    for key in dynamics:
        if isinstance(dynamics[key], np.ndarray):
            dynamics[key] = dynamics[key].tolist()

    return dynamics

def compute_importance_scores(dynamics, losses):
    """
    Compute D2Pruning importance scores from training dynamics metrics.
    Higher score = more important for training.
    """
    data_size = len(losses)
    scores = np.zeros(data_size)
    
    # D2Pruning weighting scheme
    weights = {
        'loss': 1.0,          # Higher loss = more difficult
        'forgetting': 0.7,    # More forgetting events = challenging example
        'variance': 0.5,      # Higher variance = unstable learning
        'margin': -0.3        # Lower margin = harder example
    }
    
    # Normalize losses to [0,1]
    norm_losses = (losses - np.min(losses)) / (np.max(losses) - np.min(losses) + 1e-8)
    scores += weights['loss'] * norm_losses
    
    # Add forgetting events (key D2Pruning signal)
    forgetting = np.array(dynamics['forgetting'])
    if np.max(forgetting) > 0:
        norm_forgetting = forgetting / np.max(forgetting)
        scores += weights['forgetting'] * norm_forgetting
    
    # Add variance (indicates learning difficulty)
    variance = np.array(dynamics['variance'])
    if np.max(variance) > 0:
        norm_variance = variance / np.max(variance)
        scores += weights['variance'] * norm_variance
    
    # Add accumulated margin (inverse - lower margin means harder example)
    margin = np.array(dynamics['accumulated_margin'])
    if np.max(margin) > 0:
        norm_margin = margin / np.max(margin)
        scores += weights['margin'] * (1.0 - norm_margin)
    
    # Normalize final scores to [0,1] as in D2Pruning
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
    
    return scores

def main():
    args = parse_args()

    # Set up wandb
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or f"dynamics-{args.model_name.split('/')[-1]}-{args.train_epochs}epochs",
        config=vars(args)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train", cache_dir=args.cache_dir)

    logger.info(f"Loading model: {args.model_name}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config, 
        device_map="auto",
        trust_remote_code=True,
        cache_dir=args.cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Computing training dynamics over {args.train_epochs} epochs...")

    dynamics = compute_training_dynamics(
        model, 
        tokenizer,
        dataset,
        num_epochs=args.train_epochs,
        batch_size=args.batch_size,
        log_interval=args.log_interval
    )

    logger.info("Computing loss dynamics...")
    losses = compute_loss_dynamics(model, tokenizer, dataset, args.batch_size)
    
    # Compute D2Pruning importance scores
    importance_scores = compute_importance_scores(dynamics, losses)

    # Log distribution plots to wandb
    wandb.log({
        "forgetting_hist": wandb.Histogram(np.array(dynamics['forgetting'])),
        "variance_hist": wandb.Histogram(np.array(dynamics['variance'])),
        "importance_scores_hist": wandb.Histogram(importance_scores),
        "loss_hist": wandb.Histogram(losses)
    })

    results = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "loss_dynamics": losses.tolist(),
        "training_dynamics": dynamics,
        "importance_scores": importance_scores.tolist()
    }

    output_file = output_dir / 'training_dynamics.json'
    with open(output_file, "w") as f:
        json.dump(results, f)
    
    logger.info(f"Training dynamics saved to {output_file}")
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()