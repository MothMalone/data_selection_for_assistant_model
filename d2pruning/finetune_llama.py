#!/usr/bin/env python3
"""
LLaMA-7B Fine-tuning for DialogSum

This script fine-tunes LLaMA-7B on selected DialogSum samples for summarization.
Supports both instruction-based and chat-based formatting.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    set_seed,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import wandb

# Set up logging to both file and console
log_dir = Path("../logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"finetune_{time.strftime('%Y%m%d_%H%M%S')}.log"

handlers = [
    logging.StreamHandler(),  # Console handler
    logging.FileHandler(log_file)  # File handler
]

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=handlers
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to {log_file}")


class WandbLoggingCallback(TrainerCallback):
    """Enhanced callback for detailed training metrics logging to W&B."""

    def __init__(self, wandb_logger, selection_method):
        self.wandb_logger = wandb_logger
        self.selection_method = selection_method
        self.step_count = 0

    def on_train_begin(self, args, state, control, **kwargs):
        """Log training start."""
        if self.wandb_logger:
            import wandb
            wandb.log({
                f"training/{self.selection_method}/status": "started",
                f"training/{self.selection_method}/total_steps": state.max_steps,
                f"training/{self.selection_method}/num_epochs": args.num_train_epochs
            })

    def on_step_end(self, args, state, control, **kwargs):
        """Log after each training step."""
        self.step_count += 1
        if self.wandb_logger and hasattr(kwargs.get('logs', {}), 'get'):
            logs = kwargs.get('logs', {})
            if logs:
                import wandb
                wandb.log({
                    f"training/{self.selection_method}/step": state.global_step,
                    f"training/{self.selection_method}/epoch": state.epoch,
                    f"training/{self.selection_method}/progress": state.global_step / state.max_steps
                })

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log detailed training metrics to W&B."""
        if logs and self.wandb_logger:
            import wandb

            # Prepare metrics for logging
            metrics_to_log = {}

            if "train_loss" in logs:
                metrics_to_log.update({
                    f"training/{self.selection_method}/loss": logs["train_loss"],
                    f"training/{self.selection_method}/step": state.global_step,
                    f"training/{self.selection_method}/epoch": state.epoch,
                })

            if "learning_rate" in logs:
                metrics_to_log[f"training/{self.selection_method}/learning_rate"] = logs["learning_rate"]

            if "grad_norm" in logs:
                metrics_to_log[f"training/{self.selection_method}/grad_norm"] = logs["grad_norm"]

            if "train_runtime" in logs:
                metrics_to_log[f"training/{self.selection_method}/runtime"] = logs["train_runtime"]

            if "train_samples_per_second" in logs:
                metrics_to_log[f"training/{self.selection_method}/samples_per_second"] = logs["train_samples_per_second"]

            # Log all metrics at once
            if metrics_to_log:
                wandb.log(metrics_to_log)

            # Also use the original wandb_logger method
            if "train_loss" in logs:
                self.wandb_logger.log_training_metrics(
                    selection_method=self.selection_method,
                    epoch=state.epoch,
                    step=state.global_step,
                    loss=logs["train_loss"],
                    learning_rate=logs.get("learning_rate", 0),
                    grad_norm=logs.get("grad_norm", None)
                )


class DialogSumDataFormatter:
    """Handles data formatting for different fine-tuning approaches."""
    
    def __init__(self, format_type: str = "instruction"):
        self.format_type = format_type
        
    def format_instruction(self, dialogue: str, summary: str = None) -> str:
        """Format as instruction-following task."""
        instruction = "Summarize the following dialogue:"
        
        if summary is not None:
            # Training format
            return f"### Instruction:\n{instruction}\n\n### Input:\n{dialogue}\n\n### Response:\n{summary}"
        else:
            # Inference format
            return f"### Instruction:\n{instruction}\n\n### Input:\n{dialogue}\n\n### Response:\n"
    
    def format_chat(self, dialogue: str, summary: str = None) -> str:
        """Format as LLaMA-2 chat conversation."""
        system_msg = "You are a helpful assistant that summarizes dialogues."
        user_msg = f"Please summarize this dialogue:\n\n{dialogue}"

        if summary is not None:
            # Training format - LLaMA-2 Chat format
            return f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_msg} [/INST] {summary} </s>"
        else:
            # Inference format
            return f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_msg} [/INST] "

    def format_alpaca(self, dialogue: str, summary: str = None) -> str:
        """Format as Alpaca instruction format (optimal for LLaMA)."""
        instruction = "Summarize the following dialogue:"

        if summary is not None:
            # Training format
            return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{dialogue}\n\n### Response:\n{summary}"
        else:
            # Inference format
            return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{dialogue}\n\n### Response:\n"

    def format_vicuna(self, dialogue: str, summary: str = None) -> str:
        """Format as Vicuna conversation format."""
        if summary is not None:
            # Training format
            return f"USER: Summarize the following dialogue:\n\n{dialogue}\n\nASSISTANT: {summary}"
        else:
            # Inference format
            return f"USER: Summarize the following dialogue:\n\n{dialogue}\n\nASSISTANT: "
    
    def format_sample(self, dialogue: str, summary: str = None) -> str:
        """Format a single sample based on the chosen format type."""
        if self.format_type == "instruction":
            return self.format_instruction(dialogue, summary)
        elif self.format_type == "chat":
            return self.format_chat(dialogue, summary)
        elif self.format_type == "alpaca":
            return self.format_alpaca(dialogue, summary)
        elif self.format_type == "vicuna":
            return self.format_vicuna(dialogue, summary)
        else:
            raise ValueError(f"Unknown format type: {self.format_type}")


class LlamaFineTuner:
    """Handles LLaMA-7B fine-tuning with LoRA."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-13b-hf", cache_dir: str = "./cache"):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.tokenizer = None
        self.model = None
        
    def load_model_and_tokenizer(self):
        """Load LLaMA model and tokenizer."""
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Configure quantization properly
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Load model with 4-bit quantization for memory efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def prepare_dataset(self, dialogues: List[str], summaries: List[str],
                       formatter: DialogSumDataFormatter) -> Dataset:
        """Prepare dataset for training."""
        formatted_texts = []

        for dialogue, summary in zip(dialogues, summaries):
            formatted_text = formatter.format_sample(dialogue, summary)
            formatted_texts.append(formatted_text)

        # Tokenize the data
        def tokenize_function(examples):
            # Tokenize and set labels for causal language modeling
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,  # Don't pad here, let the data collator handle it
                max_length=512,
                return_overflowing_tokens=False,
            )
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        dataset = Dataset.from_dict({"text": formatted_texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]  # Remove the original text column
        )

        return tokenized_dataset
    
    def train(self, train_dataset: Dataset, output_dir: str,
              num_epochs: int = 3, learning_rate: float = 2e-4,
              wandb_logger=None, selection_method: str = None) -> Dict[str, Any]:
        """Fine-tune the model."""
        logger.info("Starting fine-tuning...")

        # Log training start to W&B
        if wandb_logger and selection_method:
            training_config = {
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": 1,
                "gradient_accumulation_steps": 4
            }
            wandb_logger.log_training_start(
                selection_method=selection_method,
                num_samples=len(train_dataset),
                model_name=self.model_name,
                format_type="instruction",  # Default, could be parameterized
                training_config=training_config
            )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,  # Log every step for detailed tracking
            logging_first_step=True,
            save_strategy="epoch",
            eval_strategy="no",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=["wandb"] if wandb_logger else None,  # Enable W&B reporting
            run_name=f"{selection_method}_{len(train_dataset)}samples" if wandb_logger else None,
            log_level="info",
            logging_nan_inf_filter=False,  # Log all values including NaN/inf
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,  # For efficiency
        )

        # Set up callbacks
        callbacks = []
        if wandb_logger and selection_method:
            callbacks.append(WandbLoggingCallback(wandb_logger, selection_method))

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time

        # Get final training metrics
        final_loss = trainer.state.log_history[-1].get("train_loss", 0.0) if trainer.state.log_history else 0.0

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Log training completion to W&B
        if wandb_logger and selection_method:
            wandb_logger.log_training_completion(
                selection_method=selection_method,
                training_time=training_time,
                final_loss=final_loss,
                num_parameters=total_params,
                trainable_parameters=trainable_params
            )

        # Save the model
        trainer.save_model()

        logger.info(f"Fine-tuning completed in {training_time:.2f} seconds")

        return {
            "training_time": training_time,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "num_samples": len(train_dataset),
            "final_loss": final_loss,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
    
    def generate_summary(self, dialogue: str, formatter: DialogSumDataFormatter, 
                        max_length: int = 150) -> str:
        """Generate summary for a given dialogue."""
        prompt = formatter.format_sample(dialogue)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated summary part
        if "### Response:" in generated_text:
            summary = generated_text.split("### Response:")[-1].strip()
        elif "<|assistant|>" in generated_text:
            summary = generated_text.split("<|assistant|>")[-1].strip()
        else:
            summary = generated_text[len(prompt):].strip()
        
        return summary


def load_selection_results(results_file: str) -> Dict[str, Any]:
    """Load results from data selection."""
    with open(results_file, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="LLaMA-7B Fine-tuning for DialogSum")
    parser.add_argument("--selection_results", type=str, required=True,
                       help="Path to selection results JSON file")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-13b-hf",
                       help="LLaMA model name")
    parser.add_argument("--format_type", type=str, default="alpaca",
                       choices=["instruction", "chat", "alpaca", "vicuna"],
                       help="Data formatting approach")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--enable_wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--wandb_project", type=str, default="data-selection-experiments", help="W&B project name.")
    parser.add_argument("--wandb_run_group", type=str, default=None, help="W&B run group name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory for caching models and data.")
    parser.add_argument("--output_dir", type=str, default="./finetuned_models", help="Directory to save fine-tuned models.")

    args = parser.parse_args()

    set_seed(args.seed)

    # Load selection results
    logger.info(f"Loading selection results from {args.selection_results}")
    with open(args.selection_results, 'r') as f:
        selection_data = json.load(f)

    selection_method = selection_data['selection_method']
    num_samples = selection_data['num_samples']
    
    # Initialize W&B
    wandb_logger = None
    if args.enable_wandb:
        try:
            # Try to import setup_wandb_logging from wandb_logger
            sys.path.append(os.path.dirname(__file__))
            from wandb_logger import setup_wandb_logging
            
            # Use the run name if provided or create a default one
            run_name = args.wandb_run_name or f"finetune_{selection_method}_{num_samples}_seed{args.seed}"
            
            # Setup wandb config
            wandb_config = {
                **vars(args),
                "selection_method": selection_method,
                "num_samples": num_samples,
                "phase": "fine-tuning"
            }
            
            # Initialize wandb logger
            wandb_logger = setup_wandb_logging(
                config=wandb_config,
                experiment_name=run_name
            )
            
            logger.info(f"W&B logging initialized with run name: {run_name}")
            
        except ImportError:
            logger.warning("Could not import wandb_logger. Running without W&B logging.")
            
            # Fall back to basic wandb init
            run_name = f"finetune_{selection_method}_{num_samples}_seed{args.seed}"
            wandb.init(
                project=args.wandb_project,
                group=args.wandb_run_group,
                name=run_name,
                config=vars(args)
            )

    # Get dialogues and summaries from selection data
    dialogues = selection_data['selected_dialogues']
    summaries = selection_data['selected_summaries']
    
    # Create a Hugging Face dataset from the selected data
    dataset = Dataset.from_dict({
        'dialogue': dialogues,
        'summary': summaries
    })

    # Initialize components
    formatter = DialogSumDataFormatter(format_type=args.format_type)
    finetuner = LlamaFineTuner(model_name=args.model_name, cache_dir=args.cache_dir)
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"{selection_method}_{num_samples}samples_{args.format_type}"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model and tokenizer
    finetuner.load_model_and_tokenizer()
    
    # Prepare dataset
    logger.info("Preparing training dataset...")
    train_dataset = finetuner.prepare_dataset(dialogues, summaries, formatter)
    
    # Fine-tune the model
    training_results = finetuner.train(
        train_dataset=train_dataset,
        output_dir=str(output_dir),
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        wandb_logger=wandb_logger,
        selection_method=selection_method
    )
    
    # Save training results
    results = {
        "selection_method": selection_method,
        "num_samples": len(dialogues),
        "format_type": args.format_type,
        "model_name": args.model_name,
        "training_results": training_results,
        "output_dir": str(output_dir)
    }
    
    results_file = output_dir / "training_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Finish W&B logging
    if wandb_logger:
        wandb_logger.finish_experiment()

    logger.info(f"Training results saved to {results_file}")
    logger.info("Fine-tuning completed successfully!")
    
    # The model is already saved by the trainer in the output_dir
    if args.enable_wandb:
        wandb.finish()

    logger.info("Fine-tuning complete.")

if __name__ == "__main__":
    main()
