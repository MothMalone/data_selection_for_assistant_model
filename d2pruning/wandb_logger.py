#!/usr/bin/env python3
"""
Weights & Biases Logger for DialogSum Experiment

This module provides comprehensive logging functionality for the DialogSum data selection
experiment, tracking all relevant metrics throughout the pipeline.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

import wandb
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class DialogSumWandbLogger:
    """Comprehensive W&B logger for DialogSum experiments."""
    
    def __init__(self, api_key: str = None,
                 project_name: str = "dialogsum-benchmark"):
        """
        Initialize wandb logger.

        Args:
            api_key: W&B API key (if None, will try to get from environment)
            project_name: W&B project name
        """
        # Get API key from parameter, environment, or use default
        if api_key is not None:
            self.api_key = api_key
            os.environ["WANDB_API_KEY"] = api_key
        elif "WANDB_API_KEY" in os.environ:
            self.api_key = os.environ["WANDB_API_KEY"]
        else:
            # Use a default key or let wandb handle authentication
            logger.warning("No W&B API key provided. Using environment or wandb login.")
            self.api_key = None

        self.project_name = project_name
        self.run = None
        self.experiment_start_time = None
        
    def init_experiment(self, config: Dict[str, Any], experiment_name: str = None) -> None:
        """Initialize a new wandb experiment run."""
        self.experiment_start_time = time.time()
        
        # Create experiment name if not provided
        if experiment_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            experiment_name = f"dialogsum_selection_{timestamp}"
        
        # Initialize wandb run with error handling
        try:
            self.run = wandb.init(
                project=self.project_name,
                name=experiment_name,
                config=config,
                tags=["dialogsum", "data-selection", "llama", "summarization"],
                notes="DialogSum data selection experiment using statistical methods from d2pruning",
                reinit=True  # Allow multiple runs in same process
            )

            logger.info(f"Initialized W&B run: {self.run.name}")
            logger.info(f"W&B URL: {self.run.url}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            logger.warning("Continuing without W&B logging")
            self.run = None

    def _safe_log(self, metrics: Dict[str, Any]) -> None:
        """Safely log metrics to wandb with error handling."""
        if self.run is None:
            logger.debug("W&B not initialized, skipping logging")
            return

        try:
            wandb.log(metrics)
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")
            logger.debug(f"Metrics that failed to log: {metrics}")

    def log_dataset_info(self, train_size: int, test_size: int,
                        avg_dialogue_length: float, avg_summary_length: float) -> None:
        """Log dataset information."""
        dataset_metrics = {
            "dataset/train_size": train_size,
            "dataset/test_size": test_size,
            "dataset/avg_dialogue_length": avg_dialogue_length,
            "dataset/avg_summary_length": avg_summary_length,
            "dataset/dialogue_to_summary_ratio": avg_dialogue_length / avg_summary_length
        }

        self._safe_log(dataset_metrics)
        logger.info("Logged dataset information to W&B")
    
    def log_embedding_info(self, embedding_shape: tuple, embedding_model: str,
                          extraction_time: float) -> None:
        """Log embedding extraction information."""
        embedding_metrics = {
            "embeddings/num_samples": embedding_shape[0],
            "embeddings/embedding_dim": embedding_shape[1],
            "embeddings/model_name": embedding_model,
            "embeddings/extraction_time": extraction_time,
            "embeddings/samples_per_second": embedding_shape[0] / extraction_time
        }
        
        self._safe_log(embedding_metrics)
        logger.info("Logged embedding information to W&B")
    
    def log_selection_results(self, selection_method: str, selected_indices: List[int],
                            selection_time: float, total_samples: int,
                            diversity_scores: Optional[np.ndarray] = None) -> None:
        """Log data selection results."""
        selection_metrics = {
            f"selection/{selection_method}/num_selected": len(selected_indices),
            f"selection/{selection_method}/selection_time": selection_time,
            f"selection/{selection_method}/selection_ratio": len(selected_indices) / total_samples,
            f"selection/{selection_method}/samples_per_second": len(selected_indices) / selection_time
        }
        
        # Log diversity metrics if available
        if diversity_scores is not None:
            selected_diversity = diversity_scores[selected_indices]
            selection_metrics.update({
                f"selection/{selection_method}/avg_diversity": np.mean(selected_diversity),
                f"selection/{selection_method}/std_diversity": np.std(selected_diversity),
                f"selection/{selection_method}/min_diversity": np.min(selected_diversity),
                f"selection/{selection_method}/max_diversity": np.max(selected_diversity)
            })
        
        self._safe_log(selection_metrics)
        
        # Create and log selection distribution histogram
        if diversity_scores is not None:
            self._safe_log({
                f"selection/{selection_method}/diversity_distribution": wandb.Histogram(selected_diversity),
                f"selection/{selection_method}/all_diversity_distribution": wandb.Histogram(diversity_scores)
            })
        
        logger.info(f"Logged selection results for {selection_method} to W&B")
    
    def log_training_start(self, selection_method: str, num_samples: int, 
                          model_name: str, format_type: str, training_config: Dict[str, Any]) -> None:
        """Log training start information."""
        training_start_metrics = {
            f"training/{selection_method}/start_time": time.time(),
            f"training/{selection_method}/num_samples": num_samples,
            f"training/{selection_method}/model_name": model_name,
            f"training/{selection_method}/format_type": format_type,
            f"training/{selection_method}/num_epochs": training_config.get("num_epochs", 0),
            f"training/{selection_method}/learning_rate": training_config.get("learning_rate", 0),
            f"training/{selection_method}/batch_size": training_config.get("batch_size", 0)
        }
        
        wandb.log(training_start_metrics)
        logger.info(f"Logged training start for {selection_method} to W&B")
    
    def log_training_metrics(self, selection_method: str, epoch: int, step: int,
                           loss: float, learning_rate: float, grad_norm: Optional[float] = None) -> None:
        """Log training metrics during fine-tuning."""
        training_metrics = {
            f"training/{selection_method}/epoch": epoch,
            f"training/{selection_method}/step": step,
            f"training/{selection_method}/loss": loss,
            f"training/{selection_method}/learning_rate": learning_rate,
            f"training/{selection_method}/loss_smooth": loss,  # For smoothed plotting
        }

        if grad_norm is not None:
            training_metrics[f"training/{selection_method}/grad_norm"] = grad_norm

        wandb.log(training_metrics)

    def log_evaluation_progress(self, selection_method: str, current_samples: int,
                              total_samples: int, rouge_scores: Optional[Dict[str, float]] = None) -> None:
        """Log evaluation progress during generation."""
        progress_metrics = {
            f"evaluation/{selection_method}/progress_samples": current_samples,
            f"evaluation/{selection_method}/total_samples": total_samples,
            f"evaluation/{selection_method}/completion_pct": (current_samples / total_samples) * 100
        }

        if rouge_scores:
            progress_metrics.update({
                f"evaluation/{selection_method}/progress_rouge1": rouge_scores["rouge1"],
                f"evaluation/{selection_method}/progress_rouge2": rouge_scores["rouge2"],
                f"evaluation/{selection_method}/progress_rougeL": rouge_scores["rougeL"],
            })

        wandb.log(progress_metrics)
    
    def log_training_completion(self, selection_method: str, training_time: float,
                              final_loss: float, num_parameters: int,
                              trainable_parameters: int) -> None:
        """Log training completion metrics."""
        completion_metrics = {
            f"training/{selection_method}/total_time": training_time,
            f"training/{selection_method}/final_loss": final_loss,
            f"training/{selection_method}/total_parameters": num_parameters,
            f"training/{selection_method}/trainable_parameters": trainable_parameters,
            f"training/{selection_method}/trainable_ratio": trainable_parameters / num_parameters,
            f"training/{selection_method}/time_per_epoch": training_time / wandb.config.get("num_epochs", 1),
            f"training/{selection_method}/status": "completed"
        }

        wandb.log(completion_metrics)

        # Create a summary table for this training run
        training_summary = wandb.Table(
            columns=["Metric", "Value"],
            data=[
                ["Training Time (s)", f"{training_time:.2f}"],
                ["Final Loss", f"{final_loss:.4f}"],
                ["Total Parameters", f"{num_parameters:,}"],
                ["Trainable Parameters", f"{trainable_parameters:,}"],
                ["Trainable Ratio", f"{trainable_parameters/num_parameters:.4f}"],
                ["Time per Epoch (s)", f"{training_time / wandb.config.get('num_epochs', 1):.2f}"]
            ]
        )

        wandb.log({f"training/{selection_method}/summary_table": training_summary})
        logger.info(f"Logged training completion for {selection_method} to W&B")
    
    def log_evaluation_start(self, selection_method: str, num_test_samples: int) -> None:
        """Log evaluation start information."""
        eval_start_metrics = {
            f"evaluation/{selection_method}/start_time": time.time(),
            f"evaluation/{selection_method}/num_test_samples": num_test_samples
        }
        
        wandb.log(eval_start_metrics)
        logger.info(f"Logged evaluation start for {selection_method} to W&B")
    
    def log_evaluation_results(self, selection_method: str, rouge_scores: Dict[str, float],
                             generation_time: float, num_test_samples: int,
                             sample_predictions: List[str] = None,
                             sample_references: List[str] = None) -> None:
        """Log evaluation results."""
        eval_metrics = {
            f"evaluation/{selection_method}/rouge1": rouge_scores["rouge1"],
            f"evaluation/{selection_method}/rouge2": rouge_scores["rouge2"],
            f"evaluation/{selection_method}/rougeL": rouge_scores["rougeL"],
            f"evaluation/{selection_method}/rougeLsum": rouge_scores["rougeLsum"],
            f"evaluation/{selection_method}/generation_time": generation_time,
            f"evaluation/{selection_method}/samples_per_second": num_test_samples / generation_time,
            f"evaluation/{selection_method}/avg_generation_time_per_sample": generation_time / num_test_samples
        }
        
        wandb.log(eval_metrics)
        
        # Log sample predictions as a table
        if sample_predictions and sample_references:
            sample_table = wandb.Table(
                columns=["Sample_ID", "Reference", "Prediction", "Reference_Length", "Prediction_Length"],
                data=[
                    [i, ref[:200] + "..." if len(ref) > 200 else ref, 
                     pred[:200] + "..." if len(pred) > 200 else pred,
                     len(ref.split()), len(pred.split())]
                    for i, (ref, pred) in enumerate(zip(sample_references[:10], sample_predictions[:10]))
                ]
            )
            wandb.log({f"evaluation/{selection_method}/sample_predictions": sample_table})
        
        logger.info(f"Logged evaluation results for {selection_method} to W&B")
    
    def log_comparison_summary(self, comparison_results: Dict[str, Dict[str, Any]]) -> None:
        """Log final comparison summary across all methods."""
        # Create comparison table
        comparison_data = []
        for method, results in comparison_results.items():
            comparison_data.append([
                method,
                results["selection_method"],
                results["num_samples"],
                results["rouge1"],
                results["rouge2"],
                results["rougeL"],
                results["training_time"],
                results["generation_time"]
            ])
        
        comparison_table = wandb.Table(
            columns=["Method_Key", "Selection_Method", "Num_Samples", "ROUGE-1", 
                    "ROUGE-2", "ROUGE-L", "Training_Time", "Generation_Time"],
            data=comparison_data
        )
        
        wandb.log({"comparison/summary_table": comparison_table})
        
        # Log best performing method
        best_method = max(comparison_results.items(), key=lambda x: x[1]["rouge1"])
        wandb.log({
            "comparison/best_method": best_method[0],
            "comparison/best_rouge1": best_method[1]["rouge1"],
            "comparison/best_rouge2": best_method[1]["rouge2"],
            "comparison/best_rougeL": best_method[1]["rougeL"]
        })
        
        # Log method rankings
        methods_by_rouge1 = sorted(comparison_results.items(), key=lambda x: x[1]["rouge1"], reverse=True)
        for rank, (method, _) in enumerate(methods_by_rouge1, 1):
            wandb.log({f"comparison/rouge1_ranking/{method}": rank})
        
        logger.info("Logged comparison summary to W&B")
    
    def log_experiment_completion(self, total_experiment_time: float,
                                 successful_methods: List[str],
                                 failed_methods: List[str] = None) -> None:
        """Log experiment completion metrics."""
        completion_metrics = {
            "experiment/total_time": total_experiment_time,
            "experiment/successful_methods": len(successful_methods),
            "experiment/failed_methods": len(failed_methods) if failed_methods else 0,
            "experiment/success_rate": len(successful_methods) / (len(successful_methods) + len(failed_methods or []))
        }
        
        wandb.log(completion_metrics)
        
        # Log method success/failure
        for method in successful_methods:
            wandb.log({f"experiment/method_success/{method}": 1})
        
        if failed_methods:
            for method in failed_methods:
                wandb.log({f"experiment/method_success/{method}": 0})
        
        logger.info("Logged experiment completion to W&B")
    
    def log_image(self, image_path: str, caption: str = None) -> None:
        """Log an image to W&B."""
        if self.run is None:
            logger.debug("W&B not initialized, skipping image logging")
            return
            
        try:
            image_path = Path(image_path)
            if image_path.exists():
                self._safe_log({
                    f"visualization/{image_path.stem}": wandb.Image(str(image_path), caption=caption)
                })
                logger.info(f"Logged image {image_path} to W&B")
            else:
                logger.warning(f"Image file {image_path} does not exist, skipping")
        except Exception as e:
            logger.warning(f"Failed to log image to W&B: {e}")
            
    def save_artifacts(self, file_paths: Dict[str, str]) -> None:
        """Save important files as W&B artifacts."""
        for artifact_name, file_path in file_paths.items():
            if Path(file_path).exists():
                artifact = wandb.Artifact(artifact_name, type="result")
                artifact.add_file(file_path)
                wandb.log_artifact(artifact)
                logger.info(f"Saved artifact {artifact_name}: {file_path}")
    
    def finish_experiment(self) -> None:
        """Finish the wandb run."""
        if self.run:
            # Log total experiment time
            if self.experiment_start_time:
                total_time = time.time() - self.experiment_start_time
                wandb.log({"experiment/total_duration": total_time})
            
            wandb.finish()
            logger.info("Finished W&B run")
    
    def create_summary_csv(self, comparison_results: Dict[str, Dict[str, Any]], 
                          output_path: str) -> None:
        """Create and save a CSV summary of results."""
        # Prepare data for CSV
        csv_data = []
        for method_key, results in comparison_results.items():
            csv_data.append({
                "method_key": method_key,
                "selection_method": results["selection_method"],
                "num_samples": results["num_samples"],
                "rouge1": results["rouge1"],
                "rouge2": results["rouge2"],
                "rougeL": results["rougeL"],
                "training_time": results["training_time"],
                "generation_time": results["generation_time"],
                "total_time": results["training_time"] + results["generation_time"]
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df = df.sort_values("rouge1", ascending=False)  # Sort by ROUGE-1 score
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved CSV summary to {output_path}")
        
        # Also log as W&B table
        wandb.log({"results/csv_summary": wandb.Table(dataframe=df)})
        
    def finish(self):
        """Finish the wandb run."""
        if self.run:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"Error finishing W&B run: {e}")
                # Try alternative ways to finish
                try:
                    if hasattr(self.run, 'finish'):
                        self.run.finish()
                except:
                    logger.warning("Could not finish W&B run properly")


def setup_wandb_logging(config: Dict[str, Any], experiment_name: str = None) -> DialogSumWandbLogger:
    """
    Set up W&B logging for the experiment.
    
    Args:
        config: Experiment configuration
        experiment_name: Optional experiment name
        
    Returns:
        Configured DialogSumWandbLogger instance
    """
    wandb_logger = DialogSumWandbLogger()
    wandb_logger.init_experiment(config, experiment_name)
    return wandb_logger
