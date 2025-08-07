#!/usr/bin/env python3
"""
DialogSum Data Selection Script

This script implements various data selection methods for the DialogSum dataset:
- Random selection
- Moderate selection (balanced difficulty)
- K-center selection (diversity-based)
- Diversity selection (embedding-based)

Based on the d2pruning framework for data selection.
"""

import os
import sys
import json
import time
import pickle
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
from core.data.sampling import kCenterGreedy, GraphDensitySampler

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class DialogSumSelector:
    """Main class for DialogSum data selection using various methods."""

    def __init__(self, cache_dir: str = "./cache", seed: int = 42):
        self.cache_dir = cache_dir
        self.seed = seed
        os.makedirs(self.cache_dir, exist_ok=True)
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize embedding model
        self.embedding_model = None

    def load_dataset(self, dataset_name: str = "knkarthick/dialogsum"):
        """Load the DialogSum dataset."""
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, cache_dir=self.cache_dir)
        logger.info(f"Dataset loaded: {len(dataset['train'])} training samples")
        return dataset

    def get_embeddings(self, texts: List[str], model_name: str = "meta-llama/Llama-2-13b-hf") -> np.ndarray:
        """Get embeddings for texts using sentence transformers."""
        cache_file = os.path.join(
            self.cache_dir,
            f"embeddings_{model_name.replace('/', '_')}.npy"
        )

        if os.path.exists(cache_file):
            logger.info(f"Loading cached embeddings from {cache_file}")
            return np.load(cache_file)

        logger.info(f"Computing embeddings using {model_name}")
        if self.embedding_model is None or self.embedding_model.model_name != model_name:
            self.embedding_model = SentenceTransformer(model_name, cache_folder=self.cache_dir)

        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Cache embeddings
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.save(cache_file, embeddings)
        logger.info(f"Cached embeddings to {cache_file}")

        return embeddings

    def select_random(self, dataset_size: int, num_samples: int) -> List[int]:
        """Random selection baseline."""
        logger.info(f"Selecting {num_samples} samples randomly from {dataset_size}")
        np.random.seed(self.seed)
        selected_indices = np.random.choice(dataset_size, num_samples, replace=False)
        return selected_indices.tolist()

    def select_diversity(self, embeddings: np.ndarray, num_samples: int) -> List[int]:
        """Diversity-based selection using embedding clustering."""
        logger.info(f"Selecting {num_samples} samples using diversity-based method")

        # Use k-means clustering to find diverse samples
        n_clusters = min(num_samples, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Select one sample from each cluster (closest to centroid)
        selected_indices = []
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) > 0:
                # Find sample closest to cluster center
                cluster_embeddings = embeddings[cluster_indices]
                center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(cluster_embeddings - center, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(closest_idx)

        # If we need more samples, add random ones
        if len(selected_indices) < num_samples:
            remaining_indices = list(set(range(len(embeddings))) - set(selected_indices))
            additional = np.random.choice(
                remaining_indices,
                num_samples - len(selected_indices),
                replace=False
            )
            selected_indices.extend(additional.tolist())

        return selected_indices[:num_samples]

    def select_k_center(self, embeddings: np.ndarray, num_samples: int) -> List[int]:
        """K-center greedy selection for diversity."""
        logger.info(f"Selecting {num_samples} samples using k-center greedy method")

        # Use the existing kCenterGreedy implementation
        sampler = kCenterGreedy(X=embeddings, y=None, seed=self.seed)
        selected_indices = sampler.select_batch_(already_selected=None, N=num_samples)
    
        return selected_indices

    def select_moderate(self, embeddings: np.ndarray, importance_scores: np.ndarray, num_samples: int) -> List[int]:
        """Moderate selection balancing difficulty and diversity."""
        logger.info(f"Selecting {num_samples} samples using moderate method")

        # Compute distances from mean embedding (proxy for difficulty)
        mean_embedding = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - mean_embedding, axis=1)

        # Normalize distances and importance scores
        distances = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
        if importance_scores is not None:
            importance_scores = (importance_scores - importance_scores.min()) / (importance_scores.max() - importance_scores.min() + 1e-8)
            combined_scores = 0.5 * distances + 0.5 * importance_scores
        else:
            combined_scores = distances

        # Select samples from the middle range (moderate difficulty)
        sorted_indices = np.argsort(combined_scores)
        total_samples = len(sorted_indices)

        # Take samples from the middle 60% (avoiding too easy and too hard)
        start_idx = int(0.2 * total_samples)
        end_idx = int(0.8 * total_samples)
        moderate_indices = sorted_indices[start_idx:end_idx]

        # Randomly sample from moderate range
        if len(moderate_indices) >= num_samples:
            selected_indices = np.random.choice(moderate_indices, num_samples, replace=False)
        else:
            selected_indices = moderate_indices
            # Add more samples if needed
            remaining = num_samples - len(selected_indices)
            other_indices = np.concatenate([sorted_indices[:start_idx], sorted_indices[end_idx:]])
            if len(other_indices) > 0:
                additional = np.random.choice(other_indices, min(remaining, len(other_indices)), replace=False)
                selected_indices = np.concatenate([selected_indices, additional])

        return selected_indices.tolist()[:num_samples]

    def prepare_texts(self, dataset) -> List[str]:
        """Prepare text representations for embedding."""
        texts = []
        for item in dataset:
            # Combine dialogue and summary for embedding
            text = f"Dialogue: {item['dialogue']}\nSummary: {item['summary']}"
            texts.append(text)
        return texts

    def save_results(self, selected_indices: List[int], dataset, method: str,
                    num_samples: int, selection_time: float, output_dir: str,
                    embeddings: Optional[np.ndarray] = None) -> str:
        """Save selection results in the required format."""

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get selected samples
        selected_samples = dataset.select(selected_indices)

        # Compute metadata
        all_dialogue_lengths = [len(item['dialogue']) for item in dataset]
        all_summary_lengths = [len(item['summary']) for item in dataset]

        # Prepare results
        results = {
            "selection_method": method,
            "num_samples": num_samples,
            "selected_indices": selected_indices,
            "selection_time": selection_time,
            "embedding_model": "meta-llama/Llama-2-13b-h",  # Default embedding model
            "train_dataset_size": len(dataset),
            "selected_dialogues": [sample['dialogue'] for sample in selected_samples],
            "selected_summaries": [sample['summary'] for sample in selected_samples],
            "metadata": {
                "total_samples": len(dataset),
                "avg_dialogue_length": np.mean(all_dialogue_lengths),
                "avg_summary_length": np.mean(all_summary_lengths)
            }
        }

        # Save results
        output_file = os.path.join(output_dir, f"selection_results_{method}_{num_samples}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_file}")
        return output_file

    def visualize_selection(self, embeddings: np.ndarray, selected_indices: List[int],
                          method: str, output_dir: str):
        """Create visualization of selected samples."""
        try:
            from sklearn.manifold import TSNE

            logger.info("Creating t-SNE visualization...")

            # Reduce dimensionality for visualization
            tsne = TSNE(n_components=2, random_state=self.seed, perplexity=min(30, len(embeddings)//4))
            embeddings_2d = tsne.fit_transform(embeddings)

            # Create plot
            plt.figure(figsize=(12, 8))

            # Plot all samples
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                       c='lightblue', alpha=0.6, s=20, label='All samples')

            # Plot selected samples
            selected_embeddings_2d = embeddings_2d[selected_indices]
            plt.scatter(selected_embeddings_2d[:, 0], selected_embeddings_2d[:, 1],
                       c='red', alpha=0.8, s=40, label='Selected samples')

            plt.title(f'DialogSum Selection Visualization: {method}')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save plot
            viz_file = os.path.join(output_dir, f"selection_viz_{method}_{len(selected_indices)}_tsne.png")
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Visualization saved to {viz_file}")

        except ImportError:
            logger.warning("scikit-learn not available for t-SNE visualization")
        except Exception as e:
            logger.warning(f"Error creating visualization: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DialogSum Data Selection")

    # Dataset parameters
    parser.add_argument("--dataset_name", type=str, default="knkarthick/dialogsum",
                        help="HuggingFace dataset name")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Directory for caching models and data")

    # Selection parameters
    parser.add_argument("--selection_method", type=str, default="random",
                        choices=["random", "diversity", "k_center", "moderate"],
                        help="Selection method to use")
    parser.add_argument("--num_samples", type=int, default=1250,
                        help="Number of samples to select")
    parser.add_argument("--embedding_model", type=str, default="meta-llama/Llama-2-13b-hf",
                        help="Sentence transformer model for embeddings")

    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--visualize", action="store_true",
                        help="Create t-SNE visualization of selection")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    logger.info("DialogSum Data Selection")
    logger.info("=" * 50)
    logger.info(f"Selection method: {args.selection_method}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Embedding model: {args.embedding_model}")

    # Initialize selector
    selector = DialogSumSelector(cache_dir=args.cache_dir, seed=args.seed)

    # Load dataset
    dataset = selector.load_dataset(args.dataset_name)
    train_dataset = dataset["train"]

    # Prepare texts and get embeddings (for methods that need them)
    embeddings = None
    if args.selection_method in ["diversity", "k_center", "moderate"]:
        texts = selector.prepare_texts(train_dataset)
        embeddings = selector.get_embeddings(texts, args.embedding_model)

    # Perform selection
    logger.info(f"Starting {args.selection_method} selection...")
    start_time = time.time()

    if args.selection_method == "random":
        selected_indices = selector.select_random(len(train_dataset), args.num_samples)
    elif args.selection_method == "diversity":
        selected_indices = selector.select_diversity(embeddings, args.num_samples)
    elif args.selection_method == "k_center":
        selected_indices = selector.select_k_center(embeddings, args.num_samples)
    elif args.selection_method == "moderate":
        # For moderate, we can use embedding distances as importance scores
        importance_scores = np.linalg.norm(embeddings - np.mean(embeddings, axis=0), axis=1)
        selected_indices = selector.select_moderate(embeddings, importance_scores, args.num_samples)
    else:
        raise ValueError(f"Unknown selection method: {args.selection_method}")

    selection_time = time.time() - start_time

    # Save results
    output_file = selector.save_results(
        selected_indices=selected_indices,
        dataset=train_dataset,
        method=args.selection_method,
        num_samples=args.num_samples,
        selection_time=selection_time,
        output_dir=args.output_dir,
        embeddings=embeddings
    )

    # Create visualization if requested
    if args.visualize and embeddings is not None:
        selector.visualize_selection(embeddings, selected_indices, args.selection_method, args.output_dir)

    # Print summary
    logger.info("=" * 50)
    logger.info("SELECTION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Method: {args.selection_method}")
    logger.info(f"Total samples: {len(train_dataset)}")
    logger.info(f"Selected samples: {len(selected_indices)}")
    logger.info(f"Selection ratio: {len(selected_indices) / len(train_dataset):.3f}")
    logger.info(f"Selection time: {selection_time:.2f} seconds")
    logger.info(f"Results saved to: {output_file}")

    logger.info("Selection completed successfully!")


if __name__ == "__main__":
    main()