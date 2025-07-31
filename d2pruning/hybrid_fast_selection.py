#!/usr/bin/env python3
"""
Hybrid Fast Data Selection for Assistant Model Training

This module implements the hybrid fast approach combining:
1. Gradient-based selection using gradient norms from teacher model forward passes
2. Fast embedding-based clustering for diversity
3. Stratified sampling within clusters based on gradient difficulty

The approach is designed to be much faster than full generation while maintaining
both diversity and difficulty-based selection quality.
"""

import os
import json
import time
import pickle
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    set_seed
)
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class SelectionConfig:
    """Configuration for hybrid fast selection."""
    # Dataset parameters
    dataset_name: str = "knkarthick/dialogsum"
    cache_dir: str = "./cache"
    
    # Teacher model parameters
    teacher_model_name: str = "microsoft/deberta-xlarge-mnli"
    teacher_device: str = "cuda"
    teacher_batch_size: int = 8
    max_length: int = 512
    early_layer_idx: int = 40  # Layer to stop at for early stopping
    
    # Embedding parameters
    embedding_model_name: str = "all-mpnet-base-v2"
    embedding_batch_size: int = 32
    
    # Clustering parameters
    n_clusters: int = 10
    clustering_method: str = "kmeans"  # "kmeans" or "minibatch_kmeans"
    
    # Selection parameters
    num_samples: int = 50
    gradient_weight: float = 0.7  # Weight for gradient-based selection within clusters
    diversity_weight: float = 0.3  # Weight for diversity (cluster-based selection)
    
    # Caching parameters
    enable_caching: bool = True
    cache_embeddings: bool = True
    cache_gradients: bool = True
    
    # Output parameters
    output_dir: str = "./hybrid_selection_results"
    save_visualizations: bool = True
    
    # Random seed
    seed: int = 42


class UncertaintyComputor:
    """Fast uncertainty estimation using early stopping and attention weights."""

    def __init__(self, config: SelectionConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device(config.teacher_device if torch.cuda.is_available() else "cpu")
        self.early_layer_idx = config.early_layer_idx  # Use config value
        
    def _load_teacher_model(self):
        """Load teacher model and tokenizer."""
        if self.model is not None:
            return
            
        logger.info(f"Loading teacher model: {self.config.teacher_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.teacher_model_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True
        )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model based on model type
        if "llama" in self.config.teacher_model_name.lower():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.teacher_model_name,
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        elif "deberta" in self.config.teacher_model_name.lower():
            # For DeBERTa models (classification/NLI models)
            from transformers import AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.teacher_model_name,
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float32,  # Use float32 for DeBERTa to avoid dtype issues
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # For seq2seq models like BART
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.teacher_model_name,
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.model.eval()
        logger.info(f"Teacher model loaded on device: {self.device}")
    
    def compute_uncertainty_scores(self, texts: List[str], targets: Optional[List[str]] = None) -> np.ndarray:
        """
        Compute uncertainty scores using early stopping and attention analysis.
        Much faster than gradient computation.

        Args:
            texts: List of input texts
            targets: Optional list of target texts (for seq2seq models)

        Returns:
            Array of uncertainty scores for each text
        """
        self._load_teacher_model()

        # Check cache first
        cache_file = None
        if self.config.cache_gradients:  # Reuse cache flag for uncertainty
            cache_file = os.path.join(
                self.config.cache_dir,
                f"uncertainty_scores_{self.config.teacher_model_name.replace('/', '_')}_layer{self.early_layer_idx}.npy"
            )
            if os.path.exists(cache_file):
                logger.info(f"Loading cached uncertainty scores from {cache_file}")
                return np.load(cache_file)

        logger.info(f"Computing uncertainty scores for {len(texts)} samples using early stopping at layer {self.early_layer_idx}...")
        uncertainty_scores = []

        # Process in batches
        batch_size = self.config.teacher_batch_size
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing uncertainty scores"):
            batch_texts = texts[i:i + batch_size]
            batch_targets = targets[i:i + batch_size] if targets else None

            batch_scores = self._compute_batch_uncertainty_scores(batch_texts, batch_targets)
            uncertainty_scores.extend(batch_scores)

        uncertainty_scores = np.array(uncertainty_scores)

        # Cache results
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.save(cache_file, uncertainty_scores)
            logger.info(f"Cached uncertainty scores to {cache_file}")

        return uncertainty_scores
    
    def _compute_batch_uncertainty_scores(self, batch_texts: List[str], batch_targets: Optional[List[str]] = None) -> List[float]:
        """Compute uncertainty scores for a batch of texts using early stopping."""
        try:
            # Tokenize all texts in batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward pass with early stopping - no gradients needed!
            with torch.no_grad():
                if "deberta" in self.config.teacher_model_name.lower():
                    # For DeBERTa models
                    uncertainty_scores = self._compute_deberta_uncertainty(inputs)
                elif hasattr(self.model, 'transformer'):
                    # For GPT-style models
                    uncertainty_scores = self._compute_transformer_uncertainty(inputs)
                elif hasattr(self.model, 'encoder'):
                    # For BART/T5-style models
                    uncertainty_scores = self._compute_encoder_decoder_uncertainty(inputs, batch_targets)
                else:
                    # Fallback: use output logits variance
                    uncertainty_scores = self._compute_logits_uncertainty(inputs)

            return uncertainty_scores.cpu().numpy().tolist()

        except Exception as e:
            logger.warning(f"Error computing uncertainty scores: {e}")
            return [0.0] * len(batch_texts)

    def _compute_deberta_uncertainty(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute uncertainty for DeBERTa models using hidden states and attention."""
        # Get model outputs with hidden states and attention
        outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)

        # Use hidden states from early layers (up to self.early_layer_idx)
        hidden_states = outputs.hidden_states[:self.early_layer_idx + 1]  # +1 because we want layers 0 to early_layer_idx
        attentions = outputs.attentions[:self.early_layer_idx]

        uncertainty_scores = []
        batch_size = inputs['input_ids'].size(0)

        for batch_idx in range(batch_size):
            # Method 1: Hidden state variance across layers
            hidden_variances = []
            for hidden_state in hidden_states:
                # hidden_state shape: [batch, seq_len, hidden_dim]
                batch_hidden = hidden_state[batch_idx]  # [seq_len, hidden_dim]

                # Compute variance across hidden dimensions and sequence positions
                hidden_var = torch.var(batch_hidden)
                hidden_variances.append(hidden_var)

            # Method 2: Attention variance across heads and layers
            attention_variances = []
            for layer_attention in attentions:
                # layer_attention shape: [batch, heads, seq_len, seq_len]
                batch_attention = layer_attention[batch_idx]  # [heads, seq_len, seq_len]

                # Compute variance across attention heads
                attention_var = torch.var(batch_attention)
                attention_variances.append(attention_var)

            # Combine hidden state and attention uncertainties
            hidden_uncertainty = torch.mean(torch.stack(hidden_variances))
            attention_uncertainty = torch.mean(torch.stack(attention_variances)) if attention_variances else torch.tensor(0.0, device=hidden_uncertainty.device)

            # Weighted combination (favor hidden states for DeBERTa)
            combined_uncertainty = 0.7 * hidden_uncertainty + 0.3 * attention_uncertainty
            uncertainty_scores.append(combined_uncertainty)

        return torch.stack(uncertainty_scores)

    def _compute_transformer_uncertainty(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute uncertainty for transformer models using attention variance."""
        # Get hidden states from early layers
        outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)

        # Use attention weights from early layers as uncertainty proxy
        early_attentions = outputs.attentions[:self.early_layer_idx]  # First N layers

        uncertainty_scores = []
        batch_size = inputs['input_ids'].size(0)

        for batch_idx in range(batch_size):
            # Compute attention variance across heads and layers
            attention_variances = []
            for layer_attention in early_attentions:
                # layer_attention shape: [batch, heads, seq_len, seq_len]
                batch_attention = layer_attention[batch_idx]  # [heads, seq_len, seq_len]

                # Compute variance across attention heads for each position
                attention_var = torch.var(batch_attention, dim=0)  # [seq_len, seq_len]
                mean_var = torch.mean(attention_var)
                attention_variances.append(mean_var)

            # Higher variance = higher uncertainty
            uncertainty_score = torch.mean(torch.stack(attention_variances))
            uncertainty_scores.append(uncertainty_score)

        return torch.stack(uncertainty_scores)

    def _compute_encoder_decoder_uncertainty(self, inputs: Dict[str, torch.Tensor], targets: Optional[List[str]] = None) -> torch.Tensor:
        """Compute uncertainty for encoder-decoder models."""
        # Get encoder outputs with early stopping
        encoder_outputs = self.model.get_encoder()(**inputs, output_hidden_states=True, output_attentions=True)

        # Use early layer hidden states variance as uncertainty
        early_hidden_states = encoder_outputs.hidden_states[:self.early_layer_idx]

        uncertainty_scores = []
        batch_size = inputs['input_ids'].size(0)

        for batch_idx in range(batch_size):
            layer_variances = []
            for hidden_state in early_hidden_states:
                # hidden_state shape: [batch, seq_len, hidden_dim]
                batch_hidden = hidden_state[batch_idx]  # [seq_len, hidden_dim]

                # Compute variance across hidden dimensions
                hidden_var = torch.var(batch_hidden, dim=-1)  # [seq_len]
                mean_var = torch.mean(hidden_var)
                layer_variances.append(mean_var)

            uncertainty_score = torch.mean(torch.stack(layer_variances))
            uncertainty_scores.append(uncertainty_score)

        return torch.stack(uncertainty_scores)

    def _compute_logits_uncertainty(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fallback: compute uncertainty using output logits entropy."""
        outputs = self.model(**inputs)
        logits = outputs.logits

        # Compute entropy of output distribution
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # [batch, seq_len] or [batch]

        # Mean entropy per sample (handle both 2D and 1D cases)
        if entropy.dim() > 1:
            uncertainty_scores = torch.mean(entropy, dim=-1)  # [batch]
        else:
            uncertainty_scores = entropy  # Already [batch]

        return uncertainty_scores


class FastClusterer:
    """Fast embedding-based clustering for diversity sampling."""

    def __init__(self, config: SelectionConfig):
        self.config = config
        self.embedding_model = None
        self.cluster_labels = None
        self.cluster_centers = None

    def _load_embedding_model(self):
        """Load sentence transformer model."""
        if self.embedding_model is not None:
            return

        logger.info(f"Loading embedding model: {self.config.embedding_model_name}")
        self.embedding_model = SentenceTransformer(
            self.config.embedding_model_name,
            cache_folder=self.config.cache_dir
        )

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of texts."""
        self._load_embedding_model()

        # Check cache first
        cache_file = None
        if self.config.cache_embeddings:
            cache_file = os.path.join(
                self.config.cache_dir,
                f"embeddings_{self.config.embedding_model_name.replace('/', '_')}.npy"
            )
            if os.path.exists(cache_file):
                logger.info(f"Loading cached embeddings from {cache_file}")
                return np.load(cache_file)

        logger.info(f"Computing embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.config.embedding_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Cache results
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.save(cache_file, embeddings)
            logger.info(f"Cached embeddings to {cache_file}")

        return embeddings

    def cluster_embeddings(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster embeddings using fast clustering method.

        Args:
            embeddings: Array of embeddings to cluster

        Returns:
            Tuple of (cluster_labels, cluster_centers)
        """
        logger.info(f"Clustering {len(embeddings)} embeddings into {self.config.n_clusters} clusters...")

        # Choose clustering method
        if self.config.clustering_method == "minibatch_kmeans":
            clusterer = MiniBatchKMeans(
                n_clusters=self.config.n_clusters,
                random_state=self.config.seed,
                batch_size=min(1000, len(embeddings) // 10)
            )
        else:
            clusterer = KMeans(
                n_clusters=self.config.n_clusters,
                random_state=self.config.seed,
                n_init=10
            )

        # Fit clustering
        cluster_labels = clusterer.fit_predict(embeddings)
        cluster_centers = clusterer.cluster_centers_

        # Compute silhouette score for quality assessment
        if len(embeddings) > self.config.n_clusters:
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            logger.info(f"Clustering silhouette score: {silhouette_avg:.3f}")

        self.cluster_labels = cluster_labels
        self.cluster_centers = cluster_centers

        return cluster_labels, cluster_centers


class HybridFastSelector:
    """
    Main class implementing the hybrid fast selection approach.

    Combines uncertainty-based difficulty estimation with fast clustering for diversity.
    """

    def __init__(self, config: SelectionConfig):
        self.config = config
        self.uncertainty_computor = UncertaintyComputor(config)
        self.clusterer = FastClusterer(config)

        # Set random seed
        set_seed(config.seed)
        np.random.seed(config.seed)

    def select_samples(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Perform hybrid fast selection on the dataset.

        Args:
            dataset: HuggingFace dataset to select from

        Returns:
            Dictionary containing selection results and metadata
        """
        logger.info(f"Starting hybrid fast selection for {len(dataset)} samples...")
        start_time = time.time()

        # Step 1: Prepare texts for processing
        texts = self._prepare_texts(dataset)
        targets = self._prepare_targets(dataset) if self._has_targets(dataset) else None

        # Step 2: Compute embeddings for clustering
        logger.info("Step 1/4: Computing embeddings for clustering...")
        embeddings = self.clusterer.compute_embeddings(texts)

        # Step 3: Perform fast clustering
        logger.info("Step 2/4: Performing fast clustering...")
        cluster_labels, cluster_centers = self.clusterer.cluster_embeddings(embeddings)

        # Step 4: Compute uncertainty scores for difficulty estimation (much faster!)
        logger.info("Step 3/4: Computing uncertainty scores with early stopping...")
        uncertainty_scores = self.uncertainty_computor.compute_uncertainty_scores(texts, targets)

        # Step 5: Perform stratified selection within clusters
        logger.info("Step 4/4: Performing stratified selection within clusters...")
        selected_indices = self._stratified_cluster_selection(
            cluster_labels, uncertainty_scores, embeddings
        )

        # Prepare results
        selection_time = time.time() - start_time
        results = {
            "selected_indices": selected_indices,
            "selected_samples": dataset.select(selected_indices),
            "cluster_labels": cluster_labels,
            "uncertainty_scores": uncertainty_scores,  # Changed from gradient_norms
            "embeddings": embeddings,
            "cluster_centers": cluster_centers,
            "selection_time": selection_time,
            "config": self.config,
            "metadata": {
                "total_samples": len(dataset),
                "selected_samples": len(selected_indices),
                "n_clusters": self.config.n_clusters,
                "selection_method": "hybrid_fast_uncertainty",  # Updated method name
                "gradient_weight": self.config.gradient_weight,  # Keep for compatibility
                "diversity_weight": self.config.diversity_weight
            }
        }

        logger.info(f"Selection completed in {selection_time:.2f} seconds")
        logger.info(f"Selected {len(selected_indices)} samples from {len(dataset)} total")

        return results

    def _prepare_texts(self, dataset: Dataset) -> List[str]:
        """Prepare input texts from dataset."""
        if "dialogue" in dataset.column_names and "summary" in dataset.column_names:
            # DialogSum format
            return [f"Dialogue: {item['dialogue']}\nSummary: {item['summary']}"
                   for item in dataset]
        elif "text" in dataset.column_names:
            return dataset["text"]
        elif "input" in dataset.column_names:
            return dataset["input"]
        else:
            # Try to find the first text column
            text_columns = [col for col in dataset.column_names
                          if dataset[col][0] and isinstance(dataset[col][0], str)]
            if text_columns:
                return dataset[text_columns[0]]
            else:
                raise ValueError("Could not find text column in dataset")

    def _prepare_targets(self, dataset: Dataset) -> Optional[List[str]]:
        """Prepare target texts from dataset if available."""
        if "summary" in dataset.column_names:
            return dataset["summary"]
        elif "target" in dataset.column_names:
            return dataset["target"]
        elif "output" in dataset.column_names:
            return dataset["output"]
        return None

    def _has_targets(self, dataset: Dataset) -> bool:
        """Check if dataset has target/output columns."""
        target_columns = ["summary", "target", "output", "label"]
        return any(col in dataset.column_names for col in target_columns)

    def _stratified_cluster_selection(self, cluster_labels: np.ndarray,
                                    uncertainty_scores: np.ndarray,
                                    embeddings: np.ndarray) -> List[int]:
        """
        Perform stratified selection within clusters based on uncertainty scores.

        Args:
            cluster_labels: Cluster assignment for each sample
            uncertainty_scores: Uncertainty scores for difficulty estimation
            embeddings: Sample embeddings for diversity

        Returns:
            List of selected sample indices
        """
        selected_indices = []

        # Calculate samples per cluster
        unique_clusters = np.unique(cluster_labels)
        samples_per_cluster = self.config.num_samples // len(unique_clusters)
        remaining_samples = self.config.num_samples % len(unique_clusters)

        logger.info(f"Selecting ~{samples_per_cluster} samples per cluster from {len(unique_clusters)} clusters")

        for i, cluster_id in enumerate(unique_clusters):
            # Get samples in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_uncertainty_scores = uncertainty_scores[cluster_mask]

            # Determine number of samples for this cluster
            n_samples_cluster = samples_per_cluster
            if i < remaining_samples:
                n_samples_cluster += 1

            # Skip if cluster is too small
            if len(cluster_indices) <= n_samples_cluster:
                selected_indices.extend(cluster_indices.tolist())
                continue

            # Select samples within cluster based on uncertainty scores
            cluster_selected = self._select_within_cluster(
                cluster_indices, cluster_uncertainty_scores, n_samples_cluster
            )
            selected_indices.extend(cluster_selected)

        return selected_indices[:self.config.num_samples]  # Ensure exact count

    def _select_within_cluster(self, cluster_indices: np.ndarray,
                              uncertainty_scores: np.ndarray,
                              n_samples: int) -> List[int]:
        """
        Select samples within a cluster based on uncertainty scores.

        Uses a combination of highest uncertainty scores (difficulty) and some randomness.
        """
        if len(cluster_indices) <= n_samples:
            return cluster_indices.tolist()

        # Sort by uncertainty scores (descending - highest uncertainty first)
        sorted_indices = np.argsort(uncertainty_scores)[::-1]

        # Select top samples based on uncertainty weight (reusing gradient_weight config)
        n_uncertainty_samples = int(n_samples * self.config.gradient_weight)
        n_random_samples = n_samples - n_uncertainty_samples

        selected = []

        # Select highest uncertainty samples
        if n_uncertainty_samples > 0:
            uncertainty_selected = sorted_indices[:n_uncertainty_samples]
            selected.extend(cluster_indices[uncertainty_selected].tolist())

        # Select remaining samples randomly from the rest
        if n_random_samples > 0:
            remaining_indices = sorted_indices[n_uncertainty_samples:]
            if len(remaining_indices) > 0:
                random_selected = np.random.choice(
                    remaining_indices,
                    size=min(n_random_samples, len(remaining_indices)),
                    replace=False
                )
                selected.extend(cluster_indices[random_selected].tolist())

        return selected

    def save_results(self, results: Dict[str, Any], output_dir: Optional[str] = None) -> str:
        """Save selection results to disk."""
        if output_dir is None:
            output_dir = self.config.output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Save main results
        results_file = os.path.join(output_dir, "selection_results.json")

        # Prepare serializable results
        serializable_results = {
            "selected_indices": results["selected_indices"],
            "selection_time": results["selection_time"],
            "metadata": results["metadata"],
            "config": {
                "dataset_name": self.config.dataset_name,
                "teacher_model_name": self.config.teacher_model_name,
                "embedding_model_name": self.config.embedding_model_name,
                "n_clusters": self.config.n_clusters,
                "num_samples": self.config.num_samples,
                "gradient_weight": self.config.gradient_weight,
                "diversity_weight": self.config.diversity_weight,
                "seed": self.config.seed
            }
        }

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        # Save numpy arrays
        np.save(os.path.join(output_dir, "cluster_labels.npy"), results["cluster_labels"])
        np.save(os.path.join(output_dir, "uncertainty_scores.npy"), results["uncertainty_scores"])
        np.save(os.path.join(output_dir, "embeddings.npy"), results["embeddings"])
        np.save(os.path.join(output_dir, "cluster_centers.npy"), results["cluster_centers"])

        # Save selected samples as JSON
        selected_samples_file = os.path.join(output_dir, "selected_samples.json")
        selected_samples_data = []
        for i, sample in enumerate(results["selected_samples"]):
            sample_dict = dict(sample)
            sample_dict["original_index"] = results["selected_indices"][i]
            sample_dict["cluster_id"] = int(results["cluster_labels"][results["selected_indices"][i]])
            sample_dict["uncertainty_score"] = float(results["uncertainty_scores"][results["selected_indices"][i]])
            selected_samples_data.append(sample_dict)

        with open(selected_samples_file, 'w') as f:
            json.dump(selected_samples_data, f, indent=2)

        logger.info(f"Results saved to {output_dir}")
        return output_dir

    def visualize_results(self, results: Dict[str, Any], output_dir: Optional[str] = None):
        """Create visualizations of the selection results."""
        if not self.config.save_visualizations:
            return

        if output_dir is None:
            output_dir = self.config.output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Cluster distribution plot
        self._plot_cluster_distribution(results, output_dir)

        # 2. Uncertainty score distribution plot
        self._plot_uncertainty_distribution(results, output_dir)

        # 3. Selection overview plot
        self._plot_selection_overview(results, output_dir)

        logger.info(f"Visualizations saved to {output_dir}")

    def _plot_cluster_distribution(self, results: Dict[str, Any], output_dir: str):
        """Plot cluster distribution and selected samples."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Cluster sizes
        unique_clusters, cluster_counts = np.unique(results["cluster_labels"], return_counts=True)
        ax1.bar(unique_clusters, cluster_counts, alpha=0.7)
        ax1.set_xlabel("Cluster ID")
        ax1.set_ylabel("Number of Samples")
        ax1.set_title("Cluster Size Distribution")

        # Selected samples per cluster
        selected_cluster_labels = results["cluster_labels"][results["selected_indices"]]
        unique_selected, selected_counts = np.unique(selected_cluster_labels, return_counts=True)
        ax2.bar(unique_selected, selected_counts, alpha=0.7, color='orange')
        ax2.set_xlabel("Cluster ID")
        ax2.set_ylabel("Selected Samples")
        ax2.set_title("Selected Samples per Cluster")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cluster_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_uncertainty_distribution(self, results: Dict[str, Any], output_dir: str):
        """Plot uncertainty score distributions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # All uncertainty scores
        ax1.hist(results["uncertainty_scores"], bins=50, alpha=0.7, density=True)
        ax1.set_xlabel("Uncertainty Score")
        ax1.set_ylabel("Density")
        ax1.set_title("All Samples: Uncertainty Score Distribution")

        # Selected uncertainty scores
        selected_uncertainty_scores = results["uncertainty_scores"][results["selected_indices"]]
        ax2.hist(selected_uncertainty_scores, bins=30, alpha=0.7, color='orange', density=True)
        ax2.set_xlabel("Uncertainty Score")
        ax2.set_ylabel("Density")
        ax2.set_title("Selected Samples: Uncertainty Score Distribution")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "uncertainty_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_selection_overview(self, results: Dict[str, Any], output_dir: str):
        """Plot overview of selection process."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Scatter plot of uncertainty scores vs cluster assignments
        scatter = ax.scatter(
            results["cluster_labels"],
            results["uncertainty_scores"],
            alpha=0.6,
            s=20,
            c='lightblue',
            label='All samples'
        )

        # Highlight selected samples
        selected_clusters = results["cluster_labels"][results["selected_indices"]]
        selected_uncertainty = results["uncertainty_scores"][results["selected_indices"]]
        ax.scatter(
            selected_clusters,
            selected_uncertainty,
            alpha=0.8,
            s=40,
            c='red',
            label='Selected samples'
        )

        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Uncertainty Score")
        ax.set_title("Hybrid Selection Overview: Uncertainty Scores by Cluster")
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "selection_overview.png"), dpi=300, bbox_inches='tight')
        plt.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hybrid Fast Data Selection")

    # Dataset parameters
    parser.add_argument("--dataset_name", type=str, default="knkarthick/dialogsum",
                        help="HuggingFace dataset name")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Directory for caching models and data")

    # Teacher model parameters
    parser.add_argument("--teacher_model", type=str, default="microsoft/deberta-xlarge-mnli",
                        help="Teacher model for uncertainty computation")
    parser.add_argument("--teacher_batch_size", type=int, default=8,
                        help="Batch size for teacher model inference")
    parser.add_argument("--early_layer_idx", type=int, default=40,
                        help="Layer index to stop at for early stopping (40 for DeBERTa-xlarge)")

    # Embedding parameters
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer model for embeddings")
    parser.add_argument("--embedding_batch_size", type=int, default=32,
                        help="Batch size for embedding computation")

    # Clustering parameters
    parser.add_argument("--n_clusters", type=int, default=10,
                        help="Number of clusters for diversity sampling")
    parser.add_argument("--clustering_method", type=str, default="kmeans",
                        choices=["kmeans", "minibatch_kmeans"],
                        help="Clustering method to use")

    # Selection parameters
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to select")
    parser.add_argument("--gradient_weight", type=float, default=0.7,
                        help="Weight for uncertainty-based selection within clusters")
    parser.add_argument("--diversity_weight", type=float, default=0.3,
                        help="Weight for diversity (cluster-based selection)")

    # Caching parameters
    parser.add_argument("--disable_caching", action="store_true",
                        help="Disable caching of embeddings and uncertainty scores")
    parser.add_argument("--disable_embedding_cache", action="store_true",
                        help="Disable caching of embeddings")
    parser.add_argument("--disable_gradient_cache", action="store_true",
                        help="Disable caching of uncertainty scores")

    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./hybrid_selection_results",
                        help="Directory to save results")
    parser.add_argument("--disable_visualizations", action="store_true",
                        help="Disable saving visualizations")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for tokenization")

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Create configuration
    config = SelectionConfig(
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        teacher_model_name=args.teacher_model,
        teacher_batch_size=args.teacher_batch_size,
        max_length=args.max_length,
        early_layer_idx=args.early_layer_idx,
        embedding_model_name=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        n_clusters=args.n_clusters,
        clustering_method=args.clustering_method,
        num_samples=args.num_samples,
        gradient_weight=args.gradient_weight,
        diversity_weight=args.diversity_weight,
        enable_caching=not args.disable_caching,
        cache_embeddings=not args.disable_embedding_cache and not args.disable_caching,
        cache_gradients=not args.disable_gradient_cache and not args.disable_caching,
        output_dir=args.output_dir,
        save_visualizations=not args.disable_visualizations,
        seed=args.seed
    )

    logger.info("Starting Hybrid Fast Data Selection with Uncertainty-based Sampling")
    logger.info(f"Configuration: {config}")

    # Load dataset
    logger.info(f"Loading dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name, cache_dir=config.cache_dir)
    train_dataset = dataset["train"]

    logger.info(f"Dataset loaded: {len(train_dataset)} samples")

    # Initialize selector
    selector = HybridFastSelector(config)

    # Perform selection
    results = selector.select_samples(train_dataset)

    # Save results
    output_dir = selector.save_results(results)

    # Create visualizations
    selector.visualize_results(results, output_dir)

    # Print summary
    logger.info("=" * 60)
    logger.info("SELECTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples: {results['metadata']['total_samples']}")
    logger.info(f"Selected samples: {results['metadata']['selected_samples']}")
    logger.info(f"Selection ratio: {results['metadata']['selected_samples'] / results['metadata']['total_samples']:.3f}")
    logger.info(f"Number of clusters: {results['metadata']['n_clusters']}")
    logger.info(f"Gradient weight: {results['metadata']['gradient_weight']}")
    logger.info(f"Diversity weight: {results['metadata']['diversity_weight']}")
    logger.info(f"Selection time: {results['selection_time']:.2f} seconds")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)

    # Print cluster statistics
    selected_cluster_labels = results["cluster_labels"][results["selected_indices"]]
    unique_clusters, cluster_counts = np.unique(selected_cluster_labels, return_counts=True)

    logger.info("CLUSTER STATISTICS")
    logger.info("-" * 30)
    for cluster_id, count in zip(unique_clusters, cluster_counts):
        total_in_cluster = np.sum(results["cluster_labels"] == cluster_id)
        percentage = (count / total_in_cluster) * 100
        logger.info(f"Cluster {cluster_id}: {count}/{total_in_cluster} selected ({percentage:.1f}%)")

    # Print gradient statistics
    all_gradients = results["gradient_norms"]
    selected_gradients = all_gradients[results["selected_indices"]]

    logger.info("GRADIENT STATISTICS")
    logger.info("-" * 30)
    logger.info(f"All samples - Mean: {np.mean(all_gradients):.4f}, Std: {np.std(all_gradients):.4f}")
    logger.info(f"Selected samples - Mean: {np.mean(selected_gradients):.4f}, Std: {np.std(selected_gradients):.4f}")
    logger.info(f"Gradient norm ratio (selected/all): {np.mean(selected_gradients) / np.mean(all_gradients):.3f}")

    logger.info("Hybrid Fast Selection completed successfully!")


if __name__ == "__main__":
    main()
