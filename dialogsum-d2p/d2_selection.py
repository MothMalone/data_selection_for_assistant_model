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
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphDensitySampler:
    """
    Implementation of D2Pruning's graph density sampling algorithm with message passing.
    This balances difficulty and diversity using a graph-based approach.
    """
    
    def __init__(self, embeddings, importance_scores, gamma=1.0):
        """
        Initialize the graph density sampler.
        
        Args:
            embeddings: Feature embeddings for all samples
            importance_scores: Importance scores from training dynamics
            gamma: Parameter controlling the influence spread in the graph
                   (D2Pruning's key parameter for balancing diversity/difficulty)
        """
        self.embeddings = embeddings
        self.importance_scores = importance_scores
        self.gamma = gamma
        self.num_samples = len(embeddings)
        
        # Compute pairwise distances - crucial for D2Pruning's graph construction
        logger.info("Computing pairwise distances for graph construction...")
        self.distances = pairwise_distances(embeddings, metric='euclidean')
        
        # Initialize graph density scores with importance scores
        self.graph_density_scores = np.array(importance_scores).copy()
        
        logger.info(f"Graph initialized with gamma={gamma}, {self.num_samples} nodes")
        
    def select_batch(self, num_samples, log_interval=10):
        """
        Select a batch of samples using D2Pruning's iterative graph algorithm.
        
        Args:
            num_samples: Number of samples to select
            log_interval: How often to log metrics
            
        Returns:
            Indices of selected samples
        """
        # Initialize selection
        selected_indices = []
        remaining_indices = np.arange(self.num_samples)
        
        # Keep track of current graph scores
        current_scores = self.graph_density_scores.copy()
        
        # Track metrics for visualization
        selection_metrics = {
            'selected_score_avg': [],
            'selected_score_max': [],
            'selected_score_min': [],
            'remaining_score_avg': [],
            'remaining_score_max': [],
            'diversity_metric': []
        }
        
        # D2Pruning's iterative selection process
        for i in tqdm(range(num_samples), desc="Selecting samples with D2Pruning"):
            if len(remaining_indices) == 0:
                break
                
            # Select the point with highest score
            max_score_idx = np.argmax(current_scores[remaining_indices])
            idx = remaining_indices[max_score_idx]
            selected_indices.append(idx)
            
            # Save the score before removing
            selected_score = current_scores[idx]
            
            # Remove the selected index from remaining
            remaining_indices = remaining_indices[remaining_indices != idx]
            
            # Compute diversity metric (average distance to already selected)
            if len(selected_indices) > 1:
                selected_embeddings = self.embeddings[selected_indices[:-1]]
                new_embedding = self.embeddings[idx].reshape(1, -1)
                distances = pairwise_distances(new_embedding, selected_embeddings)[0]
                diversity_metric = np.mean(distances)
            else:
                diversity_metric = 0.0
            
            # Apply D2Pruning's message passing to update scores
            score_reductions = []
            for j in remaining_indices:
                # Core D2Pruning formula: reduce score based on similarity and current score
                similarity = np.exp(-self.distances[idx, j] * self.gamma)
                score_reduction = current_scores[idx] * similarity
                current_scores[j] -= score_reduction
                score_reductions.append(score_reduction)
            
            # Track metrics for logging
            selection_metrics['selected_score_avg'].append(np.mean([current_scores[idx] for idx in selected_indices]))
            selection_metrics['selected_score_max'].append(np.max([current_scores[idx] for idx in selected_indices]))
            selection_metrics['selected_score_min'].append(np.min([current_scores[idx] for idx in selected_indices]))
            selection_metrics['remaining_score_avg'].append(np.mean(current_scores[remaining_indices]) if len(remaining_indices) > 0 else 0)
            selection_metrics['remaining_score_max'].append(np.max(current_scores[remaining_indices]) if len(remaining_indices) > 0 else 0)
            selection_metrics['diversity_metric'].append(diversity_metric)
            
            # Log progress at intervals
            if i % log_interval == 0 and i > 0:
                # Log to console
                logger.info(f"Selected {i}/{num_samples} samples, "
                           f"Last score: {selected_score:.4f}, "
                           f"Avg selected: {selection_metrics['selected_score_avg'][-1]:.4f}, "
                           f"Diversity: {diversity_metric:.4f}, "
                           f"Avg reduction: {np.mean(score_reductions) if score_reductions else 0:.4f}")
                
                # Log to wandb
                wandb.log({
                    "selection_step": i,
                    "selected_count": i,
                    "last_selected_score": selected_score,
                    "avg_selected_score": selection_metrics['selected_score_avg'][-1],
                    "max_selected_score": selection_metrics['selected_score_max'][-1],
                    "min_selected_score": selection_metrics['selected_score_min'][-1],
                    "avg_remaining_score": selection_metrics['remaining_score_avg'][-1],
                    "max_remaining_score": selection_metrics['remaining_score_max'][-1],
                    "diversity_metric": diversity_metric,
                    "avg_score_reduction": np.mean(score_reductions) if score_reductions else 0
                })
        
        # Create final selection metrics visualization
        if wandb.run is not None:
            # Create progress plots for scores and diversity
            steps = list(range(len(selection_metrics['selected_score_avg'])))
            
            wandb.log({
                "selection_score_progress": wandb.plot.line_series(
                    xs=steps, 
                    ys=[
                        selection_metrics['selected_score_avg'],
                        selection_metrics['remaining_score_avg']
                    ],
                    keys=["Selected Samples Avg Score", "Remaining Samples Avg Score"],
                    title="D2Pruning Selection Progress",
                    xname="Selection Step"
                )
            })
            
            wandb.log({
                "diversity_progress": wandb.plot.line(
                    steps, selection_metrics['diversity_metric'],
                    "Selection Step", "Diversity Metric",
                    title="Diversity Evolution During Selection"
                )
            })
        
        return selected_indices

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamics_file", type=str, required=True,
                       help="Path to training dynamics JSON file")
    parser.add_argument("--dataset_name", type=str, default="knkarthick/dialogsum")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--output_dir", type=str, default="./selection_results")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=1.0,
                       help="D2Pruning gamma parameter (higher = more diversity)")
    parser.add_argument("--embedding_model", type=str, 
                       default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--log_interval", type=int, default=10,
                       help="Log every N selections")
    parser.add_argument("--wandb_project", type=str, default="dialogsum-d2pruning")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_api_key", type=str, 
                       default="445d8df72343591d6588f101349ad4752497ce62")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def generate_embeddings(model, dataset, batch_size=32, device="cuda"):
    """Generate embeddings for diversity component of D2Pruning"""
    embeddings = []
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Computing embeddings"):
        batch = dataset[i:min(i + batch_size, len(dataset))]
        texts = batch["dialogue"]  # Use dialogues for embedding
        batch_embeddings = model.encode(texts, batch_size=batch_size, 
                                       show_progress_bar=False, device=device)
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)

def main():
    args = parse_args()
    
    # Set up wandb
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or f"d2pruning-selection-gamma{args.gamma}-{args.num_samples}samples",
        config=vars(args)
    )
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train", cache_dir=args.cache_dir)
    
    # Load training dynamics file (difficulty component)
    logger.info(f"Loading training dynamics from: {args.dynamics_file}")
    with open(args.dynamics_file, "r") as f:
        dynamics_data = json.load(f)
    
    # Check if importance scores are precomputed
    if "importance_scores" in dynamics_data:
        logger.info("Using precomputed importance scores")
        importance_scores = np.array(dynamics_data["importance_scores"])
    else:
        logger.info("Using loss dynamics as importance scores")
        importance_scores = np.array(dynamics_data["loss_dynamics"])
        importance_scores = (importance_scores - np.min(importance_scores)) / (np.max(importance_scores) - np.min(importance_scores) + 1e-8)
    
    # Generate embeddings for diversity component
    logger.info(f"Generating embeddings using {args.embedding_model}")
    embedding_model = SentenceTransformer(args.embedding_model, 
                                         device="cuda" if torch.cuda.is_available() else "cpu")
    embeddings = generate_embeddings(embedding_model, dataset)
    
    # Log embedding statistics to wandb
    pca_samples = min(1000, len(embeddings))  # Limit samples for PCA visualization
    if wandb.run is not None:
        from sklearn.decomposition import PCA
        
        # Compute PCA for embedding visualization
        pca = PCA(n_components=2)
        random_indices = np.random.choice(len(embeddings), pca_samples, replace=False)
        embeddings_sample = embeddings[random_indices]
        pca_result = pca.fit_transform(embeddings_sample)
        
        # Create embedding visualization with importance scores as color
        wandb.log({
            "embedding_visualization": wandb.plot.scatter(
                wandb.Table(data=[[x, y, s] for (x, y), s in 
                                 zip(pca_result, importance_scores[random_indices])],
                           columns=["x", "y", "importance"]),
                "x", "y", "embedding_pca", {"color": "importance"}
            )
        })
    
    # Initialize D2Pruning graph sampler
    logger.info(f"Initializing D2Pruning graph sampler with gamma={args.gamma}")
    sampler = GraphDensitySampler(
        embeddings=embeddings,
        importance_scores=importance_scores,
        gamma=args.gamma
    )
    
    # Time the selection process
    logger.info(f"Selecting {args.num_samples} samples with D2Pruning algorithm...")
    start_time = time.time()
    selected_indices = sampler.select_batch(args.num_samples, log_interval=args.log_interval)
    selection_time = time.time() - start_time
    
    # Get selected examples
    selected_dataset = dataset.select(selected_indices)
    
    # Calculate metadata
    dialogue_lengths = [len(d) for d in dataset["dialogue"]]
    summary_lengths = [len(s) for s in dataset["summary"]]
    avg_dialogue_length = sum(dialogue_lengths) / len(dialogue_lengths)
    avg_summary_length = sum(summary_lengths) / len(summary_lengths)
    
    # Create selection results in the specified format
    selection_results = {
        "selection_method": "d2pruning",
        "num_samples": len(selected_indices),
        "selected_indices": selected_indices,
        "selection_time": selection_time,
        "embedding_model": args.embedding_model.split("/")[-1],
        "train_dataset_size": len(dataset),
        "selected_dialogues": [d for d in selected_dataset["dialogue"]],
        "selected_summaries": [s for s in selected_dataset["summary"]],
        "metadata": {
            "total_samples": len(dataset),
            "avg_dialogue_length": avg_dialogue_length,
            "avg_summary_length": avg_summary_length
        }
    }
    
    # Save selection results
    output_file = output_dir / f"d2pruning_selection_{args.num_samples}_gamma_{args.gamma}.json"
    with open(output_file, "w") as f:
        json.dump(selection_results, f, indent=2)
    
    # Log final metrics to wandb
    wandb.log({
        "final_num_samples": len(selected_indices),
        "selection_time": selection_time,
        "avg_dialogue_length": avg_dialogue_length,
        "avg_summary_length": avg_summary_length
    })
    
    logger.info(f"D2Pruning selection saved to {output_file}")
    logger.info(f"Selected {len(selected_indices)} samples in {selection_time:.2f} seconds")
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
