#!/usr/bin/env python3
"""
Utility functions to save and load selected samples.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


def save_selected_samples(selected_indices: List[int], 
                         samples: Dict[str, Any],
                         selection_method: str,
                         num_samples: int,
                         output_dir: str = "../selected_samples") -> str:
    """
    Save the selected samples to a JSON file and the indices to a numpy file.
    
    Args:
        selected_indices: List of selected indices
        samples: Dictionary containing dialogues and summaries
        selection_method: Selection method used
        num_samples: Number of samples selected
        output_dir: Directory to save the results
        
    Returns:
        Path to the saved JSON file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save indices as numpy array
    np.save(output_dir / "selected_indices.npy", np.array(selected_indices))
    
    # Prepare selected samples
    selected_samples = {
        'selection_method': selection_method,
        'num_samples': num_samples,
        'selected_indices': selected_indices,
        'dialogues': [samples['dialogues'][i] for i in selected_indices],
        'summaries': [samples['summaries'][i] for i in selected_indices],
        'metadata': samples.get('metadata', {})
    }
    
    # Save as JSON
    json_path = output_dir / "selected_samples.json"
    with open(json_path, 'w') as f:
        json.dump(selected_samples, f, indent=2)
    
    # Save importance scores if available
    if 'importance_scores' in samples:
        np.save(output_dir / "importance_scores.npy", samples['importance_scores'])
    
    return str(json_path)
