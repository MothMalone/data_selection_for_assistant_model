# DialogSum Data Selection Experiment

This implementation adapts the d2pruning repository's statistical data selection methods for DialogSum dataset to select 10 samples for fine-tuning LLaMA-7B on dialogue summarization.

## Overview

The experiment compares different statistical data selection methods against random selection:

- **Random Selection**: Baseline random sampling
- **Moderate Selection**: Embedding-based selection using distance from class prototypes
- **K-Center Selection**: Greedy selection for maximum diversity
- **Diversity Selection**: Selection based on embedding diversity scores

## Key Features

- **Statistical Selection Only**: No training dynamics or proxy models required
- **Embedding-Based**: Uses sentence transformers for text representation
- **Comprehensive Caching**: Robust pipeline with caching at each step
- **ROUGE Evaluation**: Standard summarization metrics
- **Multiple Formats**: Support for instruction and chat formatting
- **LoRA Fine-tuning**: Memory-efficient fine-tuning with 4-bit quantization
- **W&B Integration**: Comprehensive experiment tracking with Weights & Biases
- **CSV Export**: Automatic results export to CSV format

## Installation

1. Install the additional requirements:
```bash
pip install -r requirements_dialogsum.txt
```

2. Ensure you have access to LLaMA models (requires HuggingFace authentication)

## Usage

### Quick Start

Run the complete experiment with default settings (includes W&B logging):

```bash
cd d2pruning
python run_dialogsum_experiment.py
```

This will:
1. Select 10 samples using random, moderate, and k_center methods
2. Fine-tune LLaMA-7B on each selection
3. Evaluate on DialogSum test set
4. Compare ROUGE scores
5. Log all metrics to Weights & Biases
6. Export results to CSV

To run without W&B logging:
```bash
python run_dialogsum_experiment.py --disable_wandb
```

### Custom Configuration

```bash
python run_dialogsum_experiment.py \
    --selection_methods random moderate k_center diversity \
    --num_samples 10 \
    --model_name meta-llama/Llama-2-7b-hf \
    --format_type instruction \
    --num_epochs 3 \
    --max_test_samples 100
```

### Individual Steps

You can also run individual components:

#### 1. Data Selection
```bash
python dialogsum_selection.py \
    --num_samples 10 \
    --selection_method moderate \
    --embedding_model all-MiniLM-L6-v2
```

#### 2. Fine-tuning
```bash
python finetune_llama.py \
    --selection_results results/selection_results_moderate_10.json \
    --model_name meta-llama/Llama-2-7b-hf \
    --format_type instruction \
    --num_epochs 3
```

#### 3. Evaluation
```bash
python evaluate_summaries.py \
    --model_dirs finetuned_models/moderate_10samples_instruction \
    --base_model meta-llama/Llama-2-7b-hf \
    --max_test_samples 100
```

## Selection Methods

### 1. Random Selection
- Baseline method using random sampling
- Provides comparison baseline

### 2. Moderate Selection
- Uses embedding distances from class prototypes
- Selects samples in the middle range of distances
- Balances typical and diverse examples

### 3. K-Center Selection
- Greedy algorithm for maximum diversity
- Selects samples to maximize minimum pairwise distances
- Ensures diverse coverage of the embedding space

### 4. Diversity Selection
- Selects samples with highest diversity scores
- Based on distance from embedding centroid
- Focuses on outlier/diverse examples

## Fine-tuning Configuration

### Model Settings
- **Base Model**: LLaMA-2-7B (or compatible)
- **Quantization**: 4-bit for memory efficiency
- **LoRA**: Rank 16, Alpha 32, targeting attention layers
- **Batch Size**: 1 with gradient accumulation

### Training Settings
- **Epochs**: 3 (default)
- **Learning Rate**: 2e-4
- **Optimizer**: AdamW
- **Scheduler**: Linear warmup

### Format Types

#### Instruction Format
```
### Instruction:
Summarize the following dialogue:

### Input:
[dialogue text]

### Response:
[summary]
```

#### Chat Format
```
<|system|>
You are a helpful assistant that summarizes dialogues.
<|user|>
Please summarize this dialogue:

[dialogue text]
<|assistant|>
[summary]
```

## Evaluation Metrics

The experiment evaluates using standard ROUGE metrics:
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap  
- **ROUGE-L**: Longest common subsequence
- **ROUGE-Lsum**: Summary-level ROUGE-L

## Output Structure

```
d2pruning/
├── cache/                          # Cached datasets and embeddings
├── results/                        # Selection results
│   ├── selection_results_random_10.json
│   ├── selection_results_moderate_10.json
│   └── ...
├── finetuned_models/              # Fine-tuned models
│   ├── random_10samples_instruction/
│   ├── moderate_10samples_instruction/
│   └── ...
├── evaluation_results/            # Evaluation outputs
│   ├── comparison_summary.json
│   ├── eval_random_10samples_instruction.json
│   └── ...
└── experiment_summary.json        # Complete experiment results
```

## Results Interpretation

The experiment provides:

1. **Selection Time**: Time taken for each selection method
2. **Training Time**: Fine-tuning duration for each method
3. **ROUGE Scores**: Summarization quality metrics
4. **Generation Time**: Inference speed on test set

### Expected Findings

- **Random**: Baseline performance
- **Moderate**: Balanced selection may perform well
- **K-Center**: High diversity might help generalization
- **Diversity**: May select challenging examples

## Memory Requirements

- **GPU Memory**: ~12-16GB for LLaMA-7B with 4-bit quantization
- **RAM**: ~8-16GB depending on dataset size
- **Storage**: ~10-20GB for models and cache

## Weights & Biases Integration

The implementation includes comprehensive W&B logging with the API key `445d8df72343591d6588f101349ad4752497ce62`.

### Logged Metrics

**Dataset Metrics:**
- Train/test dataset sizes
- Average dialogue/summary lengths
- Text statistics

**Embedding Metrics:**
- Embedding extraction time
- Embedding dimensions
- Model information

**Selection Metrics:**
- Selection time and efficiency
- Diversity scores and distributions
- Selected sample characteristics

**Training Metrics:**
- Real-time loss and learning rate
- Training time and efficiency
- Model parameters (total/trainable)
- LoRA configuration

**Evaluation Metrics:**
- ROUGE scores (1, 2, L, Lsum)
- Generation time and speed
- Sample predictions vs references

**Comparison Metrics:**
- Cross-method performance comparison
- Best method identification
- Method rankings

### Testing W&B Integration

Test the W&B setup before running experiments:

```bash
python test_wandb_integration.py
```

### W&B Dashboard Features

- **Real-time Training**: Monitor loss curves during fine-tuning
- **Method Comparison**: Compare selection methods side-by-side
- **Sample Inspection**: View generated vs reference summaries
- **Performance Tracking**: Track timing and efficiency metrics
- **Experiment History**: Access all previous runs and results

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use smaller model
2. **HuggingFace Authentication**: Ensure proper token setup
3. **Slow Embedding Extraction**: Use smaller embedding model
4. **Evaluation Timeout**: Reduce max_test_samples
5. **W&B Connection Issues**: Check internet connection and API key

### Performance Tips

1. Use SSD storage for faster caching
2. Enable mixed precision training
3. Use gradient checkpointing for memory
4. Cache embeddings for repeated experiments
5. Use W&B offline mode if needed

## Customization

### Adding New Selection Methods

1. Implement method in `dialogsum_selection.py`
2. Add to argument choices
3. Update main selection logic

### Different Models

- Change `model_name` parameter
- Ensure compatibility with LoRA
- Adjust memory settings as needed

### Different Datasets

- Modify `DialogSumProcessor` class
- Update data loading and preprocessing
- Adjust evaluation metrics if needed

## Citation

If you use this implementation, please cite the original d2pruning paper and DialogSum dataset:

```bibtex
@article{d2pruning,
  title={D2 Pruning: Message Passing for Balancing Diversity and Difficulty in Data Pruning},
  author={...},
  journal={...},
  year={...}
}

@article{dialogsum,
  title={DialogSum: A Real-Life Scenario Dialogue Summarization Dataset},
  author={Chen, Yulong and Liu, Yang and Chen, Liang and Zhang, Yue},
  journal={arXiv preprint arXiv:2105.06762},
  year={2021}
}
```
