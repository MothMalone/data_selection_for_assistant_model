# DialogSum Data Selection Experiment

This project explores different data selection methods for training LLaMA-7B on the DialogSum dataset.

## Setup and Running the Experiment

1. **Environment Setup**

Make sure you have the required Python packages:

```bash
pip install -r requirements.txt
```

2. **Configure Weights & Biases (optional)**

For logging experiments, set up your W&B API key:

```bash
export WANDB_API_KEY=your_api_key
```

3. **Run the Full Experiment**

Execute all selection methods, fine-tuning, and evaluation:

```bash
bash setup_and_run_experiment.sh
```

This will:
- Select data using different methods (random, moderate, k-center, diversity)
- Fine-tune LLaMA-7B on each selection
- Evaluate all models on test data
- Log metrics to W&B (if configured)

4. **Results**

Results will be saved in these directories:
- Selected samples: `selected_samples/`
- Visualizations: `experiment_results/visualizations/`
- Evaluation results: `d2pruning/eval/`
- Fine-tuned models: `d2pruning/finetuned_models/`

## Troubleshooting

- **W&B Logging**: If W&B logging isn't working, check that your API key is correctly set.
- **Missing Visualizations**: Ensure matplotlib and scikit-learn are installed.
- **Model Loading Error**: If you encounter a model loading error, check the cache directory permissions.

## Additional Commands

Run individual steps:

```bash
# Data selection only
python d2pruning/dialogsum_selection.py --selection_method moderate --num_samples 10 --enable_wandb --visualize

# Fine-tuning only
python d2pruning/finetune_llama.py --selection_results d2pruning/results/selection_results_moderate_10.json

# Evaluation only
python d2pruning/evaluate_summaries.py --model_dirs d2pruning/finetuned_models/* --max_test_samples 50
```
