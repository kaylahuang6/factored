# Factored Expression (factored_exp)

A research project exploring factor analysis and mechanistic interpretability in tiny transformer language models.

## Overview

This project investigates how transformer models learn and represent information through the lens of factor analysis. It focuses on training small transformer models, analyzing their learned representations, and extracting interpretable factors from hidden states.

## Project Structure

```
factored_exp/
├── train.py                      # Main training script
├── model.py                      # TinyTransformerLM architecture
├── generator.py                  # Data generation and sampling utilities
├── analyze.py                    # Analysis and factor extraction
├── figure_dim_curve.py          # Generate dimension curve visualizations
├── figure_factor_panels.py      # Generate factor panel visualizations
├── very_one_test.py             # Testing and validation
├── checkpoints/                 # Saved model checkpoints (step 100-3000)
├── tiny_transformer_independent.pt  # Independent model weights
└── [visualizations]             # Generated PNG figures
```

## Key Components

### Model (model.py)
- **TinyTransformerLM**: A compact transformer-based language model
  - Configurable embedding dimension (d_model)
  - Multi-head self-attention mechanism
  - Feed-forward layers
  - Positional embeddings
  - Layer normalization

### Training (train.py)
- Uses AdamW optimizer with learning rate 1e-3
- Vocabulary size and sequence length configurable
- Saves checkpoints every 100 steps
- CUDA-enabled (falls back to CPU)

### Data Generation (generator.py)
- Samples training batches
- Maintains consistent vocabulary base
- Supports belief-aware sampling for analysis

### Analysis (analyze.py)
- Collects hidden representations from model
- Extracts beliefs and activations
- Performs dimensionality analysis
- Generates visualization data for PCA analysis

## Usage

### Training a Model
```bash
python train.py
```
Trains a transformer model for 3000 steps and saves checkpoints to `checkpoints/`.

### Analyzing Learned Factors
```bash
python analyze.py
```
Analyzes hidden representations and generates factor analysis data.

### Generating Visualizations
```bash
python figure_dim_curve.py
python figure_factor_panels.py
```
Creates dimension curves and factor panel visualizations.

## Configuration

Key hyperparameters in `train.py`:
- `VOCAB_SIZE`: 257 (BASE_VOCAB + 1)
- `SEQ_LEN`: 8
- `BATCH_SIZE`: 256
- `NUM_STEPS`: 3000
- `d_model`: 96
- `n_heads`: 3
- `n_layers`: 3
- `d_ff`: 256


## Outputs

- **Checkpoints**: Model weights saved at intervals during training
- **Visualizations**:
  - `dim_curve_95.png`: Dimension curve analysis
  - `factor_panels_hidden_pca.png`: PCA visualization of learned factors
  - `factor_panels_true_pca.png`: PCA visualization of true factors
  - `vary_one_panels.png`: Ablation study visualizations

## Research Focus

This project explores:
- How transformer models factorize learned representations
- Mechanistic interpretability through factor analysis
- Relationship between model architecture and learned factors
- PCA-based dimensionality reduction of transformer activations

## Device Support

- Automatically uses CUDA GPU if available
- Falls back to CPU for inference

-
For questions or contributions, please refer to the GitHub repository.
