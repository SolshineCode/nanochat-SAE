# nanochat-SAE Examples

This directory contains minimal working examples for using Sparse Autoencoders with nanochat.

## Quick Start

### 1. Train an SAE

Train a Sparse Autoencoder on a small dummy model (for testing):

```bash
python examples/train_sae.py --dummy --num_activations 10000 --num_epochs 5
```

Train on a real nanochat checkpoint:

```bash
python examples/train_sae.py \
    --checkpoint models/d20/base_final.pt \
    --layer 10 \
    --num_activations 1000000 \
    --num_epochs 10
```

**Important:** The examples use random tokens for activation collection. For production use, modify the code to use real training data.

### 2. Interpret Model with SAEs

Load trained SAEs and analyze feature activations:

```bash
python examples/interpret_model.py \
    --checkpoint models/d20/base_final.pt \
    --sae_dir sae_outputs/example
```

## Examples Overview

### `train_sae.py`

Complete SAE training pipeline demonstrating:
- Loading or creating a nanochat model
- Collecting activations using hooks
- Training a TopK/ReLU/Gated SAE
- Saving the trained SAE

**Arguments:**
- `--checkpoint PATH`: Path to nanochat checkpoint (optional)
- `--dummy`: Use a small dummy model for testing
- `--layer N`: Layer to train SAE on (default: 2)
- `--expansion_factor N`: SAE expansion factor (default: 8)
- `--activation TYPE`: SAE type - topk/relu/gated (default: topk)
- `--k N`: Number of active features for TopK (default: 32)
- `--num_activations N`: Number of activations to collect (default: 10000)
- `--num_epochs N`: Training epochs (default: 5)

### `interpret_model.py`

Model interpretation with trained SAEs demonstrating:
- Loading trained SAEs
- Wrapping model with interpretation
- Tracking feature activations during inference
- Analyzing which features are active

**Arguments:**
- `--checkpoint PATH`: Path to nanochat checkpoint (required)
- `--sae_dir PATH`: Directory with trained SAEs (required)
- `--prompt TEXT`: Text prompt to analyze (default: "Hello world")

## Next Steps

After running these examples:

1. **Evaluate SAE Quality:**
   ```bash
   python -m scripts.sae_eval --sae_path sae_outputs/example/best_model.pt
   ```

2. **Visualize Features:**
   ```bash
   python -m scripts.sae_viz --sae_path sae_outputs/example/best_model.pt --all_features
   ```

3. **See Full Pipeline:**
   - Check `scripts/sae_train.py` for production training
   - Check `SAE_README.md` for complete documentation

## Production Usage

These examples use **random tokens** for simplicity. For production SAE training:

1. Use real training data from your dataset
2. Collect more activations (10M+ recommended)
3. Train for more epochs (10-20)
4. Use validation set for early stopping
5. Monitor dead latent fraction

See the main training script for production-ready code:
```bash
python -m scripts.sae_train --checkpoint models/d20/base_final.pt --layer 10
```
