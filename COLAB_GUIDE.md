# Google Colab SAE Training Guide

Train Sparse Autoencoders on nanochat using Google Colab's **free T4 GPU**!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SolshineCode/nanochat-SAE/blob/main/colab_sae_training.ipynb)

## What This Notebook Does

This Colab notebook allows you to:
- ✅ Train SAEs on a **pre-trained nanochat model**
- ✅ Use your **custom reference dataset**
- ✅ Run on **free T4 GPU** (no expensive hardware needed!)
- ✅ Save checkpoints to **Google Drive**
- ✅ Visualize learned features
- ✅ Complete training in **1-2 hours per layer**

## Before You Start

### Requirements
1. **Google Account** - For Google Colab access
2. **Pre-trained Model Checkpoint** - Either:
   - Download a pre-trained nanochat model (d20 recommended, 561M params)
   - Train your own using the main pipeline
   - Use a checkpoint from Karpathy's releases (if available)
3. **Reference Dataset** (optional) - Your custom text data for activation collection
4. **Google Drive** (optional but recommended) - For saving checkpoints

### Hardware
- **GPU**: Google Colab free tier provides T4 GPU (15GB VRAM)
- **Runtime**: ~1-2 hours per layer
- **RAM**: 12-25GB system RAM (available on free tier)

## Quick Start

### Step 1: Open the Notebook
Click the "Open in Colab" badge above or upload `colab_sae_training.ipynb` to Google Colab.

### Step 2: Enable GPU
1. Go to `Runtime` → `Change runtime type`
2. Select `T4 GPU` under Hardware accelerator
3. Click `Save`

### Step 3: Run Setup Cells
Execute the first few cells to:
- Verify GPU availability
- Install dependencies (Rust, uv, PyTorch, etc.)
- Clone the nanochat-SAE repository
- Build the tokenizer

This takes ~5-10 minutes on first run.

### Step 4: Mount Google Drive (Optional)
Run the Google Drive mount cell to save checkpoints and results to your Drive.

### Step 5: Load Model
You have three options:

#### Option A: Upload Your Own Checkpoint
```python
MODEL_PATH = '/content/drive/MyDrive/nanochat-SAE/checkpoints/base_final.pt'
```

Upload a checkpoint to your Google Drive and set the path.

#### Option B: Download from URL
```python
MODEL_URL = 'https://example.com/nanochat_d20.pt'
```

If you have a public URL to a checkpoint, download it directly.

#### Option C: Use Existing Checkpoint
If you already trained a model, point to the checkpoint file.

### Step 6: Prepare Dataset (Optional)
Load your custom reference dataset:

```python
# Upload a text file
from google.colab import files
uploaded = files.upload()

# Or load from Google Drive
DATASET_PATH = '/content/drive/MyDrive/my_dataset.txt'
```

Supported formats:
- `.txt` - Plain text
- `.jsonl` - JSON lines with `text` field

If you skip this, the notebook will use random tokens (for testing only).

### Step 7: Configure SAE Training
Adjust the configuration for your needs:

```python
SAE_CONFIG = {
    'layer': 10,              # Which layer to analyze (0-19 for d20)
    'expansion_factor': 4,    # SAE size multiplier (4x for T4)
    'activation': 'topk',     # topk, relu, or gated
    'k': 32,                  # Active features
    'num_activations': 100_000,  # Fewer for faster training
    'sequence_length': 512,   # Shorter for memory
    'train_batch_size': 512,  # Batch size
    'num_epochs': 5,          # Training epochs
}
```

### Step 8: Train SAE
Run the training cells. The notebook will:
1. Collect activations from the model
2. Train the SAE
3. Save checkpoints to Google Drive
4. Show progress bars and metrics

### Step 9: Visualize Results
Explore the learned features:
- Activation frequency distributions
- Feature magnitude distributions
- Top active features
- Quality metrics (MSE, L0, explained variance)

## T4 GPU Optimizations

The notebook is pre-configured for T4 GPU constraints:

| Setting | T4 Value | Full Scale Value | Reason |
|---------|----------|------------------|--------|
| Expansion Factor | 4x | 8-16x | Smaller SAE fits in 15GB VRAM |
| Activations | 100K | 1M+ | Faster collection & training |
| Sequence Length | 512 | 1024-2048 | Reduces memory usage |
| Batch Size | 512 | 4096 | Fits in memory |
| Epochs | 5 | 10-20 | Quicker iteration |

**Result**: Training completes in 1-2 hours on free T4!

## Memory Tips

If you run out of memory:

1. **Reduce expansion factor**: Try 2x or 3x instead of 4x
2. **Collect fewer activations**: Try 50K instead of 100K
3. **Use smaller batches**: Reduce `train_batch_size` to 256
4. **Shorter sequences**: Use 256 instead of 512
5. **Restart runtime**: `Runtime` → `Restart runtime` to clear memory

## Saving Progress

### Automatic Checkpointing
The notebook saves to Google Drive automatically:
- `checkpoints/activations_layer{N}.pt` - Collected activations
- `results/layer_{N}/sae_final.pt` - Trained SAE
- `results/layer_{N}/feature_distribution.png` - Visualizations

### Manual Download
Download results directly from Colab:
```python
from google.colab import files
files.download('/content/sae_outputs/layer_10/sae_final.pt')
```

## Expected Results

After training, you should see:

### Quality Metrics
- **MSE Loss**: ~0.001-0.01 (lower is better)
- **L0 (active features)**: Close to your `k` value
- **Explained Variance**: 80-95%
- **Dead Features**: <10%

### Feature Analysis
- Activation frequency histogram
- Top 10 most active features
- Feature magnitude distribution

## Troubleshooting

### "No GPU found"
- Go to `Runtime` → `Change runtime type` → Select T4 GPU
- Click `Save` and restart runtime

### "Out of memory"
- Reduce `expansion_factor` to 2-3x
- Reduce `num_activations` to 50K
- Reduce `train_batch_size` to 256
- Restart runtime to clear memory

### "Model checkpoint not found"
- Make sure `MODEL_PATH` points to a valid checkpoint
- Check that you uploaded the file to Google Drive
- Verify the file path is correct

### "Rust/Cargo not found"
- Re-run the Rust installation cell
- Restart runtime and try again
- Check that all setup cells completed successfully

### "Module not found"
- Make sure you ran all setup cells
- Verify you're in the `/content/nanochat-SAE` directory
- Restart runtime and re-run setup

## Next Steps

After training your SAE:

### 1. Analyze Features
Use the evaluation cells to:
- Identify top active features
- Find dead features
- Measure reconstruction quality

### 2. Feature Interpretation
Download the trained SAE and use it locally with:
```bash
python -m scripts.sae_viz --sae_path results/layer_10/sae_final.pt --all_features
```

### 3. Feature Steering
Integrate with the runtime module:
```python
from sae.runtime import InterpretableModel, load_saes
interp_model = InterpretableModel(model, saes)
```

### 4. Multi-Layer Analysis
Train SAEs on multiple layers:
- Early layers (0-5): Low-level features
- Middle layers (6-14): Abstract concepts
- Late layers (15-19): Task-specific features

### 5. Scale Up
Ready for more?
- **Colab Pro**: Access to A100/V100 GPUs
- **Larger models**: Try d26 or d30
- **More activations**: Use 1M+ for better coverage
- **Bigger SAEs**: Use 8-16x expansion

## Cost Estimate

### Google Colab Free Tier
- **Cost**: $0 (completely free!)
- **GPU**: T4 (15GB VRAM)
- **Runtime**: ~1-2 hours per layer
- **Limitations**: May disconnect after 12 hours

### Google Colab Pro ($10/month)
- **GPU**: A100, V100 options
- **Runtime**: Longer sessions, faster training
- **Worth it if**: Training multiple layers, larger models

## Example Workflow

Here's a typical workflow:

```python
# 1. Start with middle layer
SAE_CONFIG['layer'] = 10

# 2. Small test run
SAE_CONFIG['num_activations'] = 10_000  # Quick test
SAE_CONFIG['num_epochs'] = 1

# 3. Full training
SAE_CONFIG['num_activations'] = 100_000
SAE_CONFIG['num_epochs'] = 5

# 4. Train on multiple layers
for layer in [5, 10, 15]:
    SAE_CONFIG['layer'] = layer
    # Run training cells...
```

## Resources

- [nanochat-SAE Documentation](https://github.com/SolshineCode/nanochat-SAE)
- [nanochat Original](https://github.com/karpathy/nanochat)
- [Anthropic: Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features)
- [OpenAI: Scaling SAEs](https://openai.com/research/sparse-autoencoders)
- [Google Colab Guide](https://colab.research.google.com/notebooks/intro.ipynb)

## FAQ

**Q: Do I need a paid Colab subscription?**
A: No! The free tier T4 GPU is sufficient for training SAEs on nanochat models.

**Q: How long does training take?**
A: Typically 1-2 hours per layer on T4 GPU with the default settings.

**Q: Can I use my own dataset?**
A: Yes! Upload your text data and the notebook will use it for activation collection.

**Q: What if I don't have a pre-trained model?**
A: You need to train a model first using the main pipeline, or find a public checkpoint to download.

**Q: Can I train on multiple layers?**
A: Yes! Just change the `layer` parameter and re-run the training cells for each layer.

**Q: Will my session time out?**
A: Free tier may disconnect after ~12 hours of inactivity. Save to Google Drive regularly!

**Q: Can I share my trained SAEs?**
A: Absolutely! Upload to Neuronpedia or share the checkpoint files.

## Contributing

Found an issue or improvement? Open a PR or issue on GitHub!

## License

MIT (same as nanochat)

---

**Ready to explore what your model learned?**

Click the "Open in Colab" badge at the top and start training! 🚀
