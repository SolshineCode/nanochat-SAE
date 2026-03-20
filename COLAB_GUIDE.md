# Google Colab SAE Training Guide

Train Sparse Autoencoders on Karpathy's pre-trained **nanochat-d32** (1.88B params) using Google Colab's **free T4 GPU** — fully automated in ~6 minutes.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SolshineCode/nanochat-SAE/blob/claude%2Fnanochat-sae-interpretability-011CUT2TocZpFerXthoW9LMf/colab_sae_training.ipynb)

## What This Notebook Does

The Colab notebook runs a complete SAE training pipeline end-to-end:

1. **Auto-downloads** the nanochat-d32 checkpoint (7.2GB) from Karpathy's HuggingFace
2. **Loads the model** in bfloat16 using memory-mapped I/O (fits in T4's 15.6GB VRAM + 12.7GB RAM)
3. **Downloads WikiText-103** for real-text activation collection
4. **Collects 50K activations** from layer 16 via PyTorch forward hooks
5. **Trains a TopK SAE** (2048 → 8192 → 2048, k=32) with 3 epochs
6. **Evaluates** reconstruction quality, sparsity, and feature statistics
7. **Generates** a 4-panel visualization (loss curve, feature frequency, magnitudes, cumulative explained variance)

No pre-trained checkpoint upload needed — everything downloads automatically.

## Verified Results

The following results were verified on Google Colab's free tier T4 GPU:

```
TRAINING COMPLETE - nanochat-d32 SAE
======================================================================
Model:   nanochat-d32 (1.88B params)
Layer:   16 (blocks.16.hook_resid_post)
SAE:     2048 -> 8192 -> 2048 (TopK, k=32)
Data:    50,000 activations from WikiText-103

Results:
  Explained Variance: 57.5%
  MSE Loss:           0.420688
  Alive Features:     2213/8192
  Best Train Loss:    0.445452
======================================================================
```

| Metric | Value |
|---|---|
| Explained Variance | 57.5% |
| MSE Loss | 0.420688 |
| Alive Features | 2,213 / 8,192 (27%) |
| L0 (active features) | 32 (matches target k) |
| Dead Features | 5,979 / 8,192 (73%) |
| Total pipeline time | ~6 minutes |

The 57.5% explained variance with 50K activations and 3 epochs is a solid starting point. Scaling up to more activations, more epochs, or larger expansion factors will improve these numbers significantly.

## Quick Start

### Step 1: Open the Notebook
Click the "Open in Colab" badge above.

### Step 2: Enable GPU
1. Go to `Runtime` → `Change runtime type`
2. Select `T4 GPU`
3. Click `Save`

### Step 3: Run All Cells
Click `Runtime` → `Run all` and accept the "not authored by Google" dialog. That's it — everything else is automated.

The pipeline stages and approximate timings:

| Stage | Time |
|---|---|
| Environment setup (Rust, deps, tokenizer build) | ~2 min |
| Model download from HuggingFace | ~1 min (cached after first run) |
| Model loading (mmap, bfloat16) | ~15 sec |
| WikiText-103 download + tokenization | ~20 sec |
| Activation collection (50K from layer 16) | ~1.5 min |
| SAE training (3 epochs, 97 steps each) | ~15 sec |
| Evaluation + visualization | ~5 sec |

## Technical Details

### Memory-Efficient Model Loading

The 1.88B parameter nanochat-d32 model is loaded using two key optimizations:

- **Memory-mapped I/O** (`torch.load(mmap=True)`): The 7.2GB checkpoint is memory-mapped rather than fully loaded into RAM, keeping peak memory usage well within Colab's ~12.7GB system RAM limit
- **bfloat16 precision**: The model is loaded in bfloat16 (required for nanochat's rotary embeddings), which halves GPU VRAM usage to ~3.7GB on the T4's 15.6GB

### Activation Collection

Activations are collected using PyTorch forward hooks on `model.transformer.h[16]` (the middle layer of the 32-layer model). Each activation is a 2048-dimensional vector (d_model) converted to float32 and stored on CPU to save GPU memory.

### Dataset

The notebook uses WikiText-103 from HuggingFace's `datasets` library (parquet format). It downloads quickly (~300MB), filters to non-empty documents, tokenizes using nanochat's RustBPE tokenizer, and produces 603 sequences of length 512.

## T4 GPU Optimizations

Settings are tuned for T4 GPU constraints:

| Setting | T4 Value | Full Scale | Reason |
|---------|----------|------------|--------|
| Model dtype | bfloat16 | bfloat16 | Required for rotary embeddings; halves VRAM |
| Checkpoint loading | mmap=True | Standard | Keeps RAM usage low on free tier |
| Expansion Factor | 4x | 8-16x | Smaller SAE fits alongside 1.88B model |
| Activations | 50K | 1M+ | Fits in free tier RAM (~400MB) |
| Sequence Length | 512 | 2048 | Reduces per-batch memory |
| Collect Batch Size | 2 | 8-16 | Conservative for 1.88B model on T4 |
| Train Batch Size | 512 | 4096 | Fits in memory |
| Epochs | 3 | 10-20 | Fast iteration; increase for better results |

## Scaling Up

To improve results beyond the defaults:

### More Activations (recommended first step)
Increase `NUM_ACTIVATIONS` from 50K to 200K+ for better feature coverage. This costs ~1.6GB RAM per 200K activations.

### More Epochs
Increase `NUM_EPOCHS` from 3 to 10-20 for lower loss and higher explained variance.

### Larger Expansion
Increase `EXPANSION` from 4 to 8 for 16,384 features (doubles SAE size to ~268MB).

### Multiple Layers
Train SAEs on different layers to compare features:
- Early layers (0-8): Syntactic features, token-level patterns
- Middle layers (8-24): Semantic features, abstract concepts
- Late layers (24-31): Task-specific features, output-relevant patterns

### Colab Pro
With Colab Pro ($10/month), you get access to A100 GPUs with 40GB+ VRAM, enabling larger batch sizes, more activations, and bigger expansion factors.

## Troubleshooting

### "No GPU found"
Go to `Runtime` → `Change runtime type` → Select T4 GPU → Save.

### "Session crashed after using all available RAM"
The model loading is already optimized with mmap. If you still hit this:
- Reduce `NUM_ACTIVATIONS` to 25K
- Reduce `COLLECT_BATCH_SIZE` to 1
- Restart runtime and re-run

### "Rust/Cargo not found"
Re-run the Rust installation cell. If it persists, restart runtime and run from the beginning.

### "Module not found" errors
Make sure the path setup cell ran successfully. The repo should be at `/content/nanochat-SAE` with sys.path configured.

### HuggingFace rate limits
If the model download fails, you may need to set your `HF_TOKEN` secret in Colab (Settings → Secrets).

## Resources

- [nanochat-SAE Repository](https://github.com/SolshineCode/nanochat-SAE)
- [nanochat Original](https://github.com/karpathy/nanochat)
- [nanochat-d32 on HuggingFace](https://huggingface.co/karpathy/nanochat-d32)
- [Anthropic: Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features)
- [OpenAI: Scaling SAEs](https://openai.com/research/sparse-autoencoders)

## License

MIT (same as nanochat)

---

**Ready to explore what nanochat-d32 learned?** Click the badge at the top and hit Run all!
