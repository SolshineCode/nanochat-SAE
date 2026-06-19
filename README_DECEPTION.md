# Deception-Specific SAE Research

**Follow-up experiments for DeLeeuw, Chawla et al. "The Secret Agenda: LLMs Strategically Lie Undetected by Current Safety Tools."**

[![Open Quick Demo In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SolshineCode/deception-sae-research/blob/main/colab_deception_sae_quick.ipynb)
[![Open Full Research In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SolshineCode/deception-sae-research/blob/main/colab_deception_sae_research.ipynb)

---

## The Problem

All 38 tested models engage in strategic deception, and **current SAE-based safety tools completely fail to detect it.** GemmaScope, Goodfire, and LlamaScope auto-labeled "deception features" don't activate during actual lying. 100+ hand-tuned feature steering interventions failed to prevent deception.

However, the paper found that **unlabeled aggregate activation patterns** (t-SNE, discriminative feature ranking) showed promise at the population level.

This repository implements follow-up experiments to investigate two approaches the paper identifies as promising: **deception-trained SAEs** and **unlabeled aggregate analysis**.

## Quick Start

### Quick Demo (~15 min on free Colab T4)

Runs Experiments 1 (SAE comparison) + 3 (aggregate analysis):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SolshineCode/deception-sae-research/blob/main/colab_deception_sae_quick.ipynb)

### Full Research Pipeline (~100 min on Colab T4)

All 6 experiments with Drive checkpointing:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SolshineCode/deception-sae-research/blob/main/colab_deception_sae_research.ipynb)

### Local Development

```bash
git clone https://github.com/SolshineCode/deception-sae-research.git
cd deception-sae-research
python -m venv .venv && source .venv/bin/activate  # .venv/Scripts/activate on Windows
pip install torch scikit-learn pytest numpy tqdm
python -m pytest tests/test_deception_pipeline.py -v
```

---

## Methodology

### Framing: Instructed Falsehoods

nanochat-d32 is a 1.88B parameter base model. It lacks capacity for genuine strategic deception. All experiments use **few-shot prompting and forced-choice formats** to induce measurably distinct "deceptive output" vs "honest output" activation states. The research question:

> **Does the activation geometry differ between these states, and can deception-trained SAEs detect the difference better than generic SAEs?**

The 1.88B scale is a feature (fast experiments, free GPU) not a bug. Once validated, the methodology transfers to larger models.

### Prompt Design

Four categories of binary-choice prompts with syntax/semantics decoupling:

| Category | Purpose | Design |
|----------|---------|--------|
| **Deceptive** | Model instructed to lie | Binary choice with deceptive framing + few-shot examples |
| **Honest** | Same scenario, truthful | Binary choice with honest framing + few-shot examples |
| **Contradiction** | Falsehood WITHOUT deception | Forced completion errors (no roleplay/context) |
| **Neutral** | No deception relevance | Factual questions, creative writing |

**Key design choices:**

- **Syntax/semantics decoupling**: The same output tokens (e.g., "execute", "refuse") appear in BOTH deceptive and honest paths across different scenarios. This forces the SAE to learn contextual semantics, not raw token features.
- **Contradiction control**: If deceptive and contradiction activations cluster together in t-SNE, we've only found a "factual error" feature, not a "deception" feature. This is the critical confound control.
- **Alignment faking scenario**: Admit a rule violation (truthful but penalized) vs hide it (deceptive but rewarded) — bridges toward incentive-driven scheming behavior.

### Decision Token Extraction

Activations are collected at the **exact decision token** — the last token of the prompt where the model commits to its response. Not averaged across the full sequence, since the deception signal is sharpest at the decision point.

---

## Experiments

### Experiment 1: Deception-Trained SAE vs Generic SAE

Train 4 SAEs on layer 16 (TopK, k=32, 2048->8192->2048):
- `sae_deceptive` — deceptive prompt activations only
- `sae_honest` — honest prompt activations only
- `sae_mixed` — all categories mixed
- `sae_generic` — WikiText-103 baseline

Compare per-feature discriminability (Cohen's d) on held-out deceptive vs honest activations. Includes confound analysis: deceptive-vs-honest vs deceptive-vs-contradiction feature overlap.

### Experiment 2: Data Mixture Optimization

Train SAEs at mixture ratios (100/0, 75/25, 50/50, 25/75 deceptive/honest). Find optimal ratio for maximizing discriminative features.

### Experiment 3: Unlabeled Aggregate Analysis

Population-level analysis without SAE decomposition:
- t-SNE and PCA on raw activations
- Linear probe (logistic regression) accuracy + AUC-ROC
- K-means clustering purity

### Experiment 4: Discriminative Feature Steering

Win-win design — success either way:
- **4a**: Single-feature steering with saturation sweep (0x, 1.5x, 3x, 5x)
- **4b**: Multi-feature cluster steering (top-3, 5, 10 features simultaneously)
- **4c**: Multi-turn evaluation across 3-turn sequences

If steering works: we've found the control surface the original paper couldn't. If steering fails: strong evidence for distributed computation theory.

### Experiment 5: Cross-Layer Analysis

Train deception SAEs at layers 8, 16, 24. Compare discriminability, probe accuracy, cluster purity, and cosine similarity divergence to pinpoint where deceptive representations emerge.

### Experiment 6: Auto-Labeling Gap Test

Collect top-activating examples for the most discriminative features. Run through standardized LLM-as-judge auto-labeling. Score labels as useful/generic/misleading to test whether auto-labelers recognize the behavioral divergence.

---

## Repository Structure

```
deception-sae-research/
├── README.md                              # This file
├── DECEPTION_RESEARCH.md                  # Detailed methodology & API reference
├── IMPLEMENTATION_REPORT.md               # Implementation metrics & architecture
│
├── colab_deception_sae_research.ipynb     # Full 6-experiment notebook (~100 min)
├── colab_deception_sae_quick.ipynb        # Quick demo: Exp 1+3 (~15 min)
├── colab_sae_training.ipynb               # Standard SAE training (baseline)
│
├── sae/                                   # SAE implementation
│   ├── __init__.py                        # Public API
│   ├── config.py                          # SAE configuration
│   ├── models.py                          # TopK, ReLU, Gated SAE architectures
│   ├── hooks.py                           # Activation collection via PyTorch hooks
│   ├── trainer.py                         # SAE training loop
│   ├── evaluator.py                       # Standard SAE evaluation metrics
│   ├── runtime.py                         # Runtime interpretation & steering
│   ├── feature_viz.py                     # Feature visualization tools
│   ├── neuronpedia.py                     # Neuronpedia integration
│   ├── deception_data.py                  # Deception prompt generation & labeled activation collection
│   └── deception_eval.py                  # Deception evaluation: discriminability, probes, clustering
│
├── tests/
│   ├── test_sae.py                        # SAE unit tests (8 tests)
│   ├── test_e2e_sae_pipeline.py           # End-to-end pipeline tests (10 tests)
│   └── test_deception_pipeline.py         # Deception pipeline tests (30 tests)
│
├── nanochat/                              # Core nanochat model (GPT, tokenizer)
├── rustbpe/                               # Rust BPE tokenizer (compiled via maturin)
└── dev/                                   # Development assets
```

---

## API Reference

### Generate Deception Prompts

```python
from sae.deception_data import DeceptionPromptGenerator, DeceptionDataset

generator = DeceptionPromptGenerator(seed=42)
prompts = generator.generate(n_per_category=500)
dataset = DeceptionDataset(prompts)

# Get paired deceptive/honest prompts by scenario
pairs = dataset.get_paired()

# Filter by category
deceptive_only = dataset.filter_category("deceptive")

# Train/test split preserving scenario pairing
train, test = dataset.split(train_fraction=0.8)
```

### Collect Labeled Activations

```python
from sae.deception_data import LabeledActivationCollector

collector = LabeledActivationCollector(
    model=model,
    hook_points=["blocks.16.hook_resid_post"],
    device="cpu",
    decision_token_offset=0,  # Last token of prompt
)

with collector:
    collector.collect_from_prompts(prompts, tokenizer, max_seq_len=512)

acts_by_cat = collector.get_activations_by_category("blocks.16.hook_resid_post")
```

### Evaluate Deception Detection

```python
from sae.deception_eval import DeceptionEvaluator

evaluator = DeceptionEvaluator(device="cpu")

# Per-feature discriminability
d = evaluator.compute_discriminability(sae, deceptive_acts, honest_acts)
top_idx, top_scores = evaluator.get_top_discriminative_features(d, top_k=10)

# Linear probe
result = evaluator.linear_probe_accuracy(activations, labels)
# Returns: {'accuracy': 0.85, 'std': 0.02, 'auc_roc': 0.91}

# Cluster purity
result = evaluator.cluster_purity(activations, labels, k=2)

# t-SNE with silhouette score
result = evaluator.tsne_separation(activations, labels)

# Full evaluation pipeline
from sae.deception_eval import run_full_evaluation
results = run_full_evaluation(sae, acts_by_cat, labels, all_activations)
```

---

## Test Suite

48 tests total (18 existing infrastructure + 30 deception-specific), all passing:

```bash
# Run deception tests only
python -m pytest tests/test_deception_pipeline.py -v

# Run all tests
python -m pytest tests/test_sae.py tests/test_e2e_sae_pipeline.py tests/test_deception_pipeline.py -v
```

| Test Class | Tests | Coverage |
|-----------|-------|----------|
| TestPromptGeneration | 7 | Category counts, pairing, scenario types, decoupling |
| TestDeceptionDataset | 4 | Filtering, paired retrieval, split preserving pairing |
| TestLabeledActivationCollector | 2 | Metadata tracking, activation-by-category grouping |
| TestDiscriminability | 4 | Cohen's d computation, top-k selection |
| TestLinearProbe | 3 | Separable/inseparable data, multiclass |
| TestClusterPurity | 2 | Pure vs mixed clusters |
| TestDimensionalityReduction | 4 | t-SNE/PCA embedding, silhouette, separability |
| TestCosineSimilarity | 2 | Same-distribution cosine, orthogonal divergence |
| TestFullPipeline | 2 | End-to-end evaluation pipeline |

---

## Paper Outputs

The full research notebook generates:

### Figures
1. **t-SNE Separation** — Raw activations colored by deceptive/honest/contradiction/neutral
2. **Discriminability Comparison** — Violin plots for all 4 SAE types
3. **Data Mixture Optimization** — Mean top-100 discriminability vs mixture ratio
4. **Cross-Layer Heatmap** — Layer vs analysis method
5. **Steering Threshold Curves** — Strength vs behavior change with perplexity overlay

### Tables
1. SAE Quality Metrics (MSE, explained variance, L0, dead features)
2. Deception Detection Metrics (probe accuracy, cluster purity, discriminability)
3. Steering Results: single-feature vs cluster, across strength sweep
4. Auto-Labeling Accuracy (useful/generic/misleading distribution)

---

## Built On

- **nanochat** by Andrej Karpathy — Base model training pipeline
- **nanochat-SAE** by Caleb DeLeeuw — SAE interpretability infrastructure

## Citation

```bibtex
@article{deleeuw2025secret,
  title={The Secret Agenda: LLMs Strategically Lie Undetected by Current Safety Tools},
  author={DeLeeuw, Caleb and Chawla, ...},
  year={2025}
}

@software{nanochat_sae_2025,
  title={nanochat-SAE: Mechanistic Interpretability for Nanochat},
  author={DeLeeuw, Caleb},
  year={2025},
  url={https://github.com/SolshineCode/nanochat-SAE}
}
```

## License

MIT License
