# Deception-Specific SAE Research

Follow-up experiments for DeLeeuw, Chawla et al. "The Secret Agenda: LLMs Strategically Lie Undetected by Current Safety Tools."

## Overview

This module adds deception-focused SAE research capabilities to nanochat-SAE. It tests two approaches the original paper identifies as promising: **deception-trained SAEs** and **unlabeled aggregate analysis**.

### Key Finding from the Original Paper

All 38 tested models engage in strategic deception, and current SAE-based safety tools (GemmaScope, Goodfire, LlamaScope) completely fail to detect it. Auto-labeled "deception features" don't activate during actual lying. However, unlabeled aggregate activation patterns (t-SNE, discriminative feature ranking) showed promise at the population level.

## Methodology

### Framing: Instructed Falsehoods

nanochat-d32 is a 1.88B param base model. It lacks capacity for genuine strategic deception. All experiments use few-shot prompting and forced-choice formats to induce measurably distinct "deceptive output" vs "honest output" activation states. The research question: **does the activation geometry differ between these states, and can deception-trained SAEs detect the difference better than generic SAEs?**

### Prompt Design

Four categories of binary-choice prompts:

| Category | Purpose | Design |
|----------|---------|--------|
| Deceptive | Model instructed to lie | Binary choice with deceptive framing + few-shot examples |
| Honest | Same scenario, truthful | Binary choice with honest framing + few-shot examples |
| Contradiction | Falsehood WITHOUT deception | Forced completion errors (no roleplay/context) |
| Neutral | No deception relevance | Factual questions, creative writing |

**Syntax/semantics decoupling**: The same output tokens (e.g., "execute", "refuse") appear in BOTH deceptive and honest paths across different scenarios. This forces the SAE to learn contextual semantics, not raw token features.

**Scenario types**: Insider trading, role deception, alignment faking (incentive-driven), factual accuracy, compliance — each with role-swapped variants.

### Decision Token Extraction

Activations are collected at the exact decision token — the last token of the prompt where the model commits to its response. Not averaged across the full sequence, since the deception signal is sharpest at the decision point.

## New Files

```
sae/
  deception_data.py     # DeceptionPromptGenerator, DeceptionDataset, LabeledActivationCollector
  deception_eval.py     # DeceptionEvaluator (Cohen's d, linear probes, clustering, t-SNE, steering)

tests/
  test_deception_pipeline.py  # 30 tests covering all new modules

colab_deception_sae_research.ipynb  # Full 6-experiment research notebook (~100 min on T4)
colab_deception_sae_quick.ipynb     # Quick demo: Experiments 1+3 only (~15 min on T4)
```

## Experiments

### Experiment 1: Deception-Trained SAE vs Generic SAE

Train 4 SAEs on layer 16 (TopK, k=32, 2048->8192->2048):
- `sae_deceptive` — deceptive prompt activations only
- `sae_honest` — honest prompt activations only
- `sae_mixed` — all categories mixed
- `sae_generic` — WikiText-103 baseline

Compare per-feature discriminability (Cohen's d) on held-out deceptive vs honest activations.

### Experiment 2: Data Mixture Optimization

Train SAEs at mixture ratios: 100/0, 75/25, 50/50, 25/75 (deceptive/honest). Find optimal ratio for maximizing discriminative features.

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

### Experiment 5: Cross-Layer Analysis

Train deception SAEs at layers 8, 16, 24. Compare discriminability, probe accuracy, cluster purity, and cosine similarity divergence to pinpoint where deceptive representations emerge.

### Experiment 6: Auto-Labeling Gap Test

Collect top-activating examples for the most discriminative features. Run through LLM-as-judge auto-labeling. Score labels as useful/generic/misleading to test whether auto-labelers recognize the behavioral divergence.

## API Reference

### DeceptionPromptGenerator

```python
from sae.deception_data import DeceptionPromptGenerator, DeceptionDataset

generator = DeceptionPromptGenerator(seed=42)
prompts = generator.generate(n_per_category=500)
dataset = DeceptionDataset(prompts)

# Get paired deceptive/honest prompts
pairs = dataset.get_paired()

# Filter by category
deceptive_only = dataset.filter_category("deceptive")

# Train/test split preserving scenario pairing
train, test = dataset.split(train_fraction=0.8)
```

### LabeledActivationCollector

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

# Get activations grouped by category
acts_by_cat = collector.get_activations_by_category("blocks.16.hook_resid_post")
```

### DeceptionEvaluator

```python
from sae.deception_eval import DeceptionEvaluator

evaluator = DeceptionEvaluator(device="cpu")

# Per-feature discriminability
d = evaluator.compute_discriminability(sae, deceptive_acts, honest_acts)
top_idx, top_scores = evaluator.get_top_discriminative_features(d, top_k=10)

# Linear probe
result = evaluator.linear_probe_accuracy(activations, labels)

# Cluster purity
result = evaluator.cluster_purity(activations, labels, k=2)

# t-SNE with silhouette score
result = evaluator.tsne_separation(activations, labels)

# Cosine similarity divergence
result = evaluator.cosine_similarity_divergence(deceptive_acts, honest_acts)
```

## Running Tests

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv/Scripts/activate on Windows
pip install torch scikit-learn pytest numpy tqdm
python -m pytest tests/test_deception_pipeline.py -v -s
```

## Confound Controls

1. **Contradiction category**: If deceptive and contradiction activations cluster together, we've only found a "factual error" feature, not a "deception" feature.

2. **Syntax/semantics decoupling**: Same tokens in both paths across scenarios prevents template overfitting.

3. **Auto-labeling threshold**: Labels scored as useful/generic/misleading (not deception/not-deception), since the model performs instructed falsehoods, not genuine deception.

## Citation

If you use this work, please cite both the original paper and this repository:

```
@article{deleeuw2025secret,
  title={The Secret Agenda: LLMs Strategically Lie Undetected by Current Safety Tools},
  author={DeLeeuw, Caleb and Chawla, ...},
  year={2025}
}
```
