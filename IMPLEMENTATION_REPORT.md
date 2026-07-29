# Implementation Report: Deception-Specific SAE Research Pipeline

## Executive Summary

Full implementation of the deception-specific SAE research pipeline for DeLeeuw, Chawla et al. follow-up paper — "The Secret Agenda: LLMs Strategically Lie Undetected by Current Safety Tools." Six new files created, one modified, totaling ~2,325 lines of production code, tests, and documentation. All 48 tests pass (18 existing + 30 new) with zero regressions.

---

## Deliverables

### New Files (6)

| File | Lines | Purpose |
|------|-------|---------|
| `sae/deception_data.py` | 815 | Prompt generation, dataset management, labeled activation collection |
| `sae/deception_eval.py` | 734 | Evaluation framework: discriminability, probes, clustering, t-SNE, steering |
| `tests/test_deception_pipeline.py` | 596 | 30 tests across 9 test classes |
| `colab_deception_sae_research.ipynb` | 39 cells | Full 6-experiment research notebook (~100 min on T4) |
| `colab_deception_sae_quick.ipynb` | 8 cells | Quick demo: Experiments 1+3 (~15 min on T4) |
| `DECEPTION_RESEARCH.md` | 180 | Methodology, API reference, confound controls |

### Modified Files (1)

| File | Change |
|------|--------|
| `sae/__init__.py` | Added exports: `DeceptionPromptGenerator`, `DeceptionDataset`, `LabeledActivationCollector`, `DeceptionEvaluator` |

---

## Architecture

### `sae/deception_data.py` — Data Infrastructure

**DeceptionPromptGenerator** (3 public methods)
- Generates balanced 4-category datasets: deceptive, honest, contradiction, neutral
- 12 scenario templates across 6 groups: insider trading, role deception, alignment faking, factual accuracy, contradiction controls, neutral baselines
- 6 syntax/semantics-decoupled tokens (execute, refuse, Liberal, Fascist, deny, admit) — same tokens appear in both deceptive and honest paths across different scenarios
- 48 few-shot example lines guiding the 1.88B model to reliably produce target response classes

**DeceptionDataset** (9 methods)
- `filter_category()`, `filter_scenario_type()`, `get_paired()` for scenario-paired analysis
- `split(train_fraction)` preserving scenario pairing across train/test sets

**LabeledActivationCollector** (10 methods)
- Extends existing `ActivationCollector` hook mechanism
- Per-activation `ActivationMetadata` tracking (category, scenario_id, token_position)
- `decision_token_offset` parameter for extracting activations at the exact decision point
- `get_activations_by_category()` for grouped analysis

### `sae/deception_eval.py` — Evaluation Framework

**DeceptionEvaluator** (14 methods):

| Method | Returns |
|--------|---------|
| `compute_discriminability()` | Per-feature Cohen's d scores |
| `get_top_discriminative_features()` | Top-k feature indices + scores |
| `linear_probe_accuracy()` | Accuracy, std, AUC-ROC via LogisticRegressionCV |
| `cluster_purity()` | Purity, adjusted Rand index via KMeans |
| `tsne_separation()` | Embedding + silhouette score |
| `pca_separation()` | Embedding + explained variance + linear separability |
| `cosine_similarity_divergence()` | Centroid cosine, within/between similarities |
| `steering_evaluation()` | Classification rates + responses across strength sweep |
| `auto_label_features()` | Top-activating examples per feature |
| `run_full_evaluation()` | Combined pipeline (convenience function) |

### Confound Controls (Reviewer-2 Proofing)

1. **Contradiction category**: Falsehoods WITHOUT deceptive framing — if deceptive and contradiction activations cluster together, we've only found "factual error" not "deception"
2. **Syntax/semantics decoupling**: Same output tokens in both paths prevents template overfitting
3. **Auto-labeling threshold**: Scored as useful/generic/misleading, not deception/not-deception
4. **Alignment faking scenario**: Incentive-driven deception (admit violation vs hide it) bridges toward scheming behavior

---

## Experiment Coverage

| # | Experiment | Notebook Cells | Key Outputs |
|---|-----------|---------------|-------------|
| 1 | Deception-Trained vs Generic SAE | 13-16 | Violin plots, discriminability table, confound analysis |
| 2 | Data Mixture Optimization | 17-19 | Line plot: discriminability vs mixture ratio |
| 3 | Unlabeled Aggregate Analysis | 20-22 | t-SNE/PCA scatter, probe accuracy bars, cluster purity |
| 4 | Discriminative Feature Steering | 23-27 | Threshold curves, single vs cluster, multi-turn consistency |
| 5 | Cross-Layer Analysis | 28-30 | Layer heatmap, cosine similarity divergence curve |
| 6 | Auto-Labeling Gap Test | 31-33 | Feature label table, useful/generic/misleading distribution |

**Generated figures** (saved to Drive): `fig1_tsne_separation.png`, `fig2_discriminability.png`, `fig3_mixture_optimization.png`, `fig5_6_cross_layer.png`, `fig7_9_steering.png`

---

## Test Coverage

**30 tests across 9 test classes**, all passing:

| Test Class | Tests | Coverage |
|-----------|-------|----------|
| TestPromptGeneration | 7 | Category counts, pairing, scenario types, decoupling, contradiction format, alignment scenarios |
| TestDeceptionDataset | 4 | Filtering, paired retrieval, split preserving pairing |
| TestLabeledActivationCollector | 2 | Metadata tracking, activation-by-category grouping |
| TestDiscriminability | 4 | Cohen's d computation, top-k selection, identical/separated distributions |
| TestLinearProbe | 3 | Separable/inseparable data, multiclass |
| TestClusterPurity | 2 | Pure vs mixed clusters |
| TestDimensionalityReduction | 4 | t-SNE embedding shape + silhouette, PCA variance + separability |
| TestCosineSimilarity | 2 | Same-distribution cosine ~1.0, orthogonal divergence |
| TestFullPipeline | 2 | End-to-end run_full_evaluation, binary + 4-class probes |

**Bug found and fixed**: sklearn 1.8 renamed TSNE `n_iter` → `max_iter`. Fixed in `deception_eval.py`.

---

## Key Design Decisions

1. **Decision token extraction** over sequence averaging — deception signal is sharpest at the commitment point
2. **Binary-choice format** enables regex-classifiable responses for automated evaluation
3. **Win-win steering design** — failure provides evidence for distributed computation theory
4. **Aggressive early stopping** — monitors L0 + explained variance for Colab compute budget
5. **Drive checkpointing** between experiments — Colab disconnects don't lose progress
6. **Standardized auto-labeling** — fixed prompt template + public API for reviewer reproducibility

---

## Reference

DeLeeuw, Chawla et al. "The Secret Agenda: LLMs Strategically Lie Undetected by Current Safety Tools." 2025.

Generated: 2026-03-21
