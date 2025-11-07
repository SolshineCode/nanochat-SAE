# Codebase Audit Report - nanochat-SAE
**Date:** November 7, 2025
**Auditor:** Claude Code
**Scope:** Full codebase audit focusing on misconfigurations and alignment with SAELens documentation

---

## Executive Summary

This audit examined the nanochat-SAE codebase to identify misconfigurations and areas where the implementation may not align with current SAELens best practices and documentation. The codebase is well-structured and implements SAE functionality from scratch without depending on the SAELens library. However, several important gaps and potential improvements were identified.

### Critical Findings: 2
### High Priority Findings: 4
### Medium Priority Findings: 4
### Low Priority Findings: 3

---

## Critical Issues

### 1. Missing Activation Normalization Parameter
**Severity:** Critical
**Location:** `sae/config.py`, `sae/models.py`

**Issue:**
The SAELens library includes a critical `normalize_activations` parameter that supports three modes:
- `'none'` - No normalization
- `'expected_average_only_in'` - Anthropic April 2024 update (not yet folded)
- `'constant_norm_rescale'` - Anthropic February 2024 update

This parameter is **completely missing** from the nanochat-SAE implementation.

**Impact:**
- SAEs trained without proper activation normalization may have poor reconstruction quality
- Results may not be comparable to SAELens-trained models
- Neuronpedia uploads may not align with expected formats

**Recommendation:**
```python
# Add to SAEConfig in sae/config.py
@dataclass
class SAEConfig:
    # ... existing fields ...
    normalize_activations: Literal["none", "expected_average_only_in", "constant_norm_rescale"] = "none"
```

Implement normalization in the encoder/decoder pipeline based on the selected mode.

**References:**
- SAELens CHANGELOG v6: "normalize_activations is now a string"
- SAELens documentation on activation normalization

---

### 2. Missing Examples Directory
**Severity:** Critical
**Location:** Root directory

**Issue:**
The README and documentation extensively reference an `examples/` directory with tutorials:
- `examples/train_sae.py`
- `examples/interpret_model.py`
- `examples/feature_steering.py`
- `examples/feature_analysis.py`

However, this directory **does not exist** in the repository.

**Impact:**
- New users cannot follow the documented examples
- Documentation is misleading
- Reduced adoption and usability

**Recommendation:**
Either:
1. Create the examples directory with the referenced files
2. Update documentation to remove references to non-existent examples
3. Add a note that examples are "coming soon" and provide alternative getting-started guides

---

## High Priority Issues

### 3. Incomplete GatedSAE Export
**Severity:** High
**Location:** `sae/__init__.py`

**Issue:**
The `GatedSAE` class is implemented in `sae/models.py` but is NOT exported in `sae/__init__.py`:

```python
# Current __init__.py
from sae.models import TopKSAE, ReLUSAE  # Missing GatedSAE!
```

**Impact:**
- Users cannot import `GatedSAE` using `from sae import GatedSAE`
- Documentation mentions three SAE types, but only two are accessible
- Inconsistent API

**Recommendation:**
```python
# sae/__init__.py
from sae.models import TopKSAE, ReLUSAE, GatedSAE  # Add GatedSAE

__all__ = [
    "SAEConfig",
    "TopKSAE",
    "ReLUSAE",
    "GatedSAE",  # Add to exports
    "ActivationCollector",
    "InterpretableModel",
    "load_saes",
]
```

---

### 4. Missing `create_sae` Function Import
**Severity:** High
**Location:** `sae/trainer.py`, `sae/runtime.py`

**Issue:**
Both `sae/trainer.py` and `sae/runtime.py` import and use `create_sae()` from `sae.models`, but this function is NOT exported in `sae/__init__.py`. Users cannot access this factory function.

**Impact:**
- Users cannot create SAE instances using the factory pattern
- Have to know implementation details to instantiate correct SAE type
- Inconsistent API

**Recommendation:**
```python
# sae/__init__.py
from sae.models import TopKSAE, ReLUSAE, GatedSAE, create_sae  # Add create_sae

__all__ = [
    "SAEConfig",
    "TopKSAE",
    "ReLUSAE",
    "GatedSAE",
    "create_sae",  # Add to exports
    "ActivationCollector",
    "InterpretableModel",
    "load_saes",
]
```

---

### 5. Incorrect Hook Implementation
**Severity:** High
**Location:** `sae/hooks.py:141-150`

**Issue:**
The `_get_module_from_hook_point()` method attempts to hook into nanochat's transformer blocks, but the implementation is incomplete and may not work correctly with the actual nanochat model structure.

```python
# Current implementation
if "hook_resid" in hook_type:
    return block  # Returns entire block
elif "attn" in hook_type:
    return block.attn
elif "mlp" in hook_type:
    return block.mlp
```

The function doesn't distinguish between different residual stream positions (pre/post attention, pre/post MLP).

**Impact:**
- May collect activations from wrong model location
- Inconsistent with hook naming convention
- Could lead to poor SAE performance

**Recommendation:**
Implement proper hook point resolution:

```python
def _get_module_from_hook_point(self, hook_point: str) -> nn.Module:
    """Get module from hook point string."""
    parts = hook_point.split(".")
    if parts[0] != "blocks":
        raise ValueError(f"Invalid hook point: {hook_point}")

    layer_idx = int(parts[1])
    block = self.model.transformer.h[layer_idx]

    if len(parts) < 3:
        raise ValueError(f"Hook point must specify hook type: {hook_point}")

    hook_type = parts[2]

    # Map hook types to actual modules/positions
    if hook_type == "hook_resid_pre":
        # Need to hook before attention
        return block  # May need custom wrapper
    elif hook_type == "hook_resid_post":
        return block  # Hook after entire block
    elif hook_type == "hook_attn_out":
        return block.attn
    elif hook_type == "hook_mlp_out":
        return block.mlp
    else:
        raise ValueError(f"Unknown hook type: {hook_type}")
```

---

### 6. Potential Tensor Device Mismatch in Hooks
**Severity:** Medium-High
**Location:** `sae/hooks.py:96`

**Issue:**
The activation collector moves activations to a target device (line 96):
```python
activation = activation.detach().to(self.device)
```

However, if the collector device is "cpu" and the model is on "cuda", this could cause performance issues during training as tensors are constantly moved between devices during forward passes.

**Impact:**
- Potential memory issues
- Slower activation collection
- May cause CUDA OOM errors

**Recommendation:**
Consider keeping activations on GPU during collection and only moving to CPU when storing:
```python
# Collect on GPU, move to CPU for storage
activation = activation.detach()
if self.device == "cpu":
    activation = activation.cpu()
```

---

## Medium Priority Issues

### 8. Missing SAELens v6 Configuration Parameters
**Severity:** Medium
**Location:** `sae/config.py`

**Issue:**
SAELens v6 introduced several new configuration parameters that are missing:
- `prepend_bos` - Whether to prepend BOS token (from v6.14.1)
- Support for temporal SAEs (v6.20.0)
- MatryoshkaBatchTopKSAE configuration (v6.15.0)

**Impact:**
- Cannot train temporal SAEs
- Missing advanced features
- Reduced compatibility with SAELens ecosystem

**Recommendation:**
Review SAELens v6 CHANGELOG and add relevant configuration options for advanced users.

---

### 9. Missing Evaluation Metrics
**Severity:** Medium
**Location:** `sae/evaluator.py`

**Issue:**
SAELens recommends tracking specific metrics prominently:
- `CE_Loss_score` - Cross-entropy loss score
- Feature dashboards should prioritize: L0, CE_Loss_score, explained variance

The current implementation tracks MSE and explained variance but not CE loss impact.

**Impact:**
- Cannot assess impact of SAE on downstream task performance
- Incomplete evaluation

**Recommendation:**
Add CE loss evaluation:

```python
def evaluate_ce_loss_impact(
    self,
    sae: BaseSAE,
    model: nn.Module,
    dataloader: DataLoader,
) -> float:
    """Evaluate impact of SAE reconstruction on model CE loss."""
    pass
```

---

### 10. Random Data in Training Script
**Severity:** Medium
**Location:** `scripts/sae_train.py:65-112`

**Issue:**
The `collect_activations_simple()` function uses **random tokens** for activation collection:

```python
# Line 92-97
tokens = torch.randint(
    0,
    model.config.vocab_size,
    (batch_size, sequence_length),
    device=device
)
```

**Impact:**
- SAEs trained on random data will not learn meaningful features
- Misleading for users who expect real data
- Comment says "in production, use real data" but provides no guidance

**Recommendation:**
1. Add a clear warning in the script output
2. Provide a separate example using real data from nanochat's dataloader
3. Consider making this function throw an error or warning by default

---

### 11. Missing Type Hints
**Severity:** Medium
**Location:** Multiple files

**Issue:**
Several functions lack proper type hints:
- Return types not specified consistently
- Some parameter types missing

**Impact:**
- Reduced code clarity
- Harder to catch type errors
- IDE support limited

**Recommendation:**
Add comprehensive type hints throughout the codebase, especially in public APIs.

---

## Low Priority Issues

### 12. Documentation Inconsistencies
**Severity:** Low
**Location:** `README.md`, `SAE_README.md`

**Issue:**
Several inconsistencies in documentation:
- README references features as "coming soon" without timeline
- Citation information has placeholder `[Your Name]` and `[YourUsername]`
- Some code examples reference non-existent files

**Impact:**
- Looks unprofessional
- May confuse users

**Recommendation:**
- Fill in all placeholder values
- Add clear status indicators for incomplete features
- Verify all code examples

---

### 13. Missing WandB Integration
**Severity:** Low
**Location:** `sae/trainer.py`, `pyproject.toml`

**Issue:**
`wandb>=0.21.3` is listed as a dependency in `pyproject.toml:19`, but there's no evidence of WandB integration in the SAE training code.

**Impact:**
- Unused dependency
- Missing experiment tracking

**Recommendation:**
Either:
1. Add WandB logging to SAE trainer
2. Remove WandB from dependencies if not used

---

### 14. Test Coverage for SAE Scripts
**Severity:** Low
**Location:** `tests/`

**Issue:**
`tests/test_sae.py` tests the core SAE implementation but doesn't test:
- `scripts/sae_train.py`
- `scripts/sae_eval.py`
- `scripts/sae_viz.py`

**Impact:**
- Integration issues may not be caught
- Scripts could break without notice

**Recommendation:**
Add integration tests for the training/evaluation/visualization scripts.

---

## Positive Findings

### Strengths of the Codebase

1. **Clean Architecture:** SAE module is well-separated from nanochat core
2. **Good Documentation:** Comprehensive README files with clear examples
3. **Multiple SAE Variants:** Supports TopK, ReLU, and Gated SAEs
4. **Proper Testing:** Core SAE functionality has good test coverage
5. **Type Safety:** Uses dataclasses and type hints in many places
6. **Neuronpedia Integration:** Thoughtful preparation for community sharing

---

## Recommendations Summary

### Immediate Actions (Critical)
1. ✅ Implement activation normalization parameter and logic
2. ✅ Create examples directory or update documentation
3. ✅ Export GatedSAE in `__init__.py`
4. ✅ Implement missing functions: `train_sae_from_activations`, `save_sae`, `load_saes`

### Short-term Actions (High Priority)
1. ✅ Fix hook implementation for correct activation collection
2. ✅ Verify dead latent resampling is implemented
3. ✅ Add warning about random data in training script
4. ✅ Test the training pipeline end-to-end

### Medium-term Actions
1. ⚠️ Add CE loss evaluation metrics
2. ⚠️ Review SAELens v6 features and add relevant ones
3. ⚠️ Add comprehensive type hints
4. ⚠️ Add WandB integration or remove dependency

### Long-term Actions
1. 📝 Create tutorial notebooks
2. 📝 Add integration tests for scripts
3. 📝 Consider optional SAELens compatibility layer
4. 📝 Document differences from SAELens

---

## Conclusion

The nanochat-SAE codebase is well-designed and implements SAE functionality from scratch. However, it has several critical missing implementations that would prevent the training scripts from working correctly. The most important issues to address are:

1. Missing core functions that are called but not implemented
2. Activation normalization alignment with SAELens best practices
3. Missing examples directory referenced in documentation
4. Incomplete hook implementation

Once these issues are addressed, the codebase will be much more robust and aligned with SAELens documentation and best practices.

---

## References

- [SAELens GitHub Repository](https://github.com/jbloomAus/SAELens)
- [SAELens Documentation](https://jbloomaus.github.io/SAELens)
- [SAELens v6 CHANGELOG](https://github.com/jbloomAus/SAELens/blob/main/CHANGELOG.md)
- [Neuronpedia Documentation](https://docs.neuronpedia.org)
- [Scaling and Evaluating Sparse Autoencoders (OpenAI)](https://arxiv.org/abs/2406.04093)
