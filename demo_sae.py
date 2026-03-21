"""
Live demo: Train an SAE on a nanochat model and explore what it learns.
"""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from nanochat.gpt import GPT, GPTConfig
from sae.config import SAEConfig
from sae.models import TopKSAE
from sae.hooks import ActivationCollector
from sae.trainer import SAETrainer
from sae.evaluator import SAEEvaluator
from sae.feature_viz import FeatureVisualizer, generate_sae_summary
from sae.runtime import InterpretableModel

torch.manual_seed(42)

# ── Step 1: Create a small nanochat GPT model ──────────────────────────────
print("=" * 70)
print("STEP 1: Creating nanochat GPT model")
print("=" * 70)

model_config = GPTConfig(
    sequence_len=128,
    vocab_size=512,
    n_layer=6,
    n_head=4,
    n_kv_head=4,
    n_embd=128,
)

model = GPT(model_config)
model.init_weights()
# Re-randomize lm_head (init_weights zeros it for training stability)
torch.nn.init.normal_(model.lm_head.weight, std=0.02)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"  Architecture: {model_config.n_layer} layers, {model_config.n_head} heads, d={model_config.n_embd}")
print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
print(f"  Vocab size: {model_config.vocab_size}")
print()

# ── Step 2: Collect activations from layer 3 ───────────────────────────────
print("=" * 70)
print("STEP 2: Collecting activations from layer 3")
print("=" * 70)

layer_idx = 3
hook_point = f"blocks.{layer_idx}.hook_resid_post"

collector = ActivationCollector(
    model=model,
    hook_points=[hook_point],
    max_activations=10_000,
    device="cpu",
)

t0 = time.time()
with torch.no_grad(), collector:
    for i in range(20):
        tokens = torch.randint(0, model_config.vocab_size, (8, model_config.sequence_len))
        model(tokens)
        if (i + 1) % 5 == 0:
            print(f"  Batch {i+1}/20 — collected {collector.counts[hook_point]:,} activations")

activations = collector.get_activations()[hook_point]
elapsed = time.time() - t0
print(f"  Done! {activations.shape[0]:,} activations of dimension {activations.shape[1]} in {elapsed:.1f}s")
print(f"  Activation stats: mean={activations.mean():.4f}, std={activations.std():.4f}")
print()

# ── Step 3: Train a TopK SAE ───────────────────────────────────────────────
print("=" * 70)
print("STEP 3: Training TopK SAE (expansion=8x, k=16)")
print("=" * 70)

sae_config = SAEConfig(
    d_in=model_config.n_embd,       # 128
    expansion_factor=8,              # d_sae = 1024
    activation="topk",
    k=16,
    hook_point=hook_point,
    batch_size=256,
    num_epochs=5,
    learning_rate=3e-4,
)

print(f"  SAE dimensions: {sae_config.d_in} -> {sae_config.d_sae} ({sae_config.expansion_factor}x expansion)")
print(f"  Sparsity: top-{sae_config.k} features active per activation")
print(f"  Training: {sae_config.num_epochs} epochs, batch_size={sae_config.batch_size}")
print()

# Split data
n_val = 1000
train_acts = activations[n_val:]
val_acts = activations[:n_val]

sae = TopKSAE(sae_config)
trainer = SAETrainer(
    sae=sae,
    config=sae_config,
    activations=train_acts,
    val_activations=val_acts,
    device="cpu",
)

t0 = time.time()
losses = []
for epoch in range(sae_config.num_epochs):
    metrics = trainer.train_epoch(verbose=False)
    losses.append(metrics["total_loss"])
    print(f"  Epoch {epoch+1}/{sae_config.num_epochs} — loss: {metrics['total_loss']:.6f}  L0: {metrics['l0']:.1f}")

elapsed = time.time() - t0
pct_reduction = (1 - losses[-1] / losses[0]) * 100
print(f"  Training complete in {elapsed:.1f}s")
print(f"  Loss reduction: {losses[0]:.6f} -> {losses[-1]:.6f} ({pct_reduction:.1f}%)")
print()

# ── Step 4: Evaluate SAE quality ───────────────────────────────────────────
print("=" * 70)
print("STEP 4: Evaluating SAE quality")
print("=" * 70)

evaluator = SAEEvaluator(sae, sae_config)
eval_metrics = evaluator.evaluate(val_acts, compute_dead_latents=True)

print(f"  Reconstruction MSE:    {eval_metrics.mse_loss:.6f}")
print(f"  Explained Variance:    {eval_metrics.explained_variance:.4f} ({eval_metrics.explained_variance*100:.1f}%)")
print(f"  Reconstruction Score:  {eval_metrics.reconstruction_score:.4f}")
print(f"  L0 (avg active):      {eval_metrics.l0_mean:.1f} ± {eval_metrics.l0_std:.1f}")
print(f"  Dead latents:          {eval_metrics.dead_latent_fraction*100:.1f}% ({int(eval_metrics.dead_latent_fraction * sae_config.d_sae)}/{sae_config.d_sae})")
print(f"  Max activation:        {eval_metrics.max_activation:.4f}")
print(f"  Mean activation:       {eval_metrics.mean_activation:.4f}")
print()

# ── Step 5: Explore discovered features ────────────────────────────────────
print("=" * 70)
print("STEP 5: Exploring discovered features")
print("=" * 70)

viz = FeatureVisualizer(sae, sae_config)
top_indices, top_freqs = viz.get_top_features(val_acts, k=20)

print(f"\n  Top 20 most active features (out of {sae_config.d_sae}):")
print(f"  {'Rank':<6} {'Feature':<10} {'Frequency':<12} {'Mean When Active':<18} {'Max Activation':<15}")
print(f"  {'-'*60}")

for i, (idx, freq) in enumerate(zip(top_indices[:20], top_freqs[:20])):
    idx_val = idx.item()
    stats = viz.get_feature_statistics(idx_val, val_acts)
    print(f"  {i+1:<6} #{idx_val:<9} {freq:.4f}       {stats['mean_when_active']:<18.4f} {stats['max_activation']:<15.4f}")

print()

# ── Step 6: Runtime interpretation ─────────────────────────────────────────
print("=" * 70)
print("STEP 6: Runtime interpretation — tracking features during inference")
print("=" * 70)

saes = {hook_point: sae}
interp_model = InterpretableModel(model, saes, device="cpu")

tokens = torch.randint(0, model_config.vocab_size, (1, 32))

with interp_model.interpretation_enabled():
    with torch.no_grad():
        logits = interp_model(tokens)
    features = interp_model.get_active_features()

feature_tensor = features[hook_point]
print(f"  Input: {tokens.shape[1]} tokens")
print(f"  Feature tensor shape: {feature_tensor.shape}")
print(f"  Active features per position: {(feature_tensor != 0).sum(dim=-1).float().mean():.1f}")

# Show which features fired for the first few positions
print(f"\n  Feature activations for first 5 token positions:")
for pos in range(min(5, feature_tensor.shape[0])):
    active = torch.nonzero(feature_tensor[pos]).squeeze(-1)
    vals = feature_tensor[pos, active]
    top5 = vals.argsort(descending=True)[:5]
    feat_strs = [f"#{active[j].item()}({vals[j]:.2f})" for j in top5]
    print(f"    Position {pos}: {', '.join(feat_strs)}")

print()

# ── Step 7: Feature steering ──────────────────────────────────────────────
print("=" * 70)
print("STEP 7: Feature steering — modifying model behavior")
print("=" * 70)

# Pick the most active feature
steer_feature = top_indices[0].item()

with torch.no_grad():
    baseline_logits = interp_model(tokens)

strengths = [0.0, 1.0, 2.0, 5.0, 10.0]
print(f"\n  Steering feature #{steer_feature} at different strengths:")
print(f"  {'Strength':<12} {'Mean Logit Diff':<18} {'Max Logit Diff':<18} {'Top Token Changed?'}")
print(f"  {'-'*65}")

for strength in strengths:
    with torch.no_grad():
        steered_logits = interp_model.steer(
            tokens,
            feature_id=(hook_point, steer_feature),
            strength=strength,
        )
    diff = (steered_logits - baseline_logits).abs()
    top_changed = (steered_logits[:, -1].argmax() != baseline_logits[:, -1].argmax()).item()
    print(f"  {strength:<12.1f} {diff.mean().item():<18.4f} {diff.max().item():<18.4f} {'YES' if top_changed else 'no'}")

print()

# ── Step 8: Save HTML dashboard ───────────────────────────────────────────
print("=" * 70)
print("STEP 8: Generating feature dashboard")
print("=" * 70)

output_dir = Path("sae_demo_output")
output_dir.mkdir(exist_ok=True)

for i, idx in enumerate(top_indices[:5]):
    idx_val = idx.item()
    dashboard_path = output_dir / f"feature_{idx_val}.html"
    viz.save_feature_dashboard(idx_val, val_acts, save_path=dashboard_path)

summary = generate_sae_summary(sae, sae_config, val_acts, save_path=output_dir / "sae_summary.json")

print(f"\n  Dashboards saved to {output_dir}/")
print()

# ── Done ───────────────────────────────────────────────────────────────────
print("=" * 70)
print("DEMO COMPLETE!")
print("=" * 70)
print(f"""
  What we just did:
  1. Created a {n_params:,}-parameter nanochat GPT model
  2. Collected {activations.shape[0]:,} activations from layer {layer_idx}
  3. Trained a {sae_config.d_sae}-feature TopK SAE ({pct_reduction:.1f}% loss reduction)
  4. Evaluated: {eval_metrics.explained_variance*100:.1f}% explained variance, {eval_metrics.dead_latent_fraction*100:.1f}% dead latents
  5. Discovered {(top_freqs > 0.01).sum().item()} features with >1% activation frequency
  6. Tracked {sae_config.k} active features per token position at runtime
  7. Demonstrated feature steering changes model output
  8. Generated HTML dashboards for top 5 features

  Open {output_dir}/feature_*.html in a browser to explore!
""")
