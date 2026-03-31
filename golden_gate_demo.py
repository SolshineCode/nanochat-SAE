"""
Build Your Own "Golden Gate Claude" with nanochat-SAE!
======================================================

A fun, educational walkthrough that recreates Anthropic's famous "Golden Gate
Claude" experiment at miniature scale using nanochat and Sparse Autoencoders.

WHAT WAS GOLDEN GATE CLAUDE?
-----------------------------
In May 2024, Anthropic discovered a single SAE feature inside Claude that
responded to the concept of the Golden Gate Bridge. When they artificially
amplified that feature during inference, Claude became *obsessed* with the
bridge -- weaving it into every response, comparing everything to its
suspension cables, and even claiming to *be* the Golden Gate Bridge.

It was a vivid demonstration that:
  1. SAEs can find human-interpretable concepts inside neural networks
  2. Amplifying a single feature can dramatically reshape model behavior
  3. We can steer AI systems at a level more precise than prompting

THIS WALKTHROUGH
-----------------
We'll recreate the same technique on a tiny nanochat model:
  Step 1: Build a small GPT model (the "brain")
  Step 2: Collect its internal activations (the "thoughts")
  Step 3: Train an SAE to decompose thoughts into features (the "concepts")
  Step 4: Find the most interesting features
  Step 5: STEER the model by cranking a feature up (make your own Golden Gate!)
  Step 6: Compare steered vs. unsteered outputs

No GPU required -- runs on CPU in under 2 minutes!

Usage:
    python golden_gate_demo.py
"""

import torch
import torch.nn.functional as F
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
from sae.feature_viz import FeatureVisualizer
from sae.runtime import InterpretableModel

torch.manual_seed(42)

# ─── Pretty printing helpers ─────────────────────────────────────────────────

BRIDGE = r"""
          _____
         /     \
    ____/       \____
   |  |    ( )    |  |
   |  |    | |    |  |
   |  |    | |    |  |
  _|  |____| |____|  |_
 / ________________________\
 \/\/\/\/\/\/\/\/\/\/\/\/\/\/
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def banner(title, subtitle=""):
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * width)


def info(msg):
    print(f"  -> {msg}")


def explain(text):
    """Print an educational explanation in a box."""
    lines = text.strip().split("\n")
    width = max(len(line) for line in lines) + 4
    print()
    print(f"  +-{'-' * width}-+")
    for line in lines:
        print(f"  |  {line:<{width}} |")
    print(f"  +-{'-' * width}-+")
    print()


# ─── STEP 1: Build the Brain ─────────────────────────────────────────────────

banner(
    "STEP 1: Build the Brain",
    "Creating a tiny GPT model -- our miniature Claude"
)

explain("""\
Real Claude has billions of parameters. Our nanochat model is much
smaller, but it has the SAME transformer architecture: attention heads,
MLP layers, residual streams -- all the pieces that make language
models work. Think of it as a toy brain we can fully inspect.""")

config = GPTConfig(
    sequence_len=128,
    vocab_size=512,
    n_layer=8,       # 8 transformer layers (our "brain" has 8 levels of processing)
    n_head=4,        # 4 attention heads per layer
    n_kv_head=4,
    n_embd=128,      # 128-dimensional hidden state (the "thought" vector)
)

model = GPT(config)
model.init_weights()
torch.nn.init.normal_(model.lm_head.weight, std=0.02)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
info(f"Model: {config.n_layer} layers, {config.n_head} heads, d={config.n_embd}")
info(f"Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")
info(f"Think of this as a miniature language model brain!")


# ─── STEP 2: Record the Brain's Thoughts ─────────────────────────────────────

banner(
    "STEP 2: Record the Brain's Thoughts",
    "Collecting activations from layer 4 (the middle of the brain)"
)

explain("""\
When a transformer processes text, each layer produces a "residual
stream" -- a vector that encodes everything the model has figured out
so far. We'll tap into layer 4 (the middle) and record thousands of
these thought-vectors.

This is like putting an EEG on our model's brain!""")

target_layer = 4
hook_point = f"blocks.{target_layer}.hook_resid_post"

collector = ActivationCollector(
    model=model,
    hook_points=[hook_point],
    max_activations=15_000,
    device="cpu",
)

t0 = time.time()
with torch.no_grad(), collector:
    for i in range(30):
        # Feed random "text" through the model
        tokens = torch.randint(0, config.vocab_size, (8, config.sequence_len))
        model(tokens)
        if (i + 1) % 10 == 0:
            info(f"Batch {i+1}/30 -- {collector.counts[hook_point]:,} thought-vectors recorded")

activations = collector.get_activations()[hook_point]
elapsed = time.time() - t0

info(f"Collected {activations.shape[0]:,} thought-vectors of dimension {activations.shape[1]}")
info(f"Time: {elapsed:.1f}s")


# ─── STEP 3: Decompose Thoughts into Concepts ────────────────────────────────

banner(
    "STEP 3: Decompose Thoughts into Concepts",
    "Training a Sparse Autoencoder to find interpretable features"
)

explain("""\
Here's the key insight behind Golden Gate Claude:

A 128-dimensional thought-vector is DENSE -- every dimension is active.
It's like hearing 128 instruments playing at once. Hard to understand!

A Sparse Autoencoder (SAE) learns to DECOMPOSE each thought into a
combination of just a FEW "concept features" from a much larger
dictionary. It's like having 1,024 possible instruments but only 16
playing at any moment.

Each feature in the SAE dictionary might correspond to a human-readable
concept: "bridge", "math", "negation", "happy", etc.

Golden Gate Claude worked because Anthropic found a single feature
that meant "Golden Gate Bridge" and turned its volume WAY up!""")

sae_config = SAEConfig(
    d_in=config.n_embd,         # 128-dim input (matches model hidden size)
    expansion_factor=8,          # 1,024 features in our dictionary
    activation="topk",           # Keep only top-k features active (sparsity!)
    k=16,                        # Only 16 "concepts" active per thought
    hook_point=hook_point,
    batch_size=256,
    num_epochs=8,
    learning_rate=3e-4,
)

info(f"SAE: {sae_config.d_in} -> {sae_config.d_sae} features (8x expansion)")
info(f"Sparsity: only {sae_config.k} of {sae_config.d_sae} features active at once")
info(f"This is like having a dictionary of {sae_config.d_sae} concepts,")
info(f"but each thought only uses {sae_config.k} of them!")
print()

n_val = 1500
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
    bar = "#" * int(30 * (epoch + 1) / sae_config.num_epochs)
    print(f"  [{bar:<30}] Epoch {epoch+1}/{sae_config.num_epochs}  loss: {metrics['total_loss']:.6f}  L0: {metrics['l0']:.1f}")

elapsed = time.time() - t0
pct = (1 - losses[-1] / losses[0]) * 100
info(f"Training complete in {elapsed:.1f}s -- loss reduced by {pct:.1f}%")


# ─── STEP 4: Evaluate & Find the Best Features ──────────────────────────────

banner(
    "STEP 4: Evaluate & Find Our 'Golden Gate' Features",
    "Which concepts did the SAE discover?"
)

explain("""\
Now we evaluate: how well does the SAE reconstruct the original
thoughts? And which features are most active?

In the real Golden Gate Claude experiment, Anthropic trained SAEs on
Claude's residual stream and found features for thousands of concepts.
One happened to strongly activate on Golden Gate Bridge text.

Our model is randomly initialized (not trained on real text), so our
features won't have obvious semantic meanings. But the MECHANISM is
identical! Each feature is a direction in thought-space that the SAE
has learned to isolate.""")

evaluator = SAEEvaluator(sae, sae_config)
eval_metrics = evaluator.evaluate(val_acts, compute_dead_latents=True)

info(f"Explained Variance:  {eval_metrics.explained_variance * 100:.1f}%")
info(f"Reconstruction MSE:  {eval_metrics.mse_loss:.6f}")
info(f"Alive features:      {int((1 - eval_metrics.dead_latent_fraction) * sae_config.d_sae)}/{sae_config.d_sae}")
info(f"L0 (active/thought): {eval_metrics.l0_mean:.1f}")

viz = FeatureVisualizer(sae, sae_config)
top_indices, top_freqs = viz.get_top_features(val_acts, k=10)

print()
info("Top 10 most active features (our 'concept dictionary'):")
print(f"  {'Rank':<6} {'Feature':<10} {'Freq':<10} {'Strength':<15} {'Max':<10}")
print(f"  {'-' * 55}")

for i, (idx, freq) in enumerate(zip(top_indices[:10], top_freqs[:10])):
    stats = viz.get_feature_statistics(idx.item(), val_acts)
    label = ""
    if i == 0:
        label = "  <-- Our 'Golden Gate' feature!"
    print(f"  {i+1:<6} #{idx.item():<9} {freq:.4f}     {stats['mean_when_active']:<15.4f} {stats['max_activation']:<10.4f}{label}")


# ─── STEP 5: THE GOLDEN GATE MOMENT -- Feature Steering! ────────────────────

banner(
    "STEP 5: THE GOLDEN GATE MOMENT!",
    "Cranking up a single feature to reshape the brain's output"
)

print(BRIDGE)

explain("""\
THIS is what made Golden Gate Claude famous!

When Anthropic found the "Golden Gate Bridge" feature, they didn't just
observe it -- they AMPLIFIED it during inference. They multiplied its
activation by a large number, effectively telling the model:

  "Whatever you're thinking about, think about the Golden Gate Bridge
   MORE. A LOT more."

The result? Claude started relating EVERYTHING to the bridge. Ask about
math? "Well, the Golden Gate Bridge has some lovely mathematical
curves..." Ask about feelings? "I feel like the Golden Gate Bridge
on a foggy morning..."

We'll do the same thing. We pick our most active feature (our stand-in
for "Golden Gate") and crank it up to various strengths.""")

golden_feature = top_indices[0].item()
info(f"Selected feature #{golden_feature} as our 'Golden Gate Bridge' feature")
print()

saes = {hook_point: sae}
interp_model = InterpretableModel(model, saes, device="cpu")

# Create a fixed input to compare steered vs unsteered
test_tokens = torch.randint(0, config.vocab_size, (1, 32))

with torch.no_grad():
    baseline_logits = interp_model(test_tokens)

baseline_probs = F.softmax(baseline_logits[:, -1, :], dim=-1)
baseline_top5 = torch.topk(baseline_probs, 5)

info("BASELINE (no steering) -- next token prediction:")
for i in range(5):
    tok = baseline_top5.indices[0, i].item()
    prob = baseline_top5.values[0, i].item()
    print(f"    Token {tok:>4}: {prob:.4f} {'|' * int(prob * 100)}")

print()
print("  Now let's crank up our 'Golden Gate' feature...")
print()

strengths = [1.0, 2.0, 5.0, 10.0, 50.0]
print(f"  {'Strength':<12} {'Top Token':<12} {'Prob':<10} {'Logit Diff (mean)':<20} {'Logit Diff (max)'}")
print(f"  {'-' * 70}")

for strength in strengths:
    with torch.no_grad():
        steered_logits = interp_model.steer(
            test_tokens,
            feature_id=(hook_point, golden_feature),
            strength=strength,
        )

    steered_probs = F.softmax(steered_logits[:, -1, :], dim=-1)
    diff = (steered_logits - baseline_logits).abs()
    top_tok = steered_probs.argmax().item()
    top_prob = steered_probs.max().item()

    marker = ""
    if strength >= 10.0:
        marker = " <-- OBSESSED!"
    elif strength >= 5.0:
        marker = " <-- strong shift"

    print(f"  {strength:<12.1f} Token {top_tok:<6} {top_prob:<10.4f} {diff.mean().item():<20.4f} {diff.max().item():.4f}{marker}")


# ─── STEP 6: Deep Dive -- Watching Features in Real Time ─────────────────────

banner(
    "STEP 6: Watching Features Fire in Real Time",
    "Like a brain scan for our model"
)

explain("""\
Let's watch what happens inside the model during inference. We'll track
which features activate at each token position -- like watching neurons
fire in real time.

Then we'll compare: what does the feature landscape look like when
we're steering vs. not steering?""")

# Unsteered feature landscape
with interp_model.interpretation_enabled():
    with torch.no_grad():
        interp_model(test_tokens)
    baseline_features = interp_model.get_active_features()[hook_point].clone()

info("UNSTEERED feature landscape (first 5 positions):")
for pos in range(min(5, baseline_features.shape[0])):
    active = torch.nonzero(baseline_features[pos]).squeeze(-1)
    vals = baseline_features[pos, active]
    top3 = vals.argsort(descending=True)[:3]
    feat_strs = [f"#{active[j].item()}({vals[j]:.2f})" for j in top3]
    is_golden = any(active[j].item() == golden_feature for j in top3)
    marker = " *GOLDEN*" if is_golden else ""
    print(f"    Pos {pos}: {', '.join(feat_strs)}{marker}")

print()

# Now steer and compare
info(f"STEERED (feature #{golden_feature} at 10x) -- same input:")

with torch.no_grad():
    # Steer and track features simultaneously using a manual hook
    steering_config = {hook_point: (golden_feature, 10.0)}
    steered_features_storage = {}
    with interp_model.steering_enabled(steering_config):
        # Attach a tracking hook alongside the steering hook
        from sae.hooks import get_module_from_hook_point
        module = get_module_from_hook_point(interp_model.model, hook_point)

        def capture_steered_features(mod, inp, output):
            activation = output[0] if isinstance(output, tuple) else output
            if activation.ndim == 3:
                activation = activation.reshape(-1, activation.shape[-1])
            steered_features_storage[hook_point] = sae.get_feature_activations(activation)

        handle = module.register_forward_hook(capture_steered_features)
        interp_model.model(test_tokens)
        handle.remove()

if hook_point in steered_features_storage:
    steered_feats = steered_features_storage[hook_point]
    for pos in range(min(5, steered_feats.shape[0])):
        active = torch.nonzero(steered_feats[pos]).squeeze(-1)
        vals = steered_feats[pos, active]
        top3 = vals.argsort(descending=True)[:3]
        feat_strs = [f"#{active[j].item()}({vals[j]:.2f})" for j in top3]
        is_golden = any(active[j].item() == golden_feature for j in top3)
        marker = " *GOLDEN*" if is_golden else ""
        print(f"    Pos {pos}: {', '.join(feat_strs)}{marker}")

explain("""\
Notice how steering changes the probability distribution over next
tokens. At high strengths, the model's output becomes dominated by
our chosen feature -- just like Golden Gate Claude became obsessed
with the bridge!

The key takeaway: features are DIRECTIONS in the model's internal
representation space. Amplifying a direction pushes all the model's
"thinking" in that direction.""")


# ─── STEP 7: Try Different Features ──────────────────────────────────────────

banner(
    "STEP 7: Feature Steering Playground",
    "What happens when we steer with different features?"
)

explain("""\
Golden Gate Claude used just ONE feature. But we have a whole
dictionary of {d_sae} features! Let's try steering with a few
different ones and see how each reshapes the output differently.

In a real (trained) model, each feature might correspond to:
  - A topic (bridges, animals, math...)
  - A style (formal, casual, poetic...)
  - A behavior (being helpful, refusing, hedging...)

This is why SAE-based steering is so powerful: you can precisely
control individual "concepts" without retraining the model!""".format(d_sae=sae_config.d_sae))

info("Steering with top 5 features at strength=10.0:")
print(f"  {'Feature':<12} {'Top Predicted Token':<22} {'Prob':<10} {'Output Changed?'}")
print(f"  {'-' * 55}")

for i, idx in enumerate(top_indices[:5]):
    feat_idx = idx.item()
    with torch.no_grad():
        steered = interp_model.steer(
            test_tokens,
            feature_id=(hook_point, feat_idx),
            strength=10.0,
        )
    steered_probs = F.softmax(steered[:, -1, :], dim=-1)
    top_tok = steered_probs.argmax().item()
    top_prob = steered_probs.max().item()
    changed = top_tok != baseline_probs.argmax().item()
    label = "YES -- different behavior!" if changed else "same"
    marker = "  <-- 'Golden Gate'" if i == 0 else ""
    print(f"  #{feat_idx:<10} Token {top_tok:<16} {top_prob:<10.4f} {label}{marker}")

print()

# Negative steering (suppression)
info("BONUS: Negative steering (suppression) -- the ANTI-Golden Gate!")
info(f"Suppressing feature #{golden_feature} (strength = -5.0):")

with torch.no_grad():
    suppressed = interp_model.steer(
        test_tokens,
        feature_id=(hook_point, golden_feature),
        strength=-5.0,
    )

suppressed_probs = F.softmax(suppressed[:, -1, :], dim=-1)
diff = (suppressed[:, -1, :] - baseline_logits[:, -1, :]).abs().mean().item()
info(f"Mean logit change from suppression: {diff:.4f}")
info("In a real model, this would be like making Claude AVOID thinking")
info("about the Golden Gate Bridge -- the opposite of the famous demo!")


# ─── STEP 8: Save Your Work ──────────────────────────────────────────────────

banner(
    "STEP 8: Save Your Work",
    "Generating feature dashboards you can explore in a browser"
)

output_dir = Path("golden_gate_output")
output_dir.mkdir(exist_ok=True)

for i, idx in enumerate(top_indices[:5]):
    dashboard_path = output_dir / f"feature_{idx.item()}.html"
    viz.save_feature_dashboard(idx.item(), val_acts, save_path=dashboard_path)

info(f"Saved 5 feature dashboards to {output_dir}/")
info("Open the HTML files in a browser to explore!")


# ─── FINALE ───────────────────────────────────────────────────────────────────

banner("CONGRATULATIONS!", "You just built your own Golden Gate Claude!")

print(BRIDGE)

print("""\
  WHAT YOU JUST DID:

  1. Built a miniature transformer (like a tiny Claude)
  2. Recorded {n_acts:,} internal "thought vectors" from layer {layer}
  3. Trained an SAE that decomposes thoughts into {d_sae} sparse features
  4. Found that only {alive} features are alive ({alive_pct:.0f}% of {d_sae})
  5. Picked feature #{golden} as your "Golden Gate" feature
  6. STEERED the model by amplifying that feature during inference
  7. Watched how different steering strengths change the output
  8. Tried steering with different features AND suppressing features

  THE BIG IDEAS:

  * SAEs decompose neural network activations into interpretable features
  * Each feature is a direction in the model's representation space
  * Amplifying a feature during inference reshapes model behavior
  * This is the EXACT technique behind Golden Gate Claude!
  * The same method works at any scale: from our toy model to Claude

  NEXT STEPS:

  * Train nanochat on real data (bash speedrun.sh) and repeat this
    with a model that has REAL semantic features
  * Try the Colab notebook: colab_sae_training.ipynb
  * Train SAEs on multiple layers and compare what they learn
  * Read Anthropic's "Scaling Monosemanticity" paper for the full story
  * Share your discoveries on Neuronpedia!

  Happy feature hunting!
""".format(
    n_acts=activations.shape[0],
    layer=target_layer,
    d_sae=sae_config.d_sae,
    alive=int((1 - eval_metrics.dead_latent_fraction) * sae_config.d_sae),
    alive_pct=(1 - eval_metrics.dead_latent_fraction) * 100,
    golden=golden_feature,
))
