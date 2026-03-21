"""
End-to-end integration test for the full SAE pipeline on a tiny nanochat model.

Tests the complete workflow:
1. Create a tiny nanochat GPT model (randomly initialized)
2. Collect activations from a layer using hooks
3. Train an SAE on those activations
4. Evaluate the SAE (reconstruction quality, sparsity, dead latents)
5. Run feature visualization
6. Test runtime interpretation (feature tracking)
7. Test feature steering

Run with: python -m pytest tests/test_e2e_sae_pipeline.py -v -s
"""

import torch
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt import GPT, GPTConfig
from sae.config import SAEConfig
from sae.models import TopKSAE, create_sae
from sae.hooks import ActivationCollector
from sae.trainer import SAETrainer
from sae.evaluator import SAEEvaluator
from sae.feature_viz import FeatureVisualizer, generate_sae_summary
from sae.runtime import InterpretableModel, save_sae, load_saes


# Tiny model config for fast CPU testing
TINY_CONFIG = GPTConfig(
    sequence_len=64,
    vocab_size=256,
    n_layer=4,
    n_head=4,
    n_kv_head=4,
    n_embd=64,
)


@pytest.fixture
def tiny_model():
    """Create a tiny nanochat GPT model for testing.

    Note: init_weights() zeros lm_head and c_proj, which is correct for
    training but means logits are always 0. We re-randomize lm_head so
    that steering tests can observe output differences.
    """
    model = GPT(TINY_CONFIG)
    model.init_weights()
    # Re-randomize lm_head so logits are non-trivial (init_weights zeros it)
    torch.nn.init.normal_(model.lm_head.weight, std=0.02)
    model.eval()
    return model


@pytest.fixture
def collected_activations(tiny_model):
    """Collect activations from layer 2 of the tiny model."""
    hook_point = "blocks.2.hook_resid_post"
    collector = ActivationCollector(
        model=tiny_model,
        hook_points=[hook_point],
        max_activations=2000,
        device="cpu",
    )

    with torch.no_grad(), collector:
        for _ in range(10):
            tokens = torch.randint(0, TINY_CONFIG.vocab_size, (4, TINY_CONFIG.sequence_len))
            tiny_model(tokens)

    activations = collector.get_activations()[hook_point]
    return activations, hook_point


@pytest.fixture
def trained_sae(collected_activations):
    """Train an SAE on the collected activations."""
    activations, hook_point = collected_activations

    config = SAEConfig(
        d_in=TINY_CONFIG.n_embd,
        expansion_factor=4,
        activation="topk",
        k=8,
        hook_point=hook_point,
        batch_size=128,
        num_epochs=3,
        learning_rate=1e-3,
    )

    sae = TopKSAE(config)

    # Split train/val
    n_val = 200
    train_acts = activations[n_val:]
    val_acts = activations[:n_val]

    trainer = SAETrainer(
        sae=sae,
        config=config,
        activations=train_acts,
        val_activations=val_acts,
        device="cpu",
    )

    # Train
    initial_loss = None
    for epoch in range(3):
        metrics = trainer.train_epoch(verbose=False)
        if initial_loss is None:
            initial_loss = metrics["total_loss"]

    return sae, config, trainer, initial_loss, metrics["total_loss"]


class TestE2EPipeline:
    """End-to-end SAE pipeline tests using a real (tiny) nanochat model."""

    def test_model_creation(self, tiny_model):
        """Test that we can create and run the tiny nanochat model."""
        tokens = torch.randint(0, TINY_CONFIG.vocab_size, (2, TINY_CONFIG.sequence_len))
        with torch.no_grad():
            logits = tiny_model(tokens)
        assert logits.shape == (2, TINY_CONFIG.sequence_len, TINY_CONFIG.vocab_size)
        print(f"  Model params: {sum(p.numel() for p in tiny_model.parameters())/1e3:.1f}K")

    def test_activation_collection(self, collected_activations):
        """Test that we can collect activations from the model."""
        activations, hook_point = collected_activations
        assert activations.shape[0] >= 2000
        assert activations.shape[1] == TINY_CONFIG.n_embd
        print(f"  Collected {activations.shape[0]} activations of dim {activations.shape[1]}")

    def test_sae_training_loss_decreases(self, trained_sae):
        """Test that SAE training reduces reconstruction loss."""
        sae, config, trainer, initial_loss, final_loss = trained_sae
        assert final_loss < initial_loss, f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        print(f"  Loss: {initial_loss:.4f} -> {final_loss:.4f} ({(1 - final_loss/initial_loss)*100:.1f}% reduction)")

    def test_sae_evaluation(self, trained_sae, collected_activations):
        """Test SAE evaluation metrics."""
        sae, config, *_ = trained_sae
        activations, _ = collected_activations

        evaluator = SAEEvaluator(sae, config)
        metrics = evaluator.evaluate(activations[:500], compute_dead_latents=True)

        assert metrics.mse_loss >= 0
        assert 0 <= metrics.explained_variance <= 1
        assert metrics.l0_mean > 0
        assert 0 <= metrics.dead_latent_fraction <= 1

        print(f"  MSE: {metrics.mse_loss:.4f}")
        print(f"  Explained Variance: {metrics.explained_variance:.4f}")
        print(f"  L0: {metrics.l0_mean:.1f}")
        print(f"  Dead Latents: {metrics.dead_latent_fraction*100:.1f}%")

    def test_feature_visualization(self, trained_sae, collected_activations):
        """Test feature visualization pipeline."""
        sae, config, *_ = trained_sae
        activations, _ = collected_activations

        viz = FeatureVisualizer(sae, config)

        # Get top features
        top_indices, top_freqs = viz.get_top_features(activations[:500], k=10)
        assert len(top_indices) == 10
        assert (top_freqs >= 0).all()
        assert (top_freqs <= 1).all()

        # Get feature statistics
        feature_idx = top_indices[0].item()
        stats = viz.get_feature_statistics(feature_idx, activations[:500])
        assert "activation_frequency" in stats
        assert "max_activation" in stats

        print(f"  Top feature {feature_idx}: freq={top_freqs[0]:.4f}")

    def test_feature_dashboard_html(self, trained_sae, collected_activations):
        """Test HTML dashboard generation."""
        sae, config, *_ = trained_sae
        activations, _ = collected_activations

        viz = FeatureVisualizer(sae, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            dashboard_path = Path(tmpdir) / "feature_0.html"
            viz.save_feature_dashboard(0, activations[:500], save_path=dashboard_path)
            assert dashboard_path.exists()
            html = dashboard_path.read_text()
            assert "Feature 0" in html
            assert "Statistics" in html
            print(f"  Dashboard generated: {len(html)} chars")

    def test_sae_summary(self, trained_sae, collected_activations):
        """Test SAE summary generation."""
        sae, config, *_ = trained_sae
        activations, _ = collected_activations

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            summary = generate_sae_summary(sae, config, activations[:500], save_path=summary_path)
            assert "top_features" in summary
            assert len(summary["top_features"]) > 0
            assert summary_path.exists()
            print(f"  Summary has {len(summary['top_features'])} top features")

    def test_save_and_load_sae(self, trained_sae):
        """Test SAE checkpoint save/load round-trip."""
        sae, config, *_ = trained_sae

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "layer_2" / "best_model.pt"
            save_sae(sae, config, save_path)

            # Load it back
            loaded_saes = load_saes(Path(tmpdir), device="cpu")
            assert config.hook_point in loaded_saes

            loaded_sae = loaded_saes[config.hook_point]

            # Verify weights match
            x = torch.randn(10, config.d_in)
            with torch.no_grad():
                orig_out, _, _ = sae(x)
                loaded_out, _, _ = loaded_sae(x)
            assert torch.allclose(orig_out, loaded_out, atol=1e-6)
            print("  Save/load round-trip: weights match")

    def test_runtime_interpretation(self, tiny_model, trained_sae):
        """Test runtime feature tracking during inference."""
        sae, config, *_ = trained_sae

        # Wrap model with interpretability
        saes = {config.hook_point: sae}
        interp_model = InterpretableModel(tiny_model, saes, device="cpu")

        tokens = torch.randint(0, TINY_CONFIG.vocab_size, (1, TINY_CONFIG.sequence_len))

        with interp_model.interpretation_enabled():
            with torch.no_grad():
                output = interp_model(tokens)
            features = interp_model.get_active_features()

        assert config.hook_point in features
        feature_tensor = features[config.hook_point]
        assert feature_tensor.shape[1] == config.d_sae

        num_active = (feature_tensor != 0).sum(dim=-1).float().mean().item()
        print(f"  Active features per position: {num_active:.1f} (expected ~{config.k})")

    def test_feature_steering(self, tiny_model, trained_sae, collected_activations):
        """Test feature steering modifies model output."""
        sae, config, *_ = trained_sae
        activations, _ = collected_activations

        # Find an actually-active feature (TopK is sparse, feature 0 may never fire)
        viz = FeatureVisualizer(sae, config)
        top_indices, top_freqs = viz.get_top_features(activations[:500], k=1)
        active_feature = top_indices[0].item()

        saes = {config.hook_point: sae}
        interp_model = InterpretableModel(tiny_model, saes, device="cpu")

        tokens = torch.randint(0, TINY_CONFIG.vocab_size, (1, TINY_CONFIG.sequence_len))

        # Run without steering
        with torch.no_grad():
            baseline_output = interp_model(tokens)

        # Run with steering (amplify the most-active feature)
        with torch.no_grad():
            steered_output = interp_model.steer(
                tokens,
                feature_id=(config.hook_point, active_feature),
                strength=5.0,
            )

        # Outputs should differ
        diff = (steered_output - baseline_output).abs().mean().item()
        assert diff > 0, f"Steering feature {active_feature} should change model output"
        print(f"  Steering feature {active_feature} diff: {diff:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
