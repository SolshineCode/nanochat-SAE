"""
Comprehensive tests for SAE trainer functionality.

The trainer handles complex logic including dead latent resampling,
learning rate scheduling, and checkpointing.

Run with: python -m pytest tests/test_sae_trainer.py -v
"""

import torch
import pytest
import sys
import tempfile
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sae.config import SAEConfig
from sae.models import TopKSAE, ReLUSAE
from sae.trainer import SAETrainer, train_sae_from_activations


def test_trainer_initialization():
    """Test SAE trainer initializes correctly."""
    config = SAEConfig(
        d_in=64,
        d_sae=256,
        activation="topk",
        k=16,
        batch_size=32,
    )

    sae = TopKSAE(config)
    activations = torch.randn(1000, config.d_in)
    val_activations = torch.randn(200, config.d_in)

    trainer = SAETrainer(
        sae=sae,
        config=config,
        activations=activations,
        val_activations=val_activations,
        device="cpu",
    )

    # Check initialization
    assert trainer.sae is not None
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None
    assert trainer.step == 0
    assert trainer.epoch == 0
    assert trainer.best_val_loss == float('inf')
    assert trainer.feature_counts.shape == (config.d_sae,)
    assert (trainer.feature_counts == 0).all()

    print("✓ Trainer initialization test passed")


def test_trainer_single_epoch():
    """Test training for one epoch."""
    config = SAEConfig(
        d_in=32,
        d_sae=128,
        activation="topk",
        k=8,
        batch_size=16,
        num_epochs=1,
    )

    sae = TopKSAE(config)
    activations = torch.randn(500, config.d_in)

    trainer = SAETrainer(
        sae=sae,
        config=config,
        activations=activations,
        device="cpu",
    )

    # Train one epoch
    metrics = trainer.train_epoch(verbose=False)

    # Check metrics returned
    assert "mse_loss" in metrics
    assert "l0" in metrics
    assert "total_loss" in metrics
    assert metrics["mse_loss"] >= 0
    assert metrics["l0"] > 0

    # Check epoch incremented
    assert trainer.epoch == 1
    assert trainer.step > 0

    # Check feature counts updated
    assert trainer.feature_counts.sum() > 0

    print("✓ Single epoch training test passed")


def test_trainer_loss_decreases():
    """Test training loss decreases over multiple epochs."""
    config = SAEConfig(
        d_in=32,
        d_sae=128,
        activation="topk",
        k=8,
        batch_size=16,
        num_epochs=5,
    )

    sae = TopKSAE(config)
    activations = torch.randn(500, config.d_in)

    trainer = SAETrainer(
        sae=sae,
        config=config,
        activations=activations,
        device="cpu",
    )

    # Train first epoch
    metrics_epoch1 = trainer.train_epoch(verbose=False)
    loss_epoch1 = metrics_epoch1["total_loss"]

    # Train few more epochs
    for _ in range(4):
        metrics = trainer.train_epoch(verbose=False)

    loss_final = metrics["total_loss"]

    # Loss should decrease
    assert loss_final < loss_epoch1, f"Loss should decrease: {loss_epoch1:.4f} -> {loss_final:.4f}"

    print("✓ Loss decrease test passed")


def test_trainer_validation():
    """Test validation evaluation works."""
    config = SAEConfig(
        d_in=32,
        d_sae=128,
        activation="topk",
        k=8,
        batch_size=16,
    )

    sae = TopKSAE(config)
    activations = torch.randn(500, config.d_in)
    val_activations = torch.randn(100, config.d_in)

    trainer = SAETrainer(
        sae=sae,
        config=config,
        activations=activations,
        val_activations=val_activations,
        device="cpu",
    )

    # Train one epoch
    trainer.train_epoch(verbose=False)

    # Evaluate
    val_metrics = trainer.evaluate()

    assert "mse_loss" in val_metrics
    assert "l0" in val_metrics
    assert "total_loss" in val_metrics
    assert val_metrics["mse_loss"] >= 0

    print("✓ Validation test passed")


def test_trainer_no_validation():
    """Test trainer works without validation set."""
    config = SAEConfig(
        d_in=32,
        d_sae=128,
        activation="topk",
        k=8,
        batch_size=16,
    )

    sae = TopKSAE(config)
    activations = torch.randn(500, config.d_in)

    trainer = SAETrainer(
        sae=sae,
        config=config,
        activations=activations,
        val_activations=None,  # No validation
        device="cpu",
    )

    # Evaluate should return empty dict
    val_metrics = trainer.evaluate()
    assert val_metrics == {}

    # Training should still work
    metrics = trainer.train_epoch(verbose=False)
    assert "total_loss" in metrics

    print("✓ No validation test passed")


def test_trainer_dead_latent_detection():
    """Test dead latents are detected correctly."""
    config = SAEConfig(
        d_in=32,
        d_sae=128,
        activation="topk",
        k=8,
        batch_size=16,
        dead_latent_threshold=0.001,  # 0.1% threshold
    )

    sae = TopKSAE(config)
    activations = torch.randn(500, config.d_in)

    trainer = SAETrainer(
        sae=sae,
        config=config,
        activations=activations,
        device="cpu",
    )

    # Manually set some features to never activate
    trainer.feature_counts[:10] = 0  # First 10 features are dead
    trainer.feature_counts[10:] = 100  # Rest are active
    trainer.step = 1000

    # Calculate activation frequency
    total_samples = trainer.step * config.batch_size
    activation_freq = trainer.feature_counts / total_samples

    # Find dead latents
    dead_mask = activation_freq < config.dead_latent_threshold
    num_dead = dead_mask.sum().item()

    # First 10 should be dead
    assert num_dead == 10
    assert dead_mask[:10].all()
    assert not dead_mask[10:].any()

    print("✓ Dead latent detection test passed")


def test_trainer_dead_latent_resampling():
    """Test dead latents are resampled correctly."""
    config = SAEConfig(
        d_in=32,
        d_sae=128,
        activation="topk",
        k=8,
        batch_size=16,
        dead_latent_threshold=0.001,
        resample_interval=100,
    )

    sae = TopKSAE(config)
    activations = torch.randn(500, config.d_in)

    trainer = SAETrainer(
        sae=sae,
        config=config,
        activations=activations,
        device="cpu",
    )

    # Manually set some features to never activate
    trainer.feature_counts[:10] = 0  # First 10 features are dead
    trainer.feature_counts[10:] = 1000  # Rest are active
    trainer.step = 10000  # Many steps so threshold is triggered

    # Save original weights
    original_enc_weights = sae.W_enc[:, :10].clone()
    original_dec_weights = sae.W_dec[:10].clone()

    # Trigger resampling
    trainer._resample_dead_latents()

    # Check weights were reinitialized for dead latents
    # Weights should be different after resampling
    assert not torch.allclose(sae.W_enc[:, :10], original_enc_weights)
    assert not torch.allclose(sae.W_dec[:10], original_dec_weights)

    # Check feature counts were reset for dead latents
    assert (trainer.feature_counts[:10] == 0).all()

    # Active features should be unchanged
    assert (trainer.feature_counts[10:] == 1000).all()

    print("✓ Dead latent resampling test passed")


def test_trainer_learning_rate_warmup():
    """Test learning rate warmup works."""
    config = SAEConfig(
        d_in=32,
        d_sae=128,
        activation="topk",
        k=8,
        batch_size=16,
        learning_rate=0.001,
        warmup_steps=100,
    )

    sae = TopKSAE(config)
    activations = torch.randn(1000, config.d_in)

    trainer = SAETrainer(
        sae=sae,
        config=config,
        activations=activations,
        device="cpu",
    )

    # Get initial learning rate (should be scaled down)
    initial_lr = trainer.optimizer.param_groups[0]['lr']
    assert initial_lr < config.learning_rate

    # Train for some steps
    for _ in range(50):
        batch = next(iter(trainer.train_loader))
        x = batch[0]
        reconstruction, features, metrics = sae(x)
        loss = metrics["total_loss"]
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        trainer.scheduler.step()
        trainer.step += 1

    # Learning rate should have increased
    mid_lr = trainer.optimizer.param_groups[0]['lr']
    assert mid_lr > initial_lr

    # Train until after warmup
    for _ in range(60):
        batch = next(iter(trainer.train_loader))
        x = batch[0]
        reconstruction, features, metrics = sae(x)
        loss = metrics["total_loss"]
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        trainer.scheduler.step()
        trainer.step += 1

    # Should reach full learning rate after warmup
    final_lr = trainer.optimizer.param_groups[0]['lr']
    assert abs(final_lr - config.learning_rate) < 1e-6

    print("✓ Learning rate warmup test passed")


def test_trainer_checkpoint_save_load():
    """Test saving and loading checkpoints."""
    config = SAEConfig(
        d_in=32,
        d_sae=128,
        activation="topk",
        k=8,
        batch_size=16,
    )

    sae = TopKSAE(config)
    activations = torch.randn(500, config.d_in)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)

        trainer = SAETrainer(
            sae=sae,
            config=config,
            activations=activations,
            device="cpu",
            save_dir=save_dir,
        )

        # Train for a bit
        trainer.train_epoch(verbose=False)
        trainer.train_epoch(verbose=False)

        # Save checkpoint
        trainer._save_checkpoint()

        # Check checkpoint exists
        checkpoint_path = save_dir / f"checkpoint_step{trainer.step}.pt"
        assert checkpoint_path.exists()

        # Check config JSON exists
        config_path = save_dir / "config.json"
        assert config_path.exists()

        # Create new trainer and load checkpoint
        sae2 = TopKSAE(config)
        trainer2 = SAETrainer(
            sae=sae2,
            config=config,
            activations=activations,
            device="cpu",
            save_dir=save_dir,
        )

        trainer2.load_checkpoint(checkpoint_path)

        # Check state was loaded
        assert trainer2.step == trainer.step
        assert trainer2.epoch == trainer.epoch
        assert torch.allclose(trainer2.feature_counts, trainer.feature_counts)

        # Check model weights match
        for p1, p2 in zip(trainer.sae.parameters(), trainer2.sae.parameters()):
            assert torch.allclose(p1, p2)

    print("✓ Checkpoint save/load test passed")


def test_trainer_best_model_tracking():
    """Test best model is saved when validation improves."""
    config = SAEConfig(
        d_in=32,
        d_sae=128,
        activation="topk",
        k=8,
        batch_size=16,
        eval_every=50,
    )

    sae = TopKSAE(config)
    activations = torch.randn(500, config.d_in)
    val_activations = torch.randn(100, config.d_in)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)

        trainer = SAETrainer(
            sae=sae,
            config=config,
            activations=activations,
            val_activations=val_activations,
            device="cpu",
            save_dir=save_dir,
        )

        # Train for several epochs
        trainer.train(num_epochs=3, verbose=False)

        # Best model should have been saved
        best_path = save_dir / "best_model.pt"
        assert best_path.exists()

        # Best val loss should have been updated
        assert trainer.best_val_loss < float('inf')

    print("✓ Best model tracking test passed")


def test_trainer_normalize_decoder():
    """Test decoder normalization is applied when configured."""
    config = SAEConfig(
        d_in=32,
        d_sae=128,
        activation="topk",
        k=8,
        batch_size=16,
        normalize_decoder=True,
    )

    sae = TopKSAE(config)
    activations = torch.randn(500, config.d_in)

    trainer = SAETrainer(
        sae=sae,
        config=config,
        activations=activations,
        device="cpu",
    )

    # Train one epoch
    trainer.train_epoch(verbose=False)

    # Check decoder weights are normalized
    decoder_norms = torch.norm(sae.W_dec, dim=1)
    assert torch.allclose(decoder_norms, torch.ones_like(decoder_norms), atol=1e-5)

    print("✓ Decoder normalization test passed")


def test_train_sae_from_activations():
    """Test convenience function for training from activations."""
    config = SAEConfig(
        d_in=32,
        d_sae=128,
        activation="topk",
        k=8,
        batch_size=16,
        num_epochs=2,
    )

    activations = torch.randn(500, config.d_in)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)

        sae, trainer = train_sae_from_activations(
            activations=activations,
            config=config,
            val_split=0.1,
            device="cpu",
            save_dir=save_dir,
            verbose=False,
        )

        # Check training completed
        assert trainer.epoch == 2
        assert trainer.step > 0

        # Check files were saved
        assert (save_dir / "config.json").exists()

    print("✓ Train from activations test passed")


def test_trainer_feature_count_accumulation():
    """Test feature counts accumulate correctly during training."""
    config = SAEConfig(
        d_in=32,
        d_sae=128,
        activation="topk",
        k=8,
        batch_size=16,
    )

    sae = TopKSAE(config)
    activations = torch.randn(500, config.d_in)

    trainer = SAETrainer(
        sae=sae,
        config=config,
        activations=activations,
        device="cpu",
    )

    # Initial counts should be zero
    assert (trainer.feature_counts == 0).all()

    # Train one batch manually
    batch = next(iter(trainer.train_loader))
    x = batch[0]

    with torch.no_grad():
        reconstruction, features, metrics = sae(x)
        active_features = (features != 0).float().sum(dim=0)
        trainer.feature_counts += active_features

    # Counts should have increased
    assert trainer.feature_counts.sum() > 0

    # Average L0 should be close to k
    avg_l0 = (features != 0).sum(dim=1).float().mean().item()
    assert abs(avg_l0 - config.k) < 2.0

    print("✓ Feature count accumulation test passed")


def run_all_tests():
    """Run all SAE trainer tests."""
    print("\n" + "="*80)
    print("Running SAE Trainer Tests")
    print("="*80 + "\n")

    try:
        test_trainer_initialization()
        test_trainer_single_epoch()
        test_trainer_loss_decreases()
        test_trainer_validation()
        test_trainer_no_validation()
        test_trainer_dead_latent_detection()
        test_trainer_dead_latent_resampling()
        test_trainer_learning_rate_warmup()
        test_trainer_checkpoint_save_load()
        test_trainer_best_model_tracking()
        test_trainer_normalize_decoder()
        test_train_sae_from_activations()
        test_trainer_feature_count_accumulation()

        print("\n" + "="*80)
        print("All SAE trainer tests passed! ✓")
        print("="*80 + "\n")

        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
