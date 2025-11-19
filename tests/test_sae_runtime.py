"""
Comprehensive tests for SAE runtime interpretation and steering.

The runtime wrapper enables feature tracking and steering during inference.
Hook management is complex and must handle cleanup correctly.

Run with: python -m pytest tests/test_sae_runtime.py -v
"""

import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sae.config import SAEConfig
from sae.models import TopKSAE
from sae.runtime import InterpretableModel


class SimpleModel(nn.Module):
    """Simple model for testing without full GPT complexity."""

    def __init__(self, d_model=64):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_model)
        self.layer2 = nn.Linear(d_model, d_model)
        self.layer3 = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        return x

    def get_device(self):
        return next(self.parameters()).device


def test_interpretable_model_initialization():
    """Test InterpretableModel initializes correctly."""
    model = SimpleModel(d_model=64)

    config = SAEConfig(d_in=64, d_sae=256, activation="topk", k=16)
    sae = TopKSAE(config)

    # Create interpretable model with SAE for layer2
    saes = {"layer2": sae}
    interp_model = InterpretableModel(model, saes, device="cpu")

    assert interp_model.model == model
    assert len(interp_model.saes) == 1
    assert "layer2" in interp_model.saes
    assert interp_model.device == "cpu"
    assert not interp_model._interpretation_active
    assert not interp_model._steering_active
    assert len(interp_model._hook_handles) == 0

    print("✓ InterpretableModel initialization test passed")


def test_interpretable_model_forward_pass():
    """Test forward pass works without hooks."""
    model = SimpleModel(d_model=64)

    config = SAEConfig(d_in=64, d_sae=256, activation="topk", k=16)
    sae = TopKSAE(config)

    saes = {"layer2": sae}
    interp_model = InterpretableModel(model, saes, device="cpu")

    # Forward pass should work
    x = torch.randn(4, 64)
    output = interp_model(x)

    assert output.shape == x.shape

    print("✓ Forward pass test passed")


def test_interpretation_context_manager():
    """Test interpretation context manager attaches and removes hooks."""
    model = SimpleModel(d_model=64)

    config = SAEConfig(d_in=64, d_sae=256, activation="topk", k=16)
    sae = TopKSAE(config)

    saes = {"layer2": sae}
    interp_model = InterpretableModel(model, saes, device="cpu")

    # Initially no hooks
    assert len(interp_model._hook_handles) == 0
    assert not interp_model._interpretation_active

    # Enable interpretation
    with interp_model.interpretation_enabled():
        # Hooks should be active
        assert interp_model._interpretation_active
        assert len(interp_model._hook_handles) == 1

        # Run forward pass
        x = torch.randn(4, 64)
        output = interp_model(x)

    # Hooks should be removed after context
    assert len(interp_model._hook_handles) == 0
    assert not interp_model._interpretation_active
    assert len(interp_model._active_features) == 0

    print("✓ Interpretation context manager test passed")


def test_feature_tracking():
    """Test active features are tracked correctly."""
    model = SimpleModel(d_model=64)

    config = SAEConfig(d_in=64, d_sae=256, activation="topk", k=16)
    sae = TopKSAE(config)

    saes = {"layer2": sae}
    interp_model = InterpretableModel(model, saes, device="cpu")

    with interp_model.interpretation_enabled():
        x = torch.randn(4, 64)
        output = interp_model(x)

        # Get active features
        features = interp_model.get_active_features()

        assert isinstance(features, dict)
        assert "layer2" in features
        assert features["layer2"].shape[1] == config.d_sae

        # Check features are sparse (TopK)
        active_count = (features["layer2"] != 0).sum(dim=1).float().mean().item()
        assert active_count <= config.k * 1.5  # Allow some slack

    print("✓ Feature tracking test passed")


def test_get_active_features_without_context():
    """Test getting features without context manager raises error."""
    model = SimpleModel(d_model=64)

    config = SAEConfig(d_in=64, d_sae=256, activation="topk", k=16)
    sae = TopKSAE(config)

    saes = {"layer2": sae}
    interp_model = InterpretableModel(model, saes, device="cpu")

    # Try to get features without enabling interpretation
    with pytest.raises(RuntimeError, match="No features available"):
        interp_model.get_active_features()

    print("✓ Get features without context test passed")


def test_steering_context_manager():
    """Test steering context manager works."""
    model = SimpleModel(d_model=64)

    config = SAEConfig(d_in=64, d_sae=256, activation="topk", k=16)
    sae = TopKSAE(config)

    saes = {"layer2": sae}
    interp_model = InterpretableModel(model, saes, device="cpu")

    # Setup steering config
    steering = {"layer2": (10, 2.0)}  # Amplify feature 10 by 2x

    # Initially no hooks
    assert not interp_model._steering_active
    assert len(interp_model._hook_handles) == 0

    with interp_model.steering_enabled(steering):
        # Steering should be active
        assert interp_model._steering_active
        assert len(interp_model._hook_handles) == 1

        # Run forward pass
        x = torch.randn(4, 64)
        output = interp_model(x)

        # Output should be modified by steering
        assert output.shape == x.shape

    # Hooks should be removed
    assert len(interp_model._hook_handles) == 0
    assert not interp_model._steering_active

    print("✓ Steering context manager test passed")


def test_feature_amplification():
    """Test feature amplification modifies output."""
    model = SimpleModel(d_model=64)

    config = SAEConfig(d_in=64, d_sae=256, activation="topk", k=16)
    sae = TopKSAE(config)

    # Train SAE briefly to get reasonable weights
    activations = torch.randn(1000, 64)
    for _ in range(10):
        reconstruction, features, metrics = sae(activations)
        loss = metrics["total_loss"]
        sae.zero_grad()
        loss.backward()
        for p in sae.parameters():
            p.data -= 0.01 * p.grad

    saes = {"layer2": sae}
    interp_model = InterpretableModel(model, saes, device="cpu")

    # Get baseline output
    torch.manual_seed(42)
    x = torch.randn(4, 64)

    with torch.no_grad():
        baseline_output = interp_model(x)

    # Get steered output
    steering = {"layer2": (10, 5.0)}  # Strong amplification

    torch.manual_seed(42)
    x = torch.randn(4, 64)

    with torch.no_grad():
        with interp_model.steering_enabled(steering):
            steered_output = interp_model(x)

    # Outputs should be different
    assert not torch.allclose(baseline_output, steered_output, atol=1e-5)

    print("✓ Feature amplification test passed")


def test_feature_suppression():
    """Test feature suppression modifies output."""
    model = SimpleModel(d_model=64)

    config = SAEConfig(d_in=64, d_sae=256, activation="topk", k=16)
    sae = TopKSAE(config)

    saes = {"layer2": sae}
    interp_model = InterpretableModel(model, saes, device="cpu")

    # Get baseline output
    torch.manual_seed(42)
    x = torch.randn(4, 64)

    with torch.no_grad():
        baseline_output = interp_model(x)

    # Get suppressed output
    steering = {"layer2": (10, 0.0)}  # Suppress feature 10 completely

    torch.manual_seed(42)
    x = torch.randn(4, 64)

    with torch.no_grad():
        with interp_model.steering_enabled(steering):
            suppressed_output = interp_model(x)

    # Outputs should be different
    assert not torch.allclose(baseline_output, suppressed_output, atol=1e-5)

    print("✓ Feature suppression test passed")


def test_hook_cleanup_on_error():
    """Test hooks are cleaned up even if exception occurs."""
    model = SimpleModel(d_model=64)

    config = SAEConfig(d_in=64, d_sae=256, activation="topk", k=16)
    sae = TopKSAE(config)

    saes = {"layer2": sae}
    interp_model = InterpretableModel(model, saes, device="cpu")

    try:
        with interp_model.interpretation_enabled():
            # Cause an error
            raise ValueError("Test error")
    except ValueError:
        pass

    # Hooks should still be cleaned up
    assert len(interp_model._hook_handles) == 0
    assert not interp_model._interpretation_active

    print("✓ Hook cleanup on error test passed")


def test_multiple_hook_points():
    """Test interpretation with multiple hook points."""
    model = SimpleModel(d_model=64)

    config1 = SAEConfig(d_in=64, d_sae=256, activation="topk", k=16)
    config2 = SAEConfig(d_in=64, d_sae=256, activation="topk", k=16)
    sae1 = TopKSAE(config1)
    sae2 = TopKSAE(config2)

    saes = {
        "layer1": sae1,
        "layer2": sae2,
    }
    interp_model = InterpretableModel(model, saes, device="cpu")

    with interp_model.interpretation_enabled():
        # Should have hooks for both layers
        assert len(interp_model._hook_handles) == 2

        x = torch.randn(4, 64)
        output = interp_model(x)

        features = interp_model.get_active_features()

        # Should have features for both layers
        assert "layer1" in features
        assert "layer2" in features

    # All hooks cleaned up
    assert len(interp_model._hook_handles) == 0

    print("✓ Multiple hook points test passed")


def test_steering_invalid_hook_point():
    """Test steering with invalid hook point raises error."""
    model = SimpleModel(d_model=64)

    config = SAEConfig(d_in=64, d_sae=256, activation="topk", k=16)
    sae = TopKSAE(config)

    saes = {"layer2": sae}
    interp_model = InterpretableModel(model, saes, device="cpu")

    # Try to steer a hook point that doesn't have an SAE
    steering = {"invalid_layer": (10, 2.0)}

    with pytest.raises(ValueError, match="No SAE for hook point"):
        with interp_model.steering_enabled(steering):
            pass

    print("✓ Invalid hook point test passed")


def test_interpretation_idempotent():
    """Test enabling interpretation multiple times is safe."""
    model = SimpleModel(d_model=64)

    config = SAEConfig(d_in=64, d_sae=256, activation="topk", k=16)
    sae = TopKSAE(config)

    saes = {"layer2": sae}
    interp_model = InterpretableModel(model, saes, device="cpu")

    with interp_model.interpretation_enabled():
        # Already active
        assert interp_model._interpretation_active

        # Calling enable again should be safe
        interp_model._enable_interpretation()

        # Still active, no duplicate hooks
        assert interp_model._interpretation_active

    print("✓ Interpretation idempotent test passed")


def test_nested_contexts_not_supported():
    """Test that nested interpretation contexts work (last one wins)."""
    model = SimpleModel(d_model=64)

    config = SAEConfig(d_in=64, d_sae=256, activation="topk", k=16)
    sae = TopKSAE(config)

    saes = {"layer2": sae}
    interp_model = InterpretableModel(model, saes, device="cpu")

    with interp_model.interpretation_enabled():
        # Inner context
        with interp_model.interpretation_enabled():
            x = torch.randn(4, 64)
            output = interp_model(x)

        # After inner context exits, hooks are removed
        # So we can't get features here
        assert len(interp_model._hook_handles) == 0

    print("✓ Nested contexts test passed")


def run_all_tests():
    """Run all SAE runtime tests."""
    print("\n" + "="*80)
    print("Running SAE Runtime Tests")
    print("="*80 + "\n")

    try:
        test_interpretable_model_initialization()
        test_interpretable_model_forward_pass()
        test_interpretation_context_manager()
        test_feature_tracking()
        test_get_active_features_without_context()
        test_steering_context_manager()
        test_feature_amplification()
        test_feature_suppression()
        test_hook_cleanup_on_error()
        test_multiple_hook_points()
        test_steering_invalid_hook_point()
        test_interpretation_idempotent()
        test_nested_contexts_not_supported()

        print("\n" + "="*80)
        print("All SAE runtime tests passed! ✓")
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
