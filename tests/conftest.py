"""
Shared pytest fixtures for nanochat-SAE tests.

This file provides reusable fixtures that multiple test files can use.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sae.config import SAEConfig
from sae.models import TopKSAE, ReLUSAE, GatedSAE


@pytest.fixture
def small_sae_config():
    """Fixture providing a small SAE config for fast tests."""
    return SAEConfig(
        d_in=32,
        d_sae=128,
        activation="topk",
        k=8,
        batch_size=16,
        num_epochs=2,
    )


@pytest.fixture
def medium_sae_config():
    """Fixture providing a medium SAE config for more thorough tests."""
    return SAEConfig(
        d_in=64,
        d_sae=256,
        activation="topk",
        k=16,
        batch_size=32,
        num_epochs=5,
    )


@pytest.fixture
def topk_sae(small_sae_config):
    """Fixture providing a TopK SAE model."""
    return TopKSAE(small_sae_config)


@pytest.fixture
def relu_sae():
    """Fixture providing a ReLU SAE model."""
    config = SAEConfig(
        d_in=32,
        d_sae=128,
        activation="relu",
        l1_coefficient=1e-3,
    )
    return ReLUSAE(config)


@pytest.fixture
def gated_sae():
    """Fixture providing a Gated SAE model."""
    config = SAEConfig(
        d_in=32,
        d_sae=128,
        activation="gated",
        l1_coefficient=1e-3,
    )
    return GatedSAE(config)


@pytest.fixture
def sample_activations_small():
    """Fixture providing small sample activations for quick tests."""
    return torch.randn(500, 32)


@pytest.fixture
def sample_activations_medium():
    """Fixture providing medium sample activations."""
    return torch.randn(2000, 64)


@pytest.fixture
def sample_activations_with_val():
    """Fixture providing train and validation activations."""
    train_activations = torch.randn(1000, 64)
    val_activations = torch.randn(200, 64)
    return train_activations, val_activations


@pytest.fixture
def simple_conversation():
    """Fixture providing a simple conversation for tokenizer tests."""
    return {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
        ]
    }


@pytest.fixture
def conversation_with_tools():
    """Fixture providing a conversation with Python tool usage."""
    return {
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me calculate that"},
                    {"type": "python", "text": "2 + 2"},
                    {"type": "python_output", "text": "4"},
                    {"type": "text", "text": "The answer is 4"},
                ]
            },
        ]
    }


@pytest.fixture
def conversation_with_system():
    """Fixture providing a conversation with system message."""
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }


@pytest.fixture
def device():
    """Fixture providing the test device (cuda if available, else cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: integration tests requiring full model"
    )
    config.addinivalue_line(
        "markers", "security: security-critical execution tests"
    )
    config.addinivalue_line(
        "markers", "gpu: tests requiring CUDA"
    )
