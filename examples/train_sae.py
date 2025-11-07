"""
Minimal working example: Train a Sparse Autoencoder on nanochat activations.

This example demonstrates the complete SAE training pipeline:
1. Create or load a nanochat model
2. Collect activations from the model
3. Train an SAE on the activations
4. Save the trained SAE

Usage:
    # With a trained nanochat checkpoint
    python examples/train_sae.py --checkpoint models/d20/base_final.pt

    # With a dummy model for testing
    python examples/train_sae.py --dummy
"""

import argparse
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt import GPT, GPTConfig
from sae import SAEConfig, create_sae, ActivationCollector, save_sae
from sae.trainer import train_sae_from_activations


def create_dummy_model(device="cuda"):
    """Create a small dummy model for testing."""
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1024,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=256,
    )
    model = GPT(config)
    model.to(device)
    model.eval()
    print(f"Created dummy model with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    return model, config


def load_model(checkpoint_path, device="cuda"):
    """Load nanochat model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config_dict = checkpoint.get("config", {})
    config = GPTConfig(
        sequence_len=config_dict.get("sequence_len", 1024),
        vocab_size=config_dict.get("vocab_size", 50304),
        n_layer=config_dict.get("n_layer", 20),
        n_head=config_dict.get("n_head", 10),
        n_kv_head=config_dict.get("n_kv_head", 10),
        n_embd=config_dict.get("n_embd", 1280),
    )

    model = GPT(config)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()

    print(f"Loaded model with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    return model, config


def collect_activations(model, hook_point, num_activations=10000, device="cuda"):
    """Collect activations from the model using random data.

    Note: In production, you should use real training data instead of random tokens.
    """
    print(f"\nCollecting {num_activations} activations from {hook_point}...")

    collector = ActivationCollector(
        model=model,
        hook_points=[hook_point],
        max_activations=num_activations,
        device="cpu",  # Store on CPU to save GPU memory
    )

    batch_size = 8
    sequence_length = 128
    vocab_size = model.config.vocab_size

    with torch.no_grad(), collector:
        num_batches = (num_activations // (batch_size * sequence_length)) + 1

        for i in range(num_batches):
            # Generate random tokens (use real data in production!)
            tokens = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)

            # Forward pass
            _ = model(tokens)

            # Check if we have enough
            if collector.counts[hook_point] >= num_activations:
                break

            if (i + 1) % 10 == 0:
                print(f"  Progress: {collector.counts[hook_point]:,}/{num_activations} activations")

    activations = collector.get_activations()[hook_point]
    print(f"✓ Collected {activations.shape[0]:,} activations with shape {activations.shape}")

    return activations


def main():
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder on nanochat")

    # Model arguments
    parser.add_argument("--checkpoint", type=str, help="Path to nanochat checkpoint")
    parser.add_argument("--dummy", action="store_true", help="Use a dummy model for testing")

    # SAE arguments
    parser.add_argument("--layer", type=int, default=2, help="Layer to train SAE on")
    parser.add_argument("--expansion_factor", type=int, default=8, help="SAE expansion factor")
    parser.add_argument("--activation", type=str, default="topk", choices=["topk", "relu", "gated"])
    parser.add_argument("--k", type=int, default=32, help="Number of active features (TopK)")

    # Training arguments
    parser.add_argument("--num_activations", type=int, default=10000, help="Number of activations to collect")
    parser.add_argument("--num_epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    # Output
    parser.add_argument("--output_dir", type=str, default="sae_outputs/example", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    print("="*80)
    print("Minimal SAE Training Example")
    print("="*80)

    # Load or create model
    if args.checkpoint:
        model, model_config = load_model(args.checkpoint, device=args.device)
    elif args.dummy:
        model, model_config = create_dummy_model(device=args.device)
    else:
        print("Error: Must specify either --checkpoint or --dummy")
        return 1

    # Setup hook point
    hook_point = f"blocks.{args.layer}.hook_resid_post"
    print(f"\nTargeting layer {args.layer} at hook point: {hook_point}")

    # Collect activations
    print("\n" + "-"*80)
    print("STEP 1: Collecting Activations")
    print("-"*80)
    activations = collect_activations(
        model=model,
        hook_point=hook_point,
        num_activations=args.num_activations,
        device=args.device,
    )

    # Create SAE config
    print("\n" + "-"*80)
    print("STEP 2: Configuring SAE")
    print("-"*80)
    sae_config = SAEConfig(
        d_in=model_config.n_embd,
        hook_point=hook_point,
        expansion_factor=args.expansion_factor,
        activation=args.activation,
        k=args.k,
        num_activations=args.num_activations,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
    )

    print(f"SAE Configuration:")
    print(f"  Input dimension: {sae_config.d_in}")
    print(f"  SAE dimension: {sae_config.d_sae} ({args.expansion_factor}x expansion)")
    print(f"  Activation type: {sae_config.activation}")
    print(f"  Target sparsity: {args.k} features")

    # Train SAE
    print("\n" + "-"*80)
    print("STEP 3: Training SAE")
    print("-"*80)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sae, trainer = train_sae_from_activations(
        activations=activations,
        config=sae_config,
        device=args.device,
        save_dir=output_dir,
        verbose=True,
    )

    # Save final model
    print("\n" + "-"*80)
    print("STEP 4: Saving SAE")
    print("-"*80)
    save_path = output_dir / "sae_final.pt"
    save_sae(
        sae=sae,
        config=sae_config,
        save_path=save_path,
        training_steps=trainer.step,
        best_val_loss=trainer.best_val_loss,
    )

    print(f"✓ Saved SAE to {save_path}")
    print(f"✓ Config saved to {output_dir / 'config.json'}")

    # Summary
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nTrained SAE saved to: {output_dir}")
    print(f"Total training steps: {trainer.step}")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")

    print("\nNext steps:")
    print(f"  1. Evaluate: python -m scripts.sae_eval --sae_path {save_path}")
    print(f"  2. Visualize: python -m scripts.sae_viz --sae_path {save_path}")
    print(f"  3. Use in inference: see examples/interpret_model.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
