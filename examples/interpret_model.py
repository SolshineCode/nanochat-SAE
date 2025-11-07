"""
Minimal working example: Use trained SAEs for model interpretation.

This example demonstrates how to:
1. Load a trained nanochat model
2. Load trained SAEs
3. Track feature activations during inference
4. Analyze which features are active

Usage:
    python examples/interpret_model.py --checkpoint models/d20/base_final.pt --sae_dir sae_outputs/
"""

import argparse
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt import GPT, GPTConfig
from sae import InterpretableModel, load_saes


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

    print(f"✓ Loaded model with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Interpret nanochat model with SAEs")

    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to nanochat checkpoint")
    parser.add_argument("--sae_dir", type=str, required=True, help="Directory containing trained SAEs")

    # Inference arguments
    parser.add_argument("--prompt", type=str, default="Hello world", help="Text prompt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    print("="*80)
    print("SAE-Based Model Interpretation Example")
    print("="*80)

    # Load model
    print("\nLoading model...")
    model, config = load_model(args.checkpoint, device=args.device)

    # Load SAEs
    print(f"\nLoading SAEs from {args.sae_dir}...")
    sae_dir = Path(args.sae_dir)
    saes = load_saes(sae_dir, device=args.device)

    if not saes:
        print(f"Error: No SAEs found in {sae_dir}")
        return 1

    print(f"✓ Loaded {len(saes)} SAEs:")
    for hook_point in saes.keys():
        print(f"  - {hook_point}")

    # Create interpretable model
    print("\nWrapping model with SAE interpretation...")
    interp_model = InterpretableModel(model, saes, device=args.device)

    # Tokenize prompt (simplified - in production use proper tokenizer)
    print(f"\nPrompt: '{args.prompt}'")
    # For this example, we'll use random tokens as a placeholder
    # In production, use: tokens = tokenizer.encode(args.prompt)
    tokens = torch.randint(0, config.vocab_size, (1, 10), device=args.device)

    # Run inference with interpretation
    print("\nRunning inference with feature tracking...")
    with torch.no_grad(), interp_model.interpretation_enabled():
        output = interp_model(tokens)
        active_features = interp_model.get_active_features()

    # Analyze active features
    print("\n" + "-"*80)
    print("Active Features Analysis")
    print("-"*80)

    for hook_point, features in active_features.items():
        # features shape: (batch, sequence, d_sae)
        # Count active features across batch and sequence
        is_active = (features != 0).float()
        num_active_per_token = is_active.sum(dim=-1).mean().item()
        total_features = features.shape[-1]
        sparsity = (num_active_per_token / total_features) * 100

        print(f"\n{hook_point}:")
        print(f"  Total features: {total_features}")
        print(f"  Active per token (avg): {num_active_per_token:.1f}")
        print(f"  Sparsity: {sparsity:.2f}%")

        # Find top-k most frequently active features
        k = 5
        feature_activation_counts = is_active.sum(dim=(0, 1))  # Sum over batch and sequence
        top_features = torch.topk(feature_activation_counts, k=min(k, total_features))

        print(f"  Top {k} most active features:")
        for idx, (feat_idx, count) in enumerate(zip(top_features.indices, top_features.values)):
            print(f"    {idx+1}. Feature {feat_idx.item()}: activated {count.item():.0f} times")

    print("\n" + "="*80)
    print("Interpretation Complete!")
    print("="*80)

    print("\nNext steps:")
    print("  1. Analyze specific features using sae_viz")
    print("  2. Steer model behavior by modifying feature activations")
    print("  3. Upload features to Neuronpedia for community analysis")

    return 0


if __name__ == "__main__":
    sys.exit(main())
