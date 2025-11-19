"""Feature visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class FeatureVisualizer:
    """Visualize feature activations during gameplay."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_feature_timeline(
        self,
        feature_history: list,
        layer: str,
        feature_idx: int,
        save_name: str = None
    ):
        """Plot activation timeline for a specific feature."""
        activations = [
            step[layer][feature_idx].item() if layer in step else 0
            for step in feature_history
        ]

        plt.figure(figsize=(12, 4))
        plt.plot(activations, color='#00ff00', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Activation')
        plt.title(f'Feature {layer}:{feature_idx} Timeline')
        plt.grid(True, alpha=0.3)

        if save_name:
            plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(
                self.output_dir / f'{layer}_{feature_idx}_timeline.png',
                dpi=150,
                bbox_inches='tight'
            )

        plt.close()

    def plot_layer_heatmap(
        self,
        features: dict,
        layer: str,
        top_k: int = 50
    ):
        """Plot heatmap of top features in a layer."""
        feat = features[layer].cpu().numpy()

        # Get top k
        top_idx = np.argsort(np.abs(feat))[-top_k:][::-1]
        top_vals = feat[top_idx]

        plt.figure(figsize=(10, 8))
        plt.barh(range(top_k), top_vals)
        plt.xlabel('Activation')
        plt.ylabel('Feature Index')
        plt.title(f'Top {top_k} Features in {layer}')
        plt.tight_layout()

        plt.savefig(
            self.output_dir / f'{layer}_heatmap.png',
            dpi=150,
            bbox_inches='tight'
        )

        plt.close()
