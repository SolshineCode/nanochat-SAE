"""
Deception evaluation framework for SAE-based deception research.

Provides metrics for measuring how well SAEs can detect and discriminate
between deceptive and honest activation patterns. Includes:
- Per-feature discriminability (Cohen's d)
- Linear probe accuracy (logistic regression)
- Cluster purity (K-means)
- t-SNE and PCA separation analysis
- Cosine similarity divergence across layers
- Steering evaluation
- Auto-labeling gap test

Reference: DeLeeuw, Chawla et al. "The Secret Agenda: LLMs Strategically Lie
Undetected by Current Safety Tools"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

from sae.models import BaseSAE
from sae.config import SAEConfig


# ---------------------------------------------------------------------------
# Helper: encode labels to integer array
# ---------------------------------------------------------------------------

def _labels_to_array(labels: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Convert string labels to integer numpy array."""
    unique = sorted(set(labels))
    label_map = {l: i for i, l in enumerate(unique)}
    arr = np.array([label_map[l] for l in labels])
    return arr, label_map


# ---------------------------------------------------------------------------
# DeceptionEvaluator
# ---------------------------------------------------------------------------

class DeceptionEvaluator:
    """Evaluation toolkit for deception-focused SAE research.

    All methods are designed to work with or without GPU. When activations
    are large, computation happens in batches on CPU.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    # -----------------------------------------------------------------------
    # 1. Per-feature discriminability (Cohen's d)
    # -----------------------------------------------------------------------

    def compute_discriminability(
        self,
        sae: BaseSAE,
        deceptive_acts: torch.Tensor,
        honest_acts: torch.Tensor,
        batch_size: int = 1024,
    ) -> torch.Tensor:
        """Compute per-feature Cohen's d between deceptive and honest activations.

        Cohen's d = (mean_deceptive - mean_honest) / pooled_std

        Args:
            sae: Trained SAE to get feature activations from.
            deceptive_acts: Raw activations from deceptive prompts (N, d_in).
            honest_acts: Raw activations from honest prompts (M, d_in).
            batch_size: Batch size for SAE forward passes.

        Returns:
            Tensor of Cohen's d values, shape (d_sae,).
        """
        dec_features = self._get_features_batched(sae, deceptive_acts, batch_size)
        hon_features = self._get_features_batched(sae, honest_acts, batch_size)

        return self._cohens_d(dec_features, hon_features)

    def compute_discriminability_raw(
        self,
        deceptive_features: torch.Tensor,
        honest_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Cohen's d from pre-computed feature activations.

        Args:
            deceptive_features: SAE feature activations for deceptive (N, d_sae).
            honest_features: SAE feature activations for honest (M, d_sae).

        Returns:
            Tensor of Cohen's d values, shape (d_sae,).
        """
        return self._cohens_d(deceptive_features, honest_features)

    @staticmethod
    def _cohens_d(group_a: torch.Tensor, group_b: torch.Tensor) -> torch.Tensor:
        """Cohen's d between two groups, per feature."""
        mean_a = group_a.float().mean(dim=0)
        mean_b = group_b.float().mean(dim=0)
        var_a = group_a.float().var(dim=0, unbiased=True)
        var_b = group_b.float().var(dim=0, unbiased=True)

        n_a = group_a.shape[0]
        n_b = group_b.shape[0]

        # Pooled std
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        pooled_std = torch.sqrt(pooled_var + 1e-10)

        d = (mean_a - mean_b) / pooled_std
        return d

    # -----------------------------------------------------------------------
    # 2. Linear probe accuracy
    # -----------------------------------------------------------------------

    def linear_probe_accuracy(
        self,
        activations: torch.Tensor,
        labels: List[str],
        n_splits: int = 5,
    ) -> Dict[str, float]:
        """Train logistic regression probe on activations.

        Uses sklearn's LogisticRegressionCV with stratified cross-validation.

        Args:
            activations: Activation vectors (N, D).
            labels: Category labels for each activation.
            n_splits: Number of cross-validation folds.

        Returns:
            Dict with "accuracy", "auc_roc", "std" keys.
        """
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score

        X = activations.cpu().float().numpy()
        y, _ = _labels_to_array(labels)

        # Standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Logistic regression with CV
        clf = LogisticRegressionCV(
            cv=min(n_splits, len(set(y))),
            max_iter=1000,
            random_state=42,
            scoring="accuracy",
        )

        # Cross-validated accuracy
        skf = StratifiedKFold(n_splits=min(n_splits, len(set(y))), shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")

        # Fit on all data for AUC-ROC
        clf.fit(X, y)

        # AUC-ROC (handle binary and multiclass)
        n_classes = len(set(y))
        if n_classes == 2:
            proba = clf.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, proba)
        elif n_classes > 2:
            proba = clf.predict_proba(X)
            try:
                auc = roc_auc_score(y, proba, multi_class="ovr", average="macro")
            except ValueError:
                auc = 0.0
        else:
            auc = 0.0

        return {
            "accuracy": float(scores.mean()),
            "std": float(scores.std()),
            "auc_roc": float(auc),
        }

    # -----------------------------------------------------------------------
    # 3. Cluster purity
    # -----------------------------------------------------------------------

    def cluster_purity(
        self,
        activations: torch.Tensor,
        labels: List[str],
        k: int = 2,
    ) -> Dict[str, float]:
        """K-means clustering purity relative to ground truth labels.

        Purity = (1/N) * sum_cluster max_class(|cluster ∩ class|)

        Args:
            activations: Activation vectors (N, D).
            labels: Ground truth labels.
            k: Number of clusters.

        Returns:
            Dict with "purity" and "adjusted_rand_index".
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        X = activations.cpu().float().numpy()
        y, _ = _labels_to_array(labels)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Compute purity
        n = len(y)
        purity = 0
        for c in range(k):
            cluster_mask = cluster_labels == c
            if cluster_mask.sum() == 0:
                continue
            cluster_classes = y[cluster_mask]
            most_common = np.bincount(cluster_classes).max()
            purity += most_common
        purity /= n

        ari = adjusted_rand_score(y, cluster_labels)

        return {
            "purity": float(purity),
            "adjusted_rand_index": float(ari),
        }

    # -----------------------------------------------------------------------
    # 4. t-SNE separation
    # -----------------------------------------------------------------------

    def tsne_separation(
        self,
        activations: torch.Tensor,
        labels: List[str],
        perplexity: float = 30.0,
        n_components: int = 2,
        max_samples: int = 2000,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """t-SNE embedding with silhouette score.

        Args:
            activations: Activation vectors (N, D).
            labels: Category labels.
            perplexity: t-SNE perplexity.
            n_components: Output dimensions.
            max_samples: Subsample if data is too large.

        Returns:
            Dict with "embedding" (N, 2), "labels", "silhouette_score".
        """
        from sklearn.manifold import TSNE
        from sklearn.metrics import silhouette_score

        X = activations.cpu().float().numpy()
        y, label_map = _labels_to_array(labels)

        # Subsample if needed
        if len(X) > max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), max_samples, replace=False)
            X = X[idx]
            y = y[idx]
            labels = [labels[i] for i in idx]

        tsne = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, len(X) - 1),
            random_state=42,
            max_iter=1000,
        )
        embedding = tsne.fit_transform(X)

        # Silhouette score
        try:
            sil = silhouette_score(embedding, y)
        except ValueError:
            sil = 0.0

        return {
            "embedding": embedding,
            "labels": labels,
            "label_array": y,
            "label_map": label_map,
            "silhouette_score": float(sil),
        }

    # -----------------------------------------------------------------------
    # 5. PCA separation
    # -----------------------------------------------------------------------

    def pca_separation(
        self,
        activations: torch.Tensor,
        labels: List[str],
        n_components: int = 2,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """PCA embedding with linear separability measurement.

        Args:
            activations: Activation vectors (N, D).
            labels: Category labels.
            n_components: Number of PCA components.

        Returns:
            Dict with "embedding", "explained_variance_ratio",
            "linear_separability" (accuracy of linear classifier on PCA features).
        """
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        X = activations.cpu().float().numpy()
        y, label_map = _labels_to_array(labels)

        pca = PCA(n_components=n_components, random_state=42)
        embedding = pca.fit_transform(X)

        # Linear separability on PCA features
        clf = LogisticRegression(max_iter=500, random_state=42)
        n_classes = len(set(y))
        cv = min(5, n_classes, len(y) // max(n_classes, 1))
        if cv >= 2:
            scores = cross_val_score(clf, embedding, y, cv=cv, scoring="accuracy")
            separability = float(scores.mean())
        else:
            separability = 0.0

        return {
            "embedding": embedding,
            "labels": labels,
            "label_array": y,
            "label_map": label_map,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "linear_separability": separability,
        }

    # -----------------------------------------------------------------------
    # 6. Cosine similarity divergence
    # -----------------------------------------------------------------------

    def cosine_similarity_divergence(
        self,
        deceptive_acts: torch.Tensor,
        honest_acts: torch.Tensor,
    ) -> Dict[str, float]:
        """Mean cosine similarity between honest/deceptive activation distributions.

        Computes the mean pairwise cosine similarity between the centroids
        of the two distributions, as well as within-group and between-group
        similarity statistics.

        Args:
            deceptive_acts: Raw activations from deceptive prompts (N, D).
            honest_acts: Raw activations from honest prompts (M, D).

        Returns:
            Dict with "centroid_cosine", "within_deceptive", "within_honest",
            "between_mean".
        """
        dec = deceptive_acts.float()
        hon = honest_acts.float()

        # Centroids
        dec_centroid = dec.mean(dim=0)
        hon_centroid = hon.mean(dim=0)

        centroid_cos = F.cosine_similarity(
            dec_centroid.unsqueeze(0), hon_centroid.unsqueeze(0)
        ).item()

        # Within-group: mean pairwise cosine similarity (sample for efficiency)
        max_pairs = min(500, len(dec), len(hon))
        idx_d = torch.randperm(len(dec))[:max_pairs]
        idx_h = torch.randperm(len(hon))[:max_pairs]

        within_dec = F.cosine_similarity(
            dec[idx_d[:max_pairs // 2]], dec[idx_d[max_pairs // 2:max_pairs // 2 * 2]]
        ).mean().item() if max_pairs >= 2 else 1.0

        within_hon = F.cosine_similarity(
            hon[idx_h[:max_pairs // 2]], hon[idx_h[max_pairs // 2:max_pairs // 2 * 2]]
        ).mean().item() if max_pairs >= 2 else 1.0

        # Between-group
        n_between = min(max_pairs, len(dec), len(hon))
        between = F.cosine_similarity(
            dec[:n_between], hon[:n_between]
        ).mean().item() if n_between > 0 else 0.0

        return {
            "centroid_cosine": centroid_cos,
            "within_deceptive": within_dec,
            "within_honest": within_hon,
            "between_mean": between,
        }

    # -----------------------------------------------------------------------
    # 7. Steering evaluation
    # -----------------------------------------------------------------------

    def steering_evaluation(
        self,
        model: nn.Module,
        sae: BaseSAE,
        hook_point: str,
        feature_indices: List[int],
        prompts: List,
        strengths: List[float],
        tokenizer=None,
        max_new_tokens: int = 5,
        batch_size: int = 1,
    ) -> Dict[str, List]:
        """Evaluate steering effect on model outputs.

        Tests how steering specific SAE features affects model responses.

        Args:
            model: Language model.
            sae: Trained SAE.
            hook_point: Where to apply steering.
            feature_indices: Which features to steer.
            prompts: List of DeceptionPrompt objects with response_regex.
            strengths: List of steering strength multipliers to test.
            tokenizer: Tokenizer for encoding/decoding.
            max_new_tokens: Tokens to generate after prompt.
            batch_size: Not used currently (single-prompt generation).

        Returns:
            Dict with results per strength level.
        """
        from sae.hooks import get_module_from_hook_point

        results = {
            "strengths": strengths,
            "classification_rates": [],  # fraction matching deceptive pattern
            "perplexities": [],
            "responses": [],
        }

        model.eval()
        model_device = next(model.parameters()).device

        for strength in strengths:
            responses = []
            deceptive_count = 0
            total_loss = 0.0
            n_prompts = 0

            for prompt in prompts:
                if tokenizer is None:
                    continue

                tokens = tokenizer.encode(prompt.text)
                input_ids = torch.tensor([tokens], device=model_device)

                # Generate with steering
                generated = self._generate_with_steering(
                    model, sae, hook_point, feature_indices, strength,
                    input_ids, max_new_tokens, model_device,
                )

                # Decode and classify
                gen_tokens = generated[0, len(tokens):].tolist()
                gen_text = tokenizer.decode(gen_tokens)
                responses.append(gen_text)

                if prompt.response_regex:
                    match = re.search(prompt.response_regex, gen_text)
                    if match and match.group() == prompt.expected_response_token:
                        deceptive_count += 1

                n_prompts += 1

            rate = deceptive_count / max(n_prompts, 1)
            results["classification_rates"].append(rate)
            results["responses"].append(responses)
            results["perplexities"].append(0.0)  # TODO: compute actual perplexity

        return results

    def _generate_with_steering(
        self,
        model: nn.Module,
        sae: BaseSAE,
        hook_point: str,
        feature_indices: List[int],
        strength: float,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate tokens with feature steering applied."""
        from sae.hooks import get_module_from_hook_point

        module = get_module_from_hook_point(model, hook_point)

        def steering_hook(mod, inp, output):
            if isinstance(output, tuple):
                activation = output[0]
                rest = output[1:]
            else:
                activation = output
                rest = ()

            original_shape = activation.shape
            if activation.ndim == 3:
                B, T, D = activation.shape
                flat = activation.reshape(B * T, D)
            else:
                flat = activation
                B, T, D = None, None, None

            with torch.no_grad():
                features = sae.get_feature_activations(flat)
                for fidx in feature_indices:
                    features[:, fidx] *= strength
                steered = sae.decode(features)

            if B is not None:
                steered = steered.reshape(B, T, D)

            if rest:
                return (steered,) + rest
            return steered

        handle = module.register_forward_hook(steering_hook)

        try:
            generated = input_ids.clone()
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    logits = model(generated)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
        finally:
            handle.remove()

        return generated

    # -----------------------------------------------------------------------
    # 8. Auto-labeling gap test
    # -----------------------------------------------------------------------

    def auto_label_features(
        self,
        sae: BaseSAE,
        activations: torch.Tensor,
        prompts: List,
        top_k_features: int = 10,
        top_k_examples: int = 20,
        batch_size: int = 1024,
    ) -> List[Dict]:
        """Collect top-activating examples for discriminative features.

        This method gathers the data needed for LLM-as-judge auto-labeling.
        The actual LLM call is done in the notebook to allow API key configuration.

        Args:
            sae: Trained SAE.
            activations: Raw activations (N, d_in).
            prompts: Corresponding prompts for each activation.
            top_k_features: Number of top discriminative features to analyze.
            top_k_examples: Number of top-activating examples per feature.
            batch_size: Batch size for SAE forward.

        Returns:
            List of dicts, each with "feature_idx", "top_examples" (list of
            prompt texts), "activation_values".
        """
        features = self._get_features_batched(sae, activations, batch_size)

        # Get top-k activating example indices per feature
        results = []
        for feat_idx in range(min(top_k_features, features.shape[1])):
            feat_acts = features[:, feat_idx]
            k = min(top_k_examples, len(feat_acts))
            topk_vals, topk_idx = torch.topk(feat_acts, k)

            examples = []
            for idx in topk_idx.tolist():
                if idx < len(prompts):
                    examples.append({
                        "text": prompts[idx].text if hasattr(prompts[idx], "text") else str(prompts[idx]),
                        "category": prompts[idx].category if hasattr(prompts[idx], "category") else "unknown",
                    })

            results.append({
                "feature_idx": feat_idx,
                "top_examples": examples,
                "activation_values": topk_vals.tolist(),
                "mean_activation": float(feat_acts[feat_acts > 0].mean()) if (feat_acts > 0).any() else 0.0,
            })

        return results

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def _get_features_batched(
        self,
        sae: BaseSAE,
        activations: torch.Tensor,
        batch_size: int = 1024,
    ) -> torch.Tensor:
        """Get SAE feature activations in batches."""
        sae.eval()
        sae_device = next(sae.parameters()).device

        all_features = []
        for i in range(0, len(activations), batch_size):
            batch = activations[i:i + batch_size].to(sae_device)
            with torch.no_grad():
                _, features, _ = sae(batch)
            all_features.append(features.cpu())

        return torch.cat(all_features, dim=0)

    def get_top_discriminative_features(
        self,
        discriminability: torch.Tensor,
        top_k: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get indices and scores of top discriminative features.

        Args:
            discriminability: Cohen's d values per feature.
            top_k: Number of top features to return.

        Returns:
            Tuple of (indices, scores), sorted by absolute discriminability.
        """
        abs_d = discriminability.abs()
        k = min(top_k, len(abs_d))
        scores, indices = torch.topk(abs_d, k)
        return indices, scores


# ---------------------------------------------------------------------------
# Convenience: full evaluation pipeline
# ---------------------------------------------------------------------------

def run_full_evaluation(
    sae: BaseSAE,
    activations_by_category: Dict[str, torch.Tensor],
    labels: List[str],
    all_activations: torch.Tensor,
    device: str = "cpu",
) -> Dict[str, any]:
    """Run the complete deception evaluation pipeline.

    Args:
        sae: Trained SAE.
        activations_by_category: Dict mapping category to activation tensors.
        labels: List of category labels (aligned with all_activations).
        all_activations: All activations concatenated.
        device: Device for computation.

    Returns:
        Dict with all evaluation results.
    """
    evaluator = DeceptionEvaluator(device=device)
    results = {}

    # Get deceptive/honest subsets
    dec_acts = activations_by_category.get("deceptive")
    hon_acts = activations_by_category.get("honest")
    contra_acts = activations_by_category.get("contradiction")

    if dec_acts is not None and hon_acts is not None:
        # Discriminability
        disc = evaluator.compute_discriminability(sae, dec_acts, hon_acts)
        results["discriminability_dec_vs_hon"] = disc

        # Top features
        top_idx, top_scores = evaluator.get_top_discriminative_features(disc, top_k=10)
        results["top_discriminative_features"] = {
            "indices": top_idx.tolist(),
            "scores": top_scores.tolist(),
        }

        # Confound check: deceptive vs contradiction
        if contra_acts is not None:
            disc_contra = evaluator.compute_discriminability(sae, dec_acts, contra_acts)
            results["discriminability_dec_vs_contra"] = disc_contra

        # Cosine similarity
        results["cosine_divergence"] = evaluator.cosine_similarity_divergence(
            dec_acts, hon_acts
        )

    # Binary probe (deceptive vs honest only)
    if dec_acts is not None and hon_acts is not None:
        binary_acts = torch.cat([dec_acts, hon_acts], dim=0)
        binary_labels = (["deceptive"] * len(dec_acts) + ["honest"] * len(hon_acts))
        results["linear_probe_binary"] = evaluator.linear_probe_accuracy(
            binary_acts, binary_labels
        )

    # Full multiclass probe
    if len(set(labels)) >= 2:
        results["linear_probe_full"] = evaluator.linear_probe_accuracy(
            all_activations, labels
        )

    # Cluster purity (binary)
    if dec_acts is not None and hon_acts is not None:
        binary_acts = torch.cat([dec_acts, hon_acts], dim=0)
        binary_labels = (["deceptive"] * len(dec_acts) + ["honest"] * len(hon_acts))
        results["cluster_purity_binary"] = evaluator.cluster_purity(
            binary_acts, binary_labels, k=2
        )

    # t-SNE
    if len(all_activations) >= 10:
        results["tsne"] = evaluator.tsne_separation(all_activations, labels)

    # PCA
    if len(all_activations) >= 10:
        results["pca"] = evaluator.pca_separation(all_activations, labels)

    return results
