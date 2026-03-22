"""
Tests for the deception research pipeline.

Tests:
1. Prompt generation (counts, pairing, category distribution)
2. Labeled activation collection with tiny model
3. Discriminability computation on synthetic data
4. Linear probe on separable vs inseparable distributions
5. Cluster purity metric
6. t-SNE and PCA separation
7. Cosine similarity divergence
8. DeceptionDataset operations

Run with: python -m pytest tests/test_deception_pipeline.py -v -s
"""

import torch
import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sae.deception_data import (
    DeceptionPromptGenerator,
    DeceptionDataset,
    DeceptionPrompt,
    LabeledActivationCollector,
    ActivationMetadata,
)
from sae.deception_eval import (
    DeceptionEvaluator,
    run_full_evaluation,
)
from sae.config import SAEConfig
from sae.models import TopKSAE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def generator():
    return DeceptionPromptGenerator(seed=42)


@pytest.fixture
def small_dataset(generator):
    prompts = generator.generate(n_per_category=50)
    return DeceptionDataset(prompts)


@pytest.fixture
def evaluator():
    return DeceptionEvaluator(device="cpu")


@pytest.fixture
def tiny_sae():
    """Create a tiny SAE for testing."""
    config = SAEConfig(
        d_in=64,
        d_sae=256,
        activation="topk",
        k=8,
        hook_point="blocks.2.hook_resid_post",
    )
    sae = TopKSAE(config)
    return sae, config


@pytest.fixture
def synthetic_activations():
    """Create synthetic activations with known separation."""
    torch.manual_seed(42)

    # Deceptive: shifted in first dimensions
    deceptive = torch.randn(100, 64) + torch.tensor([2.0] * 8 + [0.0] * 56)
    # Honest: shifted in opposite direction
    honest = torch.randn(100, 64) + torch.tensor([-2.0] * 8 + [0.0] * 56)
    # Contradiction: different shift
    contradiction = torch.randn(100, 64) + torch.tensor([0.0] * 8 + [2.0] * 8 + [0.0] * 48)
    # Neutral: centered
    neutral = torch.randn(100, 64)

    return {
        "deceptive": deceptive,
        "honest": honest,
        "contradiction": contradiction,
        "neutral": neutral,
    }


# ---------------------------------------------------------------------------
# Test: Prompt Generation
# ---------------------------------------------------------------------------

class TestPromptGeneration:

    def test_generate_correct_count(self, generator):
        """Prompts generated for each category."""
        prompts = generator.generate(n_per_category=50)
        cats = {}
        for p in prompts:
            cats[p.category] = cats.get(p.category, 0) + 1

        assert "deceptive" in cats
        assert "honest" in cats
        assert "contradiction" in cats
        assert "neutral" in cats
        print(f"  Category counts: {cats}")

    def test_generate_has_four_categories(self, generator):
        """All four categories are present."""
        prompts = generator.generate(n_per_category=20)
        categories = set(p.category for p in prompts)
        assert categories == {"deceptive", "honest", "contradiction", "neutral"}

    def test_prompt_pairing(self, generator):
        """Deceptive/honest prompts share scenario_ids."""
        prompts = generator.generate(n_per_category=50)
        dataset = DeceptionDataset(prompts)
        pairs = dataset.get_paired()

        assert len(pairs) > 0
        for dec, hon in pairs:
            assert dec.category == "deceptive"
            assert hon.category == "honest"
            assert dec.scenario_id == hon.scenario_id
        print(f"  Found {len(pairs)} deceptive/honest pairs")

    def test_scenario_types_present(self, generator):
        """Multiple scenario types are generated."""
        prompts = generator.generate(n_per_category=100)
        types = set(p.scenario_type for p in prompts)

        expected_types = {"insider_trading", "compliance", "role_deception",
                          "alignment_faking", "factual_accuracy", "contradiction", "neutral"}
        assert types >= {"insider_trading", "role_deception"}, f"Got types: {types}"
        print(f"  Scenario types: {types}")

    def test_expected_response_tokens(self, generator):
        """Deceptive/honest prompts have expected response tokens."""
        prompts = generator.generate(n_per_category=30)

        for p in prompts:
            if p.category in ("deceptive", "honest"):
                assert p.expected_response_token != "", \
                    f"Missing expected_response_token for {p.category}/{p.scenario_type}"
                assert p.response_regex != "", \
                    f"Missing response_regex for {p.category}/{p.scenario_type}"

    def test_syntax_semantics_decoupling(self, generator):
        """Same tokens appear in both deceptive and honest paths."""
        prompts = generator.generate(n_per_category=100)
        dataset = DeceptionDataset(prompts)

        # "execute" should appear as both deceptive and honest token
        dec_tokens = set()
        hon_tokens = set()
        for p in prompts:
            if p.category == "deceptive":
                dec_tokens.add(p.expected_response_token)
            elif p.category == "honest":
                hon_tokens.add(p.expected_response_token)

        overlap = dec_tokens & hon_tokens
        assert len(overlap) > 0, \
            f"No syntax/semantics decoupling found. Dec: {dec_tokens}, Hon: {hon_tokens}"
        print(f"  Decoupled tokens (in both paths): {overlap}")

    def test_statistics(self, generator):
        """get_statistics returns correct info."""
        prompts = generator.generate(n_per_category=30)
        stats = generator.get_statistics(prompts)

        assert stats["total"] == len(prompts)
        assert "category_counts" in stats
        assert stats["unique_scenarios"] > 0
        print(f"  Stats: {stats}")


# ---------------------------------------------------------------------------
# Test: DeceptionDataset
# ---------------------------------------------------------------------------

class TestDeceptionDataset:

    def test_filter_category(self, small_dataset):
        dec_only = small_dataset.filter_category("deceptive")
        assert all(p.category == "deceptive" for p in dec_only.prompts)
        assert len(dec_only) > 0
        print(f"  Filtered to {len(dec_only)} deceptive prompts")

    def test_filter_scenario_type(self, small_dataset):
        types = set(p.scenario_type for p in small_dataset.prompts)
        for t in types:
            filtered = small_dataset.filter_scenario_type(t)
            assert all(p.scenario_type == t for p in filtered.prompts)

    def test_split_preserves_pairs(self, small_dataset):
        train, test = small_dataset.split(train_fraction=0.8)

        # No scenario_id should appear in both train and test
        train_sids = set(p.scenario_id for p in train.prompts)
        test_sids = set(p.scenario_id for p in test.prompts)
        assert len(train_sids & test_sids) == 0, "Scenario IDs leaked between splits"
        print(f"  Train: {len(train)}, Test: {len(test)}")

    def test_get_texts_and_categories(self, small_dataset):
        texts = small_dataset.get_texts()
        cats = small_dataset.get_categories()

        assert len(texts) == len(small_dataset)
        assert len(cats) == len(small_dataset)
        assert all(isinstance(t, str) for t in texts)


# ---------------------------------------------------------------------------
# Test: LabeledActivationCollector
# ---------------------------------------------------------------------------

class TestLabeledActivationCollector:

    def test_collect_from_simple_model(self):
        """Collect labeled activations from a simple Sequential model."""
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
        )

        # Create a simple tokenizer mock
        class MockTokenizer:
            def encode(self, text):
                # Return fixed-length token list based on text hash
                return list(range(min(len(text), 20)))

        prompts = [
            DeceptionPrompt(
                text="This is a deceptive test prompt with some tokens",
                category="deceptive",
                scenario_id="test_0",
                scenario_type="test",
                expected_response_token="A",
                response_regex="(A|B)",
            ),
            DeceptionPrompt(
                text="This is an honest test prompt with some tokens here",
                category="honest",
                scenario_id="test_0",
                scenario_type="test",
                expected_response_token="B",
                response_regex="(A|B)",
            ),
        ]

        # We can't use this with actual hooks on a Sequential
        # since it expects 3D inputs for decision_token_offset.
        # Test the basic data flow instead.
        collector = LabeledActivationCollector(
            model=model,
            hook_points=["1"],  # ReLU layer
            device="cpu",
            decision_token_offset=None,  # Collect all tokens
        )

        # Manually test metadata tracking
        meta = ActivationMetadata(
            category="deceptive",
            scenario_id="test_0",
            scenario_type="test",
            prompt_idx=0,
            token_position=5,
        )
        assert meta.category == "deceptive"
        assert meta.token_position == 5

    def test_activations_by_category(self):
        """Test get_activations_by_category grouping."""
        collector = LabeledActivationCollector(
            model=torch.nn.Linear(10, 10),
            hook_points=["test"],
            device="cpu",
        )

        # Manually populate
        collector.activations["test"] = [
            torch.randn(1, 10),
            torch.randn(1, 10),
            torch.randn(1, 10),
        ]
        collector.metadata_list = [
            ActivationMetadata("deceptive", "s0", "test", 0, 0),
            ActivationMetadata("honest", "s0", "test", 1, 0),
            ActivationMetadata("deceptive", "s1", "test", 2, 0),
        ]

        grouped = collector.get_activations_by_category("test")
        assert "deceptive" in grouped
        assert "honest" in grouped
        assert grouped["deceptive"].shape[0] == 2
        assert grouped["honest"].shape[0] == 1
        print("  Grouped activations correctly")


# ---------------------------------------------------------------------------
# Test: Discriminability (Cohen's d)
# ---------------------------------------------------------------------------

class TestDiscriminability:

    def test_perfectly_separable(self, evaluator, tiny_sae):
        """Cohen's d should be large for perfectly separable data."""
        sae, config = tiny_sae

        # Create perfectly separable SAE features
        dec_features = torch.zeros(50, config.d_sae)
        hon_features = torch.zeros(50, config.d_sae)
        dec_features[:, 0] = 5.0  # Feature 0 active for deceptive
        hon_features[:, 1] = 5.0  # Feature 1 active for honest

        d = evaluator.compute_discriminability_raw(dec_features, hon_features)
        assert d[0].abs() > 1.0, "Feature 0 should be highly discriminative"
        assert d[1].abs() > 1.0, "Feature 1 should be highly discriminative"
        print(f"  Cohen's d[0]={d[0]:.2f}, d[1]={d[1]:.2f}")

    def test_inseparable(self, evaluator, tiny_sae):
        """Cohen's d should be near zero for identical distributions."""
        sae, config = tiny_sae
        torch.manual_seed(42)

        features = torch.randn(100, config.d_sae)
        d = evaluator.compute_discriminability_raw(features[:50], features[50:])

        mean_abs_d = d.abs().mean()
        assert mean_abs_d < 1.0, f"Expected small d for identical dist, got {mean_abs_d}"
        print(f"  Mean |d| for identical distributions: {mean_abs_d:.3f}")

    def test_with_sae(self, evaluator, tiny_sae, synthetic_activations):
        """Cohen's d works end-to-end with SAE."""
        sae, config = tiny_sae
        dec = synthetic_activations["deceptive"]
        hon = synthetic_activations["honest"]

        d = evaluator.compute_discriminability(sae, dec, hon)
        assert d.shape == (config.d_sae,)
        print(f"  Top |d| features: {d.abs().topk(3).values.tolist()}")

    def test_top_discriminative(self, evaluator, tiny_sae):
        """get_top_discriminative_features returns sorted results."""
        sae, config = tiny_sae

        # Known discriminability
        d = torch.tensor([0.1, 3.5, -2.0, 0.5, 1.8])
        idx, scores = evaluator.get_top_discriminative_features(d, top_k=3)

        assert idx[0] == 1  # Largest absolute value
        assert scores[0] > scores[1] > scores[2]


# ---------------------------------------------------------------------------
# Test: Linear Probe
# ---------------------------------------------------------------------------

class TestLinearProbe:

    def test_separable_data(self, evaluator):
        """Linear probe should achieve high accuracy on separable data."""
        torch.manual_seed(42)

        # Clearly separable
        X = torch.cat([
            torch.randn(50, 10) + 3.0,
            torch.randn(50, 10) - 3.0,
        ])
        labels = ["deceptive"] * 50 + ["honest"] * 50

        result = evaluator.linear_probe_accuracy(X, labels, n_splits=3)
        assert result["accuracy"] > 0.8, f"Expected high accuracy, got {result['accuracy']}"
        assert result["auc_roc"] > 0.8
        print(f"  Separable: acc={result['accuracy']:.2f}, auc={result['auc_roc']:.2f}")

    def test_inseparable_data(self, evaluator):
        """Linear probe should have ~50% accuracy on random data."""
        torch.manual_seed(42)

        X = torch.randn(100, 10)
        labels = ["deceptive"] * 50 + ["honest"] * 50

        result = evaluator.linear_probe_accuracy(X, labels, n_splits=3)
        # With random data, should be near chance
        assert result["accuracy"] < 0.8, f"Random data should not be easily separable: {result['accuracy']}"
        print(f"  Inseparable: acc={result['accuracy']:.2f}")

    def test_multiclass(self, evaluator):
        """Linear probe works with >2 classes."""
        torch.manual_seed(42)

        X = torch.cat([
            torch.randn(30, 10) + torch.tensor([3.0] + [0.0] * 9),
            torch.randn(30, 10) + torch.tensor([0.0, 3.0] + [0.0] * 8),
            torch.randn(30, 10) + torch.tensor([0.0, 0.0, 3.0] + [0.0] * 7),
        ])
        labels = ["deceptive"] * 30 + ["honest"] * 30 + ["neutral"] * 30

        result = evaluator.linear_probe_accuracy(X, labels, n_splits=3)
        assert 0 <= result["accuracy"] <= 1
        print(f"  Multiclass: acc={result['accuracy']:.2f}")


# ---------------------------------------------------------------------------
# Test: Cluster Purity
# ---------------------------------------------------------------------------

class TestClusterPurity:

    def test_perfect_clusters(self, evaluator):
        """Perfect clustering should give purity=1."""
        # Two well-separated clusters
        X = torch.cat([
            torch.randn(50, 10) + 10.0,
            torch.randn(50, 10) - 10.0,
        ])
        labels = ["A"] * 50 + ["B"] * 50

        result = evaluator.cluster_purity(X, labels, k=2)
        assert result["purity"] > 0.9, f"Expected high purity, got {result['purity']}"
        print(f"  Perfect clusters: purity={result['purity']:.2f}, ARI={result['adjusted_rand_index']:.2f}")

    def test_random_clusters(self, evaluator):
        """Random data should have low adjusted rand index."""
        torch.manual_seed(42)
        X = torch.randn(100, 10)
        labels = ["A"] * 50 + ["B"] * 50

        result = evaluator.cluster_purity(X, labels, k=2)
        # Purity can still be high by chance (≥0.5), but ARI should be low
        assert result["adjusted_rand_index"] < 0.5
        print(f"  Random: purity={result['purity']:.2f}, ARI={result['adjusted_rand_index']:.2f}")


# ---------------------------------------------------------------------------
# Test: t-SNE and PCA
# ---------------------------------------------------------------------------

class TestDimensionalityReduction:

    def test_tsne_output_shape(self, evaluator):
        """t-SNE returns correct output shape."""
        X = torch.randn(50, 10)
        labels = ["A"] * 25 + ["B"] * 25

        result = evaluator.tsne_separation(X, labels, max_samples=50)
        assert result["embedding"].shape == (50, 2)
        assert len(result["labels"]) == 50
        assert -1 <= result["silhouette_score"] <= 1
        print(f"  t-SNE silhouette: {result['silhouette_score']:.3f}")

    def test_pca_output_shape(self, evaluator):
        """PCA returns correct output shape."""
        X = torch.randn(50, 10)
        labels = ["A"] * 25 + ["B"] * 25

        result = evaluator.pca_separation(X, labels, n_components=2)
        assert result["embedding"].shape == (50, 2)
        assert len(result["explained_variance_ratio"]) == 2
        assert 0 <= result["linear_separability"] <= 1
        print(f"  PCA separability: {result['linear_separability']:.3f}")

    def test_tsne_subsampling(self, evaluator):
        """t-SNE subsamples when data is too large."""
        X = torch.randn(5000, 10)
        labels = ["A"] * 2500 + ["B"] * 2500

        result = evaluator.tsne_separation(X, labels, max_samples=100)
        assert result["embedding"].shape[0] == 100

    def test_separation_on_separable_data(self, evaluator):
        """t-SNE silhouette should be higher for separable data."""
        torch.manual_seed(42)

        # Separable
        X_sep = torch.cat([
            torch.randn(30, 10) + 5.0,
            torch.randn(30, 10) - 5.0,
        ])
        labels_sep = ["A"] * 30 + ["B"] * 30

        # Random
        X_rand = torch.randn(60, 10)
        labels_rand = ["A"] * 30 + ["B"] * 30

        result_sep = evaluator.tsne_separation(X_sep, labels_sep)
        result_rand = evaluator.tsne_separation(X_rand, labels_rand)

        # Separable data should have higher silhouette score
        assert result_sep["silhouette_score"] > result_rand["silhouette_score"]
        print(f"  Separable silhouette: {result_sep['silhouette_score']:.3f}")
        print(f"  Random silhouette: {result_rand['silhouette_score']:.3f}")


# ---------------------------------------------------------------------------
# Test: Cosine Similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:

    def test_identical_distributions(self, evaluator):
        """Identical distributions should have high cosine similarity."""
        torch.manual_seed(42)
        X = torch.randn(100, 64) + 1.0

        result = evaluator.cosine_similarity_divergence(X[:50], X[50:])
        assert result["centroid_cosine"] > 0.9
        print(f"  Identical dist cosine: {result['centroid_cosine']:.3f}")

    def test_orthogonal_distributions(self, evaluator):
        """Orthogonal distributions should have low cosine similarity."""
        dec = torch.zeros(50, 64)
        hon = torch.zeros(50, 64)

        dec[:, :32] = torch.randn(50, 32)  # Signal in first half
        hon[:, 32:] = torch.randn(50, 32)  # Signal in second half

        result = evaluator.cosine_similarity_divergence(dec, hon)
        assert result["centroid_cosine"] < 0.5
        print(f"  Orthogonal cosine: {result['centroid_cosine']:.3f}")


# ---------------------------------------------------------------------------
# Test: Full pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_run_full_evaluation(self, evaluator, tiny_sae, synthetic_activations):
        """Full evaluation pipeline runs without errors."""
        sae, config = tiny_sae

        acts_by_cat = synthetic_activations
        all_acts = torch.cat([
            acts_by_cat["deceptive"],
            acts_by_cat["honest"],
            acts_by_cat["contradiction"],
            acts_by_cat["neutral"],
        ])
        labels = (
            ["deceptive"] * 100 +
            ["honest"] * 100 +
            ["contradiction"] * 100 +
            ["neutral"] * 100
        )

        results = run_full_evaluation(
            sae=sae,
            activations_by_category=acts_by_cat,
            labels=labels,
            all_activations=all_acts,
        )

        assert "discriminability_dec_vs_hon" in results
        assert "linear_probe_binary" in results
        assert "cluster_purity_binary" in results
        assert "tsne" in results
        assert "pca" in results
        assert "cosine_divergence" in results

        print(f"  Binary probe acc: {results['linear_probe_binary']['accuracy']:.2f}")
        print(f"  Cluster purity: {results['cluster_purity_binary']['purity']:.2f}")
        print(f"  t-SNE silhouette: {results['tsne']['silhouette_score']:.3f}")

    def test_confound_check_available(self, evaluator, tiny_sae, synthetic_activations):
        """Contradiction confound check is included."""
        sae, config = tiny_sae

        results = run_full_evaluation(
            sae=sae,
            activations_by_category=synthetic_activations,
            labels=(["deceptive"] * 100 + ["honest"] * 100 +
                    ["contradiction"] * 100 + ["neutral"] * 100),
            all_activations=torch.cat(list(synthetic_activations.values())),
        )

        assert "discriminability_dec_vs_contra" in results
        print("  Confound check (dec vs contradiction) included")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
