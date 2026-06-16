"""
tests/test_evaluation.py
=========================
Unit tests for the evaluation framework metrics.
Tests are purely functional — no DB or model loading required.
"""
import pytest
import numpy as np

from app.ml.evaluation import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    compute_query_metrics,
)


class TestPrecisionAtK:
    def test_perfect_retrieval(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b", "c", "d", "e"}
        assert precision_at_k(retrieved, relevant, 5) == 1.0

    def test_no_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = {"x", "y", "z"}
        assert precision_at_k(retrieved, relevant, 3) == 0.0

    def test_half_relevant(self):
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 4) == 0.5

    def test_k_larger_than_retrieved(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        # k=10 but only 2 retrieved
        assert precision_at_k(retrieved, relevant, 10) == pytest.approx(0.2, abs=1e-6)

    def test_k_zero(self):
        assert precision_at_k(["a"], {"a"}, 0) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_partial_recall(self):
        retrieved = ["a", "x", "y", "z", "b"]
        relevant = {"a", "b", "c", "d"}
        # Top-5 retrieves a,b out of 4 relevant
        assert recall_at_k(retrieved, relevant, 5) == 0.5

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], set(), 5) == 0.0


class TestReciprocalRank:
    def test_first_position(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert reciprocal_rank(retrieved, relevant) == 1.0

    def test_second_position(self):
        retrieved = ["x", "a", "b"]
        relevant = {"a"}
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)

    def test_not_found(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert reciprocal_rank(retrieved, relevant) == 0.0

    def test_multiple_relevant(self):
        retrieved = ["x", "a", "b"]
        relevant = {"a", "b"}
        # First relevant at position 2
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)


class TestNDCGAtK:
    def test_perfect_ranking(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)

    def test_no_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = {"x", "y", "z"}
        assert ndcg_at_k(retrieved, relevant, 3) == 0.0

    def test_reversed_ranking(self):
        """Worst-case ranking: relevant items at bottom."""
        retrieved = ["x", "y", "a", "b"]
        relevant = {"a", "b"}
        # DCG = 1/log2(4) + 1/log2(5)
        dcg = 1.0 / np.log2(4) + 1.0 / np.log2(5)
        # IDCG = 1/log2(2) + 1/log2(3)
        idcg = 1.0 / np.log2(2) + 1.0 / np.log2(3)
        expected = dcg / idcg
        assert ndcg_at_k(retrieved, relevant, 4) == pytest.approx(expected, abs=1e-4)

    def test_k_zero(self):
        assert ndcg_at_k(["a"], {"a"}, 0) == 0.0

    def test_empty_relevant(self):
        assert ndcg_at_k(["a", "b"], set(), 5) == 0.0


class TestComputeQueryMetrics:
    def test_all_metrics_keys(self):
        metrics = compute_query_metrics(["a", "b"], {"a"})
        assert set(metrics.keys()) == {
            "precision_at_5",
            "precision_at_10",
            "recall_at_10",
            "mrr",
            "ndcg_at_10",
        }

    def test_all_metrics_range(self):
        retrieved = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        relevant = {"a", "c", "e"}
        metrics = compute_query_metrics(retrieved, relevant)
        for v in metrics.values():
            assert 0.0 <= v <= 1.0, f"Metric out of range: {v}"

    def test_perfect_metrics(self):
        retrieved = ["a", "b", "c", "d", "e"] * 2
        relevant = set(retrieved[:10])
        metrics = compute_query_metrics(retrieved, relevant)
        assert metrics["precision_at_5"] == 1.0
        assert metrics["mrr"] == 1.0
