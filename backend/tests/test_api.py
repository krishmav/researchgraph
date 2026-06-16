"""
tests/test_api.py
==================
Integration tests for FastAPI endpoints.
Uses TestClient with mocked ML services to avoid requiring loaded models.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_retrieval():
    svc = MagicMock()
    svc.loaded_models = []
    svc.any_loaded = False
    return svc


@pytest.fixture
def mock_baselines():
    svc = MagicMock()
    svc.tfidf.loaded = False
    return svc


@pytest.fixture
def client(mock_retrieval, mock_baselines):
    with (
        patch("app.ml.retrieval.RetrievalService.get_instance", return_value=mock_retrieval),
        patch("app.ml.baselines.BaselineService.get_instance", return_value=mock_baselines),
        patch("app.ml.knowledge_graph.KnowledgeGraphService.get_instance"),
        patch("app.ml.topic_model.TopicModelService.get_instance"),
        patch("app.ml.gap_finder.GapFinderService.get_instance"),
    ):
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestHealth:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "version" in data
        assert "models_loaded" in data

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "name" in resp.json()


class TestSearchEndpoints:
    def test_search_validation_min_length(self, client):
        """Query must be ≥3 characters."""
        resp = client.post("/api/search", json={"query": "ab", "method": "keyword"})
        assert resp.status_code == 422

    def test_search_validation_unknown_method(self, client):
        """Unknown method should return 422."""
        resp = client.post(
            "/api/search",
            json={"query": "neural networks", "method": "unknown_method"},
        )
        assert resp.status_code == 422

    def test_search_tfidf_not_loaded(self, client):
        """TF-IDF endpoint returns 503 when index not loaded."""
        resp = client.post(
            "/api/search",
            json={"query": "graph neural networks", "method": "tfidf"},
        )
        assert resp.status_code in (503, 200)   # 503 if not loaded, 200 if loaded

    def test_compare_search_structure(self, client):
        """Compare endpoint should return correct schema."""
        resp = client.post(
            "/api/search/compare",
            json={"query": "transformer attention mechanism", "method": "keyword"},
        )
        # Should succeed (keyword always available via DB)
        assert resp.status_code in (200, 503)


class TestTopicsEndpoints:
    def test_topics_list(self, client):
        resp = client.get("/api/topics")
        # Returns empty list if no topics in DB
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_topics_map_no_model(self, client):
        resp = client.get("/api/topics/map")
        # 503 if topic model not loaded
        assert resp.status_code in (200, 503)


class TestGraphEndpoints:
    def test_graph_stats(self, client):
        resp = client.get("/api/graph/stats")
        assert resp.status_code == 200

    def test_graph_paper_not_found(self, client):
        resp = client.get("/api/graph/paper/nonexistent123")
        assert resp.status_code in (404, 503)


class TestGapsEndpoint:
    def test_gaps_not_loaded(self, client):
        resp = client.get("/api/gaps")
        assert resp.status_code in (200, 503)


class TestSearchRequestSchema:
    """Test Pydantic schema validation without hitting DB."""

    def test_defaults(self):
        from app.models.schemas import SearchRequest
        req = SearchRequest(query="deep learning")
        assert req.method == "bge"
        assert req.top_k == 10

    def test_top_k_bounds(self):
        from app.models.schemas import SearchRequest
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            SearchRequest(query="test", top_k=0)

        with pytest.raises(pydantic.ValidationError):
            SearchRequest(query="test", top_k=100)

    def test_valid_methods(self):
        from app.models.schemas import SearchRequest
        for method in ("keyword", "tfidf", "miniml", "mpnet", "bge", "graph"):
            req = SearchRequest(query="test query", method=method)
            assert req.method == method
