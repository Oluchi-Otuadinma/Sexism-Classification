"""API endpoint tests with a mocked model manager.

The model is normally loaded at startup by the FastAPI lifespan handler
(BERTweet via AutoModelForSequenceClassification); here PRELOAD_MODEL=false
(see conftest.py) keeps tests hermetic and each test mocks the manager's
predict method instead.
"""
import pytest
from fastapi.testclient import TestClient

from src.api import fastapi_main


class MockClient:
    """Stands in for ModelManager (duck-typed: predict/get_cache_stats/clear_cache)."""

    def __init__(self):
        self.calls = []

    def predict(self, text):
        self.calls.append(text)
        if text == "boom":
            return {"error": "inference failed"}
        return {"label": "sexist", "confidence": 0.93, "cached": False}

    def info(self):
        return {
            "backend": "local",
            "model_id": "vinai/bertweet-base",
            "loaded": False,
            "device": "cpu",
            "fresh_head": True,
            "labels": ["not sexist", "sexist"],
        }

    def get_cache_stats(self):
        return {"hits": 0, "misses": 0, "size": 0}

    def clear_cache(self):
        return None


@pytest.fixture()
def client(monkeypatch):
    mock = MockClient()
    with TestClient(fastapi_main.app) as c:
        # Replace the lifespan-created manager's methods with the mock
        monkeypatch.setattr(c.app.state.model_manager, "predict", mock.predict)
        monkeypatch.setattr(c.app.state.model_manager, "get_cache_stats", mock.get_cache_stats)
        monkeypatch.setattr(c.app.state.model_manager, "clear_cache", mock.clear_cache)
        yield c, mock


def test_health(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "healthy")


def test_predict_success(client):
    c, mock = client
    r = c.post("/predict", json={"text": "women belong in the kitchen"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "sexist"
    assert 0 <= body["confidence"] <= 1
    assert mock.calls == ["women belong in the kitchen"]


def test_predict_rejects_blank(client):
    c, _ = client
    r = c.post("/predict", json={"text": "   "})
    assert r.status_code == 422


def test_predict_rejects_too_long(client):
    c, _ = client
    r = c.post("/predict", json={"text": "x" * 5001})
    assert r.status_code == 422


def test_batch(client):
    c, mock = client
    r = c.post("/predict/batch", json={"texts": ["one", "two", "boom"]})
    assert r.status_code == 200
    body = r.json()
    assert body["total_count"] == 3
    assert body["success_count"] == 2
    assert body["error_count"] == 1
    assert body["predictions"][2]["label"] == "error"


def test_batch_rejects_empty_list(client):
    c, _ = client
    r = c.post("/predict/batch", json={"texts": []})
    assert r.status_code == 422


def test_cache_stats(client):
    c, _ = client
    r = c.get("/cache/stats")
    assert r.status_code == 200


def test_health_reports_model_info(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    model = r.json().get("model")
    assert model is not None
    assert model["backend"] == "local"
    assert model["model_id"] == "vinai/bertweet-base"
    assert model["loaded"] is False  # PRELOAD_MODEL=false in tests


def test_lifespan_creates_model_manager(client):
    """The lifespan handler must attach a ModelManager to app.state at startup."""
    c, _ = client
    from src.api.model_manager import ModelManager

    assert isinstance(c.app.state.model_manager, ModelManager)
