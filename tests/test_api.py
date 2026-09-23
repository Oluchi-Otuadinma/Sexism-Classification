"""API endpoint tests with a mocked HuggingFace client."""
import pytest
from fastapi.testclient import TestClient

from src.api import fastapi_main


class MockClient:
    """Stands in for HuggingFaceClient."""

    def __init__(self):
        self.calls = []

    def predict(self, text):
        self.calls.append(text)
        if text == "boom":
            return {"error": "inference failed"}
        return {"label": "sexist", "confidence": 0.93, "cached": False}

    def get_cache_stats(self):
        return {"hits": 0, "misses": 0, "size": 0}

    def clear_cache(self):
        return None


@pytest.fixture()
def client(monkeypatch):
    mock = MockClient()
    monkeypatch.setattr(fastapi_main, "get_client", lambda: mock)
    with TestClient(fastapi_main.app) as c:
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
