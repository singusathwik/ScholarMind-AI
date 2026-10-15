from fastapi.testclient import TestClient
import pytest

from backend.main import create_app


@pytest.fixture
def client(engine):
    with TestClient(create_app(engine)) as client:
        yield client


def test_health_and_details(client):
    assert client.get("/api/health").json()["index_ready"]
    assert client.get("/api/models").status_code == 200
    assert client.get("/api/papers/a").json()["title"] == "Cancer immune therapy"
    assert client.get("/api/papers/unknown").status_code == 404
    result = client.get("/api/papers/a/recommendations").json()["results"]
    assert all(p["paper_id"] != "a" for p in result)
    assert client.get("/api/papers/a/recommendations?top_k=0").status_code == 422


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "rerank"])
def test_search(client, mode):
    response = client.post("/api/search", json={"query": "vitamin", "mode": mode, "top_k": 2})
    assert response.status_code == 200
    assert response.json()["returned_count"] == 2
    assert response.json()["mode"] == mode


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x", "top_k": 0},
        {"query": "x", "mode": "fake"},
        {"query": "x" * 1001},
    ],
)
def test_validation(client, payload):
    assert client.post("/api/search", json=payload).status_code == 422


def test_missing_index():
    app = create_app()
    client = TestClient(app)
    assert not client.get("/api/health").json()["index_ready"]
    assert client.post("/api/search", json={"query": "test"}).status_code == 503
