import pytest

from ml.retrieval.hybrid import rrf


def test_weighted_rrf_zero_and_scale():
    left, right = [("a", 1), ("b", 0)], [("b", 99), ("c", 0)]
    assert rrf([left, right], weights=[1, 0]) == rrf([left])
    assert rrf([left, right], weights=[0, 1]) == rrf([right])
    assert rrf([left, right], weights=[1, 3])[0][0] == "b"
    for weights in [[1], [-1, 1], [float("nan"), 1], [float("inf"), 1]]:
        with pytest.raises(ValueError):
            rrf([left, right], weights=weights)


@pytest.mark.parametrize("weight", [0, 0.5, 1])
def test_optimized_blend_tail_and_prefix(engine, weight):
    engine.optimized_profile = {
        "candidates": 4,
        "rerank_depth": 2,
        "constant": 10,
        "lexical_weight": 2,
        "title_weight": 0.5,
        "cross_encoder_weight": weight,
        "blend_constant": 10,
    }
    # Verify ranking against independent composition, including preserved tail.
    lexical = engine.bm25.search("vitamin", 4)
    dense = engine.store.search([1, 0], 4)
    from ml.retrieval.bm25 import BM25

    titles = BM25(list(engine.papers.values()), "title").search("vitamin", 4)
    fused = rrf([lexical, dense, titles], 10, 4, [2, 1, 0.5])
    ce = engine.get_reranker().rank("vitamin", fused[:2], engine.papers, "both")
    expected = rrf([fused[:2], ce], 10, 2, [1 - weight, weight]) + fused[2:]
    actual, _, _ = engine.search("vitamin", "optimized", top_k=4)
    assert [pid for pid, _ in actual] == [pid for pid, _ in expected]
    assert len(set(pid for pid, _ in actual)) == 4
    assert engine.search("vitamin", "optimized", top_k=1)[0] == actual[:1]
    assert engine.search("vitamin", "optimized", field="missing")[0] == []


def test_profile_rejects_unmatched_index(engine):
    with pytest.raises((ValueError, FileNotFoundError)):
        engine.get_optimized_profile()


def test_api_optimized_profile_error(engine):
    from fastapi.testclient import TestClient
    from backend.main import create_app

    with TestClient(create_app(engine)) as client:
        assert not client.get("/api/health").json()["optimized_available"]
        response = client.post("/api/search", json={"query": "vitamin", "mode": "optimized"})
        assert response.status_code in {422, 503}


def test_api_does_not_replace_optimized_with_legacy_reranking(engine):
    from fastapi.testclient import TestClient
    from backend.main import create_app

    engine.optimized_profile = {
        "candidates": 4,
        "rerank_depth": 2,
        "constant": 5,
        "lexical_weight": 1,
        "title_weight": 0,
        "cross_encoder_weight": 0.75,
        "blend_constant": 10,
    }
    with TestClient(create_app(engine)) as client:
        response = client.post(
            "/api/search",
            json={
                "query": "vitamin",
                "mode": "optimized",
                "reranking": True,
            },
        )
        assert response.status_code == 200
        assert response.json()["mode"] == "optimized"
        assert client.get("/api/health").json()["optimized_available"]
