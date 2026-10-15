import numpy as np
import pytest

from ml.retrieval.bm25 import BM25
from ml.retrieval.faiss_store import VectorStore
from ml.retrieval.hybrid import rrf
from ml.retrieval.reranker import Reranker


def test_bm25(papers):
    model = BM25(papers)
    assert model.search("vitamin", 2)[0][0] == "b"
    assert model.search("", 3) == model.search("nonexistentterm", 3) == []
    assert model.search("vitamin vitamin", 2)[0][0] == "b"
    assert model.search("cells", 3) == model.search("cells", 3)
    scores = [s for _, s in model.search("vitamin", 4)]
    assert scores == sorted(scores, reverse=True)


def test_rrf():
    ranking = rrf([[("a", 900), ("b", 0.1)], [("b", 0.9), ("c", 0.5)]])
    assert ranking[0][0] == "b"
    assert ranking[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert len(rrf([[("a", 1), ("a", 1)]])) == 1
    assert rrf([[("b", 1)], [("a", 1)]])[0][0] == "a"


def test_faiss_persistence(tmp_path):
    (tmp_path / "papers.json").write_text("[]")
    store = VectorStore(["a", "b", "c"], np.array([[1, 0], [1, 0], [0, 1]]))
    store.save(tmp_path, {"model": "unit"})
    loaded, _ = VectorStore.load(tmp_path, "unit")
    assert loaded.search([1, 0], 2) == [("a", 1.0), ("b", 1.0)]
    with pytest.raises(ValueError, match="mismatch"):
        VectorStore.load(tmp_path, "other")
    with pytest.raises(ValueError):
        loaded.search([1, 0, 0], 2)
    (tmp_path / "embeddings.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        VectorStore.load(tmp_path)


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "rerank"])
def test_pipeline(engine, mode):
    ranking, timing, _ = engine.search("vitamin", mode, top_k=2)
    assert len(ranking) == 2
    assert set(pid for pid, _ in ranking) <= set(engine.papers)
    assert timing["total_ms"] >= 0


def test_recommendations(engine):
    results = engine.recommend("a", 3)
    assert results[0][0] == "d"
    assert "a" not in [pid for pid, _ in results]
    assert len(set(pid for pid, _ in results)) == 3
    with pytest.raises(KeyError):
        engine.recommend("missing")


def test_filters_and_blank(engine):
    assert engine.search("bone", year=2020)[0] == []
    assert engine.search("bone", field="biology")[0] == []
    with pytest.raises(ValueError):
        engine.search(" ")


def test_reranker_identity(papers):
    class Model:
        def predict(self, pairs, **kwargs):
            assert pairs[0][0] == "query"
            return [0.1, 0.8]

    reranker = Reranker.__new__(Reranker)
    reranker.model = Model()
    assert reranker.rank("query", [("a", 10), ("b", 2)], {p.paper_id: p for p in papers}) == [
        ("b", 0.8),
        ("a", 0.1),
    ]
