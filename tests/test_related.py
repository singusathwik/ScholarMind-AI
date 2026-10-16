import math

import numpy as np
import pytest

from ml.evaluation.related import (
    co_evidence,
    components,
    disjoint_development,
    grouped_bootstrap,
    related_metrics,
)
from ml.retrieval.related import RelatedRecommender


def test_proxy_split_removes_both_ends_and_deduplicates():
    train = co_evidence({"q1": {"a": 1, "b": 1, "c": 0}, "q2": {"a": 1, "b": 1}, "q3": {"c": 1, "d": 1}})
    test = co_evidence({"q4": {"a": 1, "e": 1}})
    assert train["a"] == {"b"}
    assert disjoint_development(train, test) == {"c": {"d"}, "d": {"c"}}
    assert co_evidence({"single": {"a": 1}}) == {}


def test_components_preserve_transitive_dependence():
    labels = components({"a": {"b"}, "b": {"c"}, "x": {"y"}})
    assert labels["a"] == labels["b"] == labels["c"]
    assert labels["x"] == labels["y"] != labels["a"]


def test_metrics_penalize_unretrieved_positives():
    result = related_metrics([("a", 2), ("x", 1)], {"a", "b"})
    assert result["Recall@5"] == 0.5
    assert result["Precision@5"] == 0.2
    assert result["NDCG@5"] == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert result["MAP@100"] == 0.5
    with pytest.raises(ValueError):
        related_metrics([("a", 2), ("a", 1)], {"a"})


def test_cluster_bootstrap_weighting_and_constant_gain():
    result = grouped_bootstrap([0, 0, 0], [0.2, 0.2, 0.2], ["a", "a", "b"], [1, 3, 2])
    assert result["mean_difference"] == pytest.approx(0.2)
    assert result["ci95"] == pytest.approx([0.2, 0.2])
    assert result["groups"] == 2
    with pytest.raises(ValueError):
        grouped_bootstrap([0], [1], ["a"])


def test_related_self_exclusion_prefix_and_fallback(engine):
    legacy = engine.recommend("a", 3, "dense")
    assert engine.recommend("a", 3) == legacy
    with pytest.raises(ValueError):
        engine.recommend("a", 3, "hybrid")
    engine.related_profile = {"method": "hybrid", "rrf_constant": 5, "lexical_weight": 0.5}
    hybrid = engine.recommend("a", 3)
    assert "a" not in [p for p, _ in hybrid]
    assert len({p for p, _ in hybrid}) == 3
    assert engine.recommend("a", 1) == hybrid[:1]
    with pytest.raises(ValueError):
        engine.recommend("a", 0)
    with pytest.raises(ValueError):
        engine.recommend("a", method="invalid")


def test_complete_abstract_lexical_signal_and_candidate_restriction(engine):
    model = RelatedRecommender(engine.papers, engine.store)
    _, lexical = model.components("a", candidates=["a", "b", "d"])
    assert lexical[0][0] == "d"
    assert all(p != "a" for p, _ in lexical)
    with pytest.raises(ValueError):
        model.components("a", candidates=["b", "b"])
    with pytest.raises(ValueError):
        model.components("a", candidates=["missing"])
    assert np.isfinite([s for _, s in lexical]).all()


def test_recommendation_api_labels_match_method(engine):
    from backend.main import create_app
    from fastapi.testclient import TestClient

    engine.related_profile = {"method": "hybrid", "rrf_constant": 5, "lexical_weight": 0.5}
    with TestClient(create_app(engine)) as client:
        result = client.get("/api/papers/a/recommendations").json()
        assert result["method"] == "hybrid"
        assert all(r["score_type"] == "related rank fusion" for r in result["results"])
        original = client.get("/api/papers/a/recommendations?method=dense").json()
        assert original["method"] == "dense"
        assert all(r["score_type"] == "cosine similarity" for r in original["results"])
        assert client.get("/api/papers/a/recommendations?method=bad").status_code == 422
        assert client.get("/api/papers/missing/recommendations").status_code == 404


def test_official_click_midrank_ties():
    from scripts.benchmark_related import click_metrics

    result = click_metrics({"clicked": 1.0, "tie": 1.0, "low": 0.0}, "clicked")
    assert result["NDCG@10"] == pytest.approx(1 / math.log2(2.5))
    assert result["P@1"] == 0
    assert result["MRR"] == pytest.approx(1 / 1.5)
