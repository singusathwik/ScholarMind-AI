import numpy as np
import pytrec_eval
import pytest

from ml.evaluation.metrics import metrics
from ml.evaluation.statistical_tests import paired_bootstrap


def test_metrics_against_trec():
    rng = np.random.default_rng(42)
    for _ in range(20):
        ids = [str(i) for i in range(130)]
        judgments = {pid: int(rng.integers(0, 4)) for pid in ids[:30]}
        rng.shuffle(ids)
        ranking = [(pid, float(100 - i)) for i, pid in enumerate(ids[:100])]
        ours = metrics(ranking, judgments)
        ref = pytrec_eval.RelevanceEvaluator(
            {"q": judgments}, {"ndcg_cut.10", "recall.10,100", "map", "recip_rank", "P.10"}
        ).evaluate({"q": dict(ranking)})["q"]
        for a, b in [
            ("NDCG@10", "ndcg_cut_10"),
            ("Recall@10", "recall_10"),
            ("Recall@100", "recall_100"),
            ("MAP@100", "map"),
            ("MRR@100", "recip_rank"),
            ("Precision@10", "P_10"),
        ]:
            assert ours[a] == pytest.approx(ref[b])


def test_empty_and_perfect():
    assert metrics([], {"a": 1})["NDCG@10"] == 0
    assert metrics([("a", 1)], {"a": 1})["NDCG@10"] == 1
    assert metrics([("a", 1)], {"a": 1})["Precision@10"] == 0.1
    with pytest.raises(ValueError):
        metrics([("a", 1), ("a", 0)], {"a": 1})


def test_bootstrap():
    result = paired_bootstrap([0, 0, 0], [1, 1, 1], repetitions=100)
    assert result["ci95"] == [1.0, 1.0]
    assert result["mean_difference"] == 1
