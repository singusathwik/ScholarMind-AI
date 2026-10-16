"""Independent saved-metric checks and serving/benchmark ranking parity."""

import csv
import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import pytrec_eval

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.pipelines.search_pipeline import SearchEngine


def main():
    local_path = Path("experiments/results/related-scifact-test/results.json")
    external_path = Path("experiments/results/related-scidocs-test/results.json")
    local, external = [json.loads(p.read_text()) for p in [local_path, external_path]]
    digest = hashlib.sha256(Path("experiments/configs/related-scifact.json").read_bytes()).hexdigest()
    assert local["profile_sha256"] == external["profile_sha256"] == digest
    mapping = {
        "NDCG@5": "ndcg_cut_5",
        "Recall@5": "recall_5",
        "Recall@100": "recall_100",
        "MAP@100": "map",
        "MRR@100": "recip_rank",
        "Precision@5": "P_5",
    }
    evaluator = pytrec_eval.RelevanceEvaluator(
        local["qrels"], {"ndcg_cut.5", "recall.5,100", "map", "recip_rank", "P.5"}
    )
    engine = SearchEngine("indexes/scifact")
    local_checks = 0
    for method, rankings in local["runs"].items():
        trec = evaluator.evaluate(
            {pid: {p: float(len(ids) - i) for i, p in enumerate(ids)} for pid, ids in rankings.items()}
        )
        for row in local["per_query"][method]:
            pid = row["paper_id"]
            assert pid not in rankings[pid]
            actual = engine.recommend(pid, 100, method)
            assert [p for p, _ in actual] == rankings[pid]
            assert engine.recommend(pid, 5, method) == actual[:5]
            for metric, reference in mapping.items():
                assert abs(row[metric] - trec[pid][reference]) < 1e-10
                local_checks += 1
            assert row["Hit@5"] == float(trec[pid]["recall_5"] > 0)
            local_checks += 1
        for metric in local["summary"][method]:
            assert (
                abs(
                    local["summary"][method][metric]
                    - sum(r[metric] for r in local["per_query"][method]) / len(rankings)
                )
                < 1e-10
            )
    with Path("data/scidocs-recomm/test.csv").open(encoding="utf-8", newline="") as f:
        events = list(csv.DictReader(f))
    propensity = json.loads(Path("data/scidocs-recomm/propensity_scores.json").read_text())["scores"]
    external_checks = 0
    for method, records in external["per_query"].items():
        for row in records:
            event = events[int(row["event_id"])]
            candidates = event["similar_papers_avail_at_time"].split(",")
            scores = external["runs"][method][row["event_id"]]
            assert set(scores) == set(candidates)
            assert row["paper_id"] == event["pid"] and row["clicked_pid"] == event["clicked_pid"]
            assert abs(row["weight"] - 1 / propensity[candidates.index(event["clicked_pid"])]) < 1e-12
            positive = scores[event["clicked_pid"]]
            negatives = [scores[p] for p in candidates if p != event["clicked_pid"]]
            # Official evaluator convention: tied alternatives each count half above/below.
            above = sum(1 if s > positive else 0.5 if s == positive else 0 for s in negatives)
            expected = {
                "NDCG@10": 1 / math.log2(2 + above),
                "P@1": float(above == 0),
                "MRR": 1 / (1 + above),
                "Pairwise accuracy": (9 - above) / 9,
            }
            for metric, value in expected.items():
                assert abs(row[metric] - value) < 1e-10
                external_checks += 1
        for metric in ["NDCG@10", "P@1", "MRR", "Pairwise accuracy"]:
            expected = sum(r[metric] for r in records) / len(records)
            weighted = sum(r[metric] * r["weight"] for r in records) / sum(r["weight"] for r in records)
            assert abs(external["summary"][method][metric] - expected) < 1e-10
            assert abs(external["summary"][method]["Adj-" + metric] - weighted) < 1e-10
    result = {
        "local_metric_checks": local_checks,
        "external_metric_checks": external_checks,
        "serving_parity_checks": len(local["qrels"]) * 2,
        "all_aggregate_metrics_verified": True,
        "profile_hash_verified": True,
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "scikit-learn", "scipy", "faiss-cpu", "pytrec-eval-terrier"]
        },
    }
    Path("experiments/results/related-verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
