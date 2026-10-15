"""Re-evaluate saved ranked IDs with the independent TREC evaluator."""

import argparse
import json
import sys
from pathlib import Path

import pytrec_eval

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.datasets.loader import load_corpus, load_judgments


def verify(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    papers, _ = load_corpus(Path("data") / result["dataset"])
    _, qrels = load_judgments(Path("data") / result["dataset"], papers, result["split"])
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {"ndcg_cut.10", "recall.10,100", "map", "recip_rank", "P.10"}
    )
    mapping = {
        "NDCG@10": "ndcg_cut_10",
        "Recall@10": "recall_10",
        "Recall@100": "recall_100",
        "MAP@100": "map",
        "MRR@100": "recip_rank",
        "Precision@10": "P_10",
    }
    checks = 0
    for mode, rankings in result["runs"].items():
        trec_run = {}
        for qid, ids in rankings.items():
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate document IDs in saved ranking")
            trec_run[qid] = {pid: float(len(ids) - rank) for rank, pid in enumerate(ids)}
        expected = evaluator.evaluate(trec_run)
        for row in result["per_query"][mode]:
            for metric, trec_metric in mapping.items():
                assert abs(row[metric] - expected[row["query_id"]][trec_metric]) < 1e-10
                checks += 1
        for metric in mapping:
            mean = sum(row[metric] for row in result["per_query"][mode]) / len(rankings)
            assert abs(mean - result["summary"][mode][metric]) < 1e-10
    print(f"{path}: {checks} per-query metric comparisons and all aggregate metrics agree with pytrec_eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    for path in parser.parse_args().paths:
        verify(path)
