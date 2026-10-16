"""Tune paper recommendation fusion on paper-disjoint training proxy labels only."""

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.datasets.loader import load_corpus, load_judgments
from ml.evaluation.related import co_evidence, disjoint_development, related_metrics
from ml.pipelines.search_pipeline import SearchEngine
from ml.retrieval.related import RelatedRecommender
from ml.retrieval.hybrid import rrf


def main():
    protocol = json.loads(Path("experiments/configs/related-study.json").read_text())
    engine = SearchEngine("indexes/scifact")
    papers, stats = load_corpus(Path("data/scifact"))
    assert engine.meta["stats"]["fingerprint"] == stats["fingerprint"]
    _, train = load_judgments(Path("data/scifact"), papers, "train")
    _, test = load_judgments(Path("data/scifact"), papers, "test")
    test = co_evidence(test)
    dev = disjoint_development(co_evidence(train), test)
    model = RelatedRecommender(engine.papers, engine.store)
    grid = [{"method": "dense"}] + [
        dict(method="hybrid", rrf_constant=k, lexical_weight=w)
        for k, w in itertools.product(protocol["grid"]["rrf_constant"], protocol["grid"]["lexical_weight"])
    ]
    rows = [[] for _ in grid]
    for pid in sorted(dev):
        dense, lexical = model.components(pid)
        for cfg, output in zip(grid, rows):
            ranking = (
                dense
                if cfg["method"] == "dense"
                else rrf([dense, lexical], cfg["rrf_constant"], 100, [1, cfg["lexical_weight"]])
            )
            output.append({"paper_id": pid, **related_metrics(ranking, dev[pid])})
    best = max(range(len(grid)), key=lambda i: np.mean([r["NDCG@5"] for r in rows[i]]))
    profile = {
        **grid[best],
        "dataset": "scifact",
        "corpus_fingerprint": stats["fingerprint"],
        "model": engine.meta["model"],
        "revision": engine.meta["revision"],
        "representation": engine.meta["representation"],
        "selection": "paper-disjoint training co-evidence proxy",
        "lexical": protocol["lexical"],
        "candidates": 100,
    }
    trials = [
        dict(
            configuration=cfg,
            summary={k: float(np.mean([r[k] for r in output])) for k in output[0] if k != "paper_id"},
        )
        for cfg, output in zip(grid, rows)
    ]
    out = Path("experiments/results/related-development")
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": protocol,
        "train_anchors_before_exclusion": len(co_evidence(train)),
        "development_anchors": len(dev),
        "test_anchors_reserved": len(test),
        "development_qrels": {p: sorted(v) for p, v in sorted(dev.items())},
        "reserved_test_papers": sorted(test),
        "trials": trials,
        "selected": profile,
        "label_checksums": {
            s: hashlib.sha256(Path(f"data/scifact/qrels/{s}.tsv").read_bytes()).hexdigest()
            for s in ["train", "test"]
        },
    }
    (out / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path("experiments/configs/related-scifact.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )
    print(json.dumps({"development_anchors": len(dev), "trials": trials, "selected": profile}, indent=2))
    print("Profile frozen. No test ranking metrics have been computed.")


if __name__ == "__main__":
    main()
