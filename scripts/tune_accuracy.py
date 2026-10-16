"""Train-only selection. Test judgments are never loaded by this script."""

import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.datasets.loader import load_corpus, load_judgments
from ml.evaluation.metrics import metrics
from ml.pipelines.search_pipeline import SearchEngine
from ml.retrieval.bm25 import BM25
from ml.retrieval.hybrid import rrf


def main():
    protocol = json.loads(Path("experiments/configs/accuracy-study.json").read_text())
    engine = SearchEngine("indexes/scifact")
    papers, stats = load_corpus(Path("data/scifact"))
    assert stats["fingerprint"] == engine.meta["stats"]["fingerprint"]
    queries, qrels = load_judgments(Path("data/scifact"), papers, "train")
    title = BM25(papers, "title")
    grid = [
        dict(constant=k, lexical_weight=w, title_weight=t)
        for k, w, t in itertools.product(
            protocol["fusion_grid"]["constant"],
            protocol["fusion_grid"]["lexical_weight"],
            protocol["fusion_grid"]["title_weight"],
        )
    ]
    scores = [[] for _ in grid]
    cached = {}
    vectors = engine.get_encoder().encode(list(queries.values()), progress=True)
    for n, ((qid, query), vector) in enumerate(zip(queries.items(), vectors), 1):
        rankings = [
            engine.bm25.search(query, 100),
            engine.store.search(vector, 100),
            title.search(query, 100),
        ]
        cached[qid] = rankings
        for i, cfg in enumerate(grid):
            ranking = rrf(rankings, cfg["constant"], 100, [cfg["lexical_weight"], 1, cfg["title_weight"]])
            scores[i].append(metrics(ranking, qrels[qid])["NDCG@10"])
        if n % 100 == 0:
            print(f"Fusion selection: {n}/{len(queries)}", flush=True)
    best = max(range(len(grid)), key=lambda i: np.mean(scores[i]))
    cfg = grid[best]
    print(f"Selected fusion: {cfg}; train nDCG={np.mean(scores[best]):.6f}", flush=True)
    selected_ids = sorted(random.Random(42).sample(sorted(queries), 200))
    blend_scores = {str(w): [] for w in protocol["blend_weights"]}
    cache_dir = Path(".cache/accuracy-study")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(
        json.dumps({"index": engine.meta, "config": cfg}, sort_keys=True).encode()
    ).hexdigest()
    for n, qid in enumerate(selected_ids, 1):
        fused = rrf(cached[qid], cfg["constant"], 100, [cfg["lexical_weight"], 1, cfg["title_weight"]])
        cache_path = cache_dir / f"{cache_key}-{qid}.json"
        if cache_path.exists():
            head = json.loads(cache_path.read_text())
        else:
            head = engine.get_reranker().rank(queries[qid], fused[:20], engine.papers, engine.representation)
            cache_path.write_text(json.dumps(head))
        for weight in protocol["blend_weights"]:
            ranking = rrf([fused[:20], head], 10, 20, [1 - weight, weight]) + fused[20:]
            blend_scores[str(weight)].append(metrics(ranking, qrels[qid])["NDCG@10"])
        if n % 10 == 0:
            print(f"Rerank selection: {n}/200", flush=True)
    weight = max(protocol["blend_weights"], key=lambda w: np.mean(blend_scores[str(w)]))
    profile = {
        **cfg,
        "cross_encoder_weight": weight,
        "blend_constant": 10,
        "rerank_depth": 20,
        "candidates": 100,
        "dataset": "scifact",
        "corpus_fingerprint": stats["fingerprint"],
        "model": engine.meta["model"],
        "revision": engine.meta["revision"],
        "reranker": engine.meta["reranker"],
        "reranker_revision": engine.meta["reranker_revision"],
        "representation": engine.representation,
        "selection_split": "train",
        "seed": 42,
    }
    output = Path("experiments/results/accuracy-development")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": protocol,
        "fusion_query_ids": list(queries),
        "rerank_query_ids": selected_ids,
        "fusion_trials": [dict(**c, ndcg=float(np.mean(s))) for c, s in zip(grid, scores)],
        "blend_trials": {w: float(np.mean(s)) for w, s in blend_scores.items()},
        "selected": profile,
    }
    (output / "selection.json").write_text(json.dumps(report, indent=2))
    Path("experiments/configs/optimized-scifact.json").write_text(json.dumps(profile, indent=2))
    print(json.dumps(profile, indent=2), flush=True)
    print("Configuration frozen. Test evaluation may now begin.", flush=True)


if __name__ == "__main__":
    main()
