import csv
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ml.datasets.loader import load_corpus, load_judgments
from ml.evaluation.metrics import metrics
from ml.evaluation.statistical_tests import paired_bootstrap
from ml.pipelines.search_pipeline import SearchEngine


def benchmark(
    dataset="scifact",
    split="test",
    limit=None,
    index=None,
    output=None,
    modes=None,
    rerank_depth=20,
    constant=60,
    candidates=100,
):
    engine = SearchEngine(index or f"indexes/{dataset}")
    papers, stats = load_corpus(Path("data") / dataset)
    if stats["fingerprint"] != engine.meta["stats"]["fingerprint"]:
        raise ValueError("Benchmark corpus differs from index")
    queries, qrels = load_judgments(Path("data") / dataset, papers, split)
    if limit:
        queries = dict(list(queries.items())[:limit])
    modes = modes or ["bm25", "dense", "hybrid", "rerank"]
    summary, per_query, runs = {}, {}, {}
    for mode in modes:
        engine.search("immune cell response", mode, 100, candidates, rerank_depth, constant)
        rows, run = [], {}
        for n, (qid, query) in enumerate(queries.items(), 1):
            ranking, timings, _ = engine.search(query, mode, 100, candidates, rerank_depth, constant)
            rows.append(dict(query_id=qid, **metrics(ranking, qrels[qid]), **timings))
            run[qid] = [pid for pid, _ in ranking]
            if n % 25 == 0:
                print(f"{mode}: {n}/{len(queries)}", flush=True)
        per_query[mode], runs[mode] = rows, run
        summary[mode] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key != "query_id"}
        summary[mode].update(
            P50_ms=float(np.percentile([r["total_ms"] for r in rows], 50)),
            P95_ms=float(np.percentile([r["total_ms"] for r in rows], 95)),
        )
    comparisons = {}
    for a, b in [("bm25", "dense"), ("bm25", "hybrid"), ("dense", "hybrid"), ("hybrid", "rerank")]:
        if a in per_query and b in per_query:
            comparisons[f"{b} minus {a}"] = paired_bootstrap(
                [r["NDCG@10"] for r in per_query[a]], [r["NDCG@10"] for r in per_query[b]]
            )
    result = {
        "executed": True,
        "dataset": dataset,
        "split": split,
        "query_count": len(queries),
        "partial": limit is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "No test tuning; depth 100; rerank head then preserve tail; unjudged nonrelevant",
        "configuration": {
            "candidates": candidates,
            "rrf_constant": constant,
            "rerank_depth": rerank_depth,
            "seed": 42,
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "device": "cpu",
            "torch_threads": 4,
        },
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in ["torch", "sentence-transformers", "faiss-cpu", "numpy", "rank-bm25"]
        },
        "index": engine.meta,
        "summary": summary,
        "comparisons": comparisons,
        "per_query": per_query,
        "runs": runs,
    }
    folder = Path(output or f"experiments/results/{dataset}-{split}")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (folder / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", *next(iter(summary.values())).keys()])
        writer.writeheader()
        writer.writerows(dict(mode=mode, **row) for mode, row in summary.items())
    with (folder / "per_query.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", *next(iter(per_query.values()))[0].keys()])
        writer.writeheader()
        writer.writerows(dict(mode=m, **row) for m, rows in per_query.items() for row in rows)
    plot(summary, folder)
    return result


def plot(summary, folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, metric in zip(axes, ["NDCG@10", "P95_ms"]):
        ax.bar(list(summary), [r[metric] for r in summary.values()], color="#23786b")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
    fig.savefig(folder / "comparison.png", dpi=160)
    plt.close(fig)
