"""Evaluate a frozen related-paper profile; local proxy and external clicks stay separate."""

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.datasets.loader import Paper, load_corpus, load_judgments
from ml.evaluation.related import co_evidence, components, grouped_bootstrap, related_metrics
from ml.pipelines.search_pipeline import SearchEngine
from ml.retrieval.faiss_store import VectorStore
from ml.retrieval.hybrid import rrf
from ml.retrieval.related import RelatedRecommender


def frozen():
    path = Path("experiments/configs/related-scifact.json")
    raw = path.read_bytes()
    profile = json.loads(raw)
    development = json.loads(Path("experiments/results/related-development/selection.json").read_text())
    assert profile == development["selected"]
    return profile, development, hashlib.sha256(raw).hexdigest()


def save(result, name):
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    result["executed"] = True
    result["test_tuning"] = False
    folder = Path("experiments/results") / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (folder / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", *next(iter(result["summary"].values()))])
        writer.writeheader()
        writer.writerows(dict(method=m, **r) for m, r in result["summary"].items())
    print(json.dumps({k: result[k] for k in ["task", "summary", "comparison"]}, indent=2), flush=True)


def local():
    profile, development, digest = frozen()
    engine = SearchEngine("indexes/scifact")
    papers, stats = load_corpus(Path("data/scifact"))
    assert stats["fingerprint"] == profile["corpus_fingerprint"]
    _, qrels = load_judgments(Path("data/scifact"), papers, "test")
    labels = co_evidence(qrels)
    assert not set(labels).intersection(development["development_qrels"])
    assert sorted(labels) == development["reserved_test_papers"]
    group = components(labels)
    model = RelatedRecommender(engine.papers, engine.store)
    rows = {"dense": [], "hybrid": []}
    runs = {"dense": {}, "hybrid": {}}
    for pid in sorted(labels):
        for method in rows:
            ranking = model.rank(pid, 100, profile if method == "hybrid" else None)
            assert pid not in [p for p, _ in ranking]
            runs[method][pid] = [p for p, _ in ranking]
            rows[method].append(
                {"paper_id": pid, "component": group[pid], **related_metrics(ranking, labels[pid])}
            )
    summary = {
        m: {
            k: float(np.mean([r[k] for r in records]))
            for k in records[0]
            if k not in {"paper_id", "component"}
        }
        for m, records in rows.items()
    }
    comparison = grouped_bootstrap(
        [r["NDCG@5"] for r in rows["dense"]],
        [r["NDCG@5"] for r in rows["hybrid"]],
        [r["component"] for r in rows["dense"]],
    )
    deltas = [b["NDCG@5"] - a["NDCG@5"] for a, b in zip(rows["dense"], rows["hybrid"])]
    save(
        {
            "task": "SciFact co-evidence proxy, full-corpus paper retrieval",
            "dataset": "scifact",
            "label_type": "proxy: papers judged relevant to the same claim",
            "split": "test",
            "anchors": len(labels),
            "documents": len(papers),
            "development_anchors": development["development_anchors"],
            "positive_directed_pairs": sum(map(len, labels.values())),
            "configuration": profile,
            "profile_sha256": digest,
            "summary": summary,
            "comparison": comparison,
            "per_query": rows,
            "runs": runs,
            "qrels": {p: {n: 1 for n in sorted(v)} for p, v in sorted(labels.items())},
            "outcomes": {
                "improved": sum(d > 1e-12 for d in deltas),
                "worse": sum(d < -1e-12 for d in deltas),
                "unchanged": sum(abs(d) <= 1e-12 for d in deltas),
            },
            "limits": "Small co-evidence proxy; unjudged papers counted nonrelevant, not proven irrelevant. No user relevance ratings. Cluster bootstrap preserves linked-paper dependence.",
        },
        "related-scifact-test",
    )


def click_metrics(scores, clicked):
    positive = scores[clicked]
    above = sum(s > positive for p, s in scores.items() if p != clicked)
    ties = sum(s == positive for p, s in scores.items() if p != clicked)
    rank = 1 + above + ties / 2
    return {
        "NDCG@10": 1 / math.log2(rank + 1),
        "P@1": float(rank == 1),
        "MRR": 1 / rank,
        "Pairwise accuracy": (len(scores) - rank) / (len(scores) - 1),
    }


def external():
    profile, _, digest = frozen()
    folder = Path("data/scidocs-recomm")
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["model"] == profile["model"] and manifest["revision"] == profile["revision"]
    assert (
        hashlib.sha256((folder / "embeddings.npy").read_bytes()).hexdigest() == manifest["embedding_sha256"]
    )
    for name in ["metadata.json", "test.csv", "propensity_scores.json"]:
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == manifest[name]["sha256"]
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    ids = manifest["ids"]
    papers = {
        p: Paper(p, metadata[p].get("title") or "", metadata[p].get("abstract") or "", "scidocs") for p in ids
    }
    model = RelatedRecommender(
        papers, VectorStore(ids, np.load(folder / "embeddings.npy", allow_pickle=False))
    )
    model.fit_lexical()
    with (folder / "test.csv").open(encoding="utf-8", newline="") as f:
        events = list(csv.DictReader(f))
    propensity = json.loads((folder / "propensity_scores.json").read_text())["scores"]
    rows = {"dense": [], "hybrid": []}
    runs = {"dense": {}, "hybrid": {}}
    for i, event in enumerate(events):
        candidates = event["similar_papers_avail_at_time"].split(",")
        assert len(candidates) == len(set(candidates)) == 10
        assert event["pid"] not in candidates and event["clicked_pid"] in candidates
        dense, lexical = model.components(event["pid"], candidates=candidates)
        hybrid = (
            rrf([dense, lexical], profile["rrf_constant"], 10, [1, profile["lexical_weight"]])
            if profile["method"] != "dense"
            else dense
        )
        event_id = str(i)
        for method, ranking in [("dense", dense), ("hybrid", hybrid)]:
            scores = dict(ranking)
            runs[method][event_id] = scores
            rows[method].append(
                {
                    "event_id": event_id,
                    "paper_id": event["pid"],
                    "clicked_pid": event["clicked_pid"],
                    "display_position": candidates.index(event["clicked_pid"]) + 1,
                    "weight": 1 / propensity[candidates.index(event["clicked_pid"])],
                    **click_metrics(scores, event["clicked_pid"]),
                }
            )
        if (i + 1) % 100 == 0:
            print(f"External click evaluation: {i + 1}/{len(events)}", flush=True)
    summary = {}
    for method, records in rows.items():
        summary[method] = {}
        for metric in ["NDCG@10", "P@1", "MRR", "Pairwise accuracy"]:
            values = [r[metric] for r in records]
            summary[method][metric] = float(np.mean(values))
            summary[method]["Adj-" + metric] = float(
                np.average(values, weights=[r["weight"] for r in records])
            )
    comparison = grouped_bootstrap(
        [r["NDCG@10"] for r in rows["dense"]],
        [r["NDCG@10"] for r in rows["hybrid"]],
        [r["paper_id"] for r in rows["dense"]],
        [r["weight"] for r in rows["dense"]],
    )
    save(
        {
            "task": "SciDocs recommendation test: rerank ten logged candidates per click event",
            "label_type": "observed clicks, adjusted for display-position propensity",
            "split": "test",
            "events": len(events),
            "seed_papers": len({r["pid"] for r in events}),
            "documents": len(ids),
            "configuration": profile,
            "profile_sha256": digest,
            "summary": summary,
            "comparison": comparison,
            "comparison_metric": "Adj-NDCG@10",
            "per_query": rows,
            "runs": runs,
            "source": {k: v for k, v in manifest.items() if k != "ids"},
            "lexical_fit_scope": "All unlabeled text in the 8758-document union of test seed/candidate papers; no click labels used for fitting",
            "tie_policy": "Half credit for score ties, matching official SciDocs click evaluator",
            "limits": "Conditional ranking among ten preselected displayed papers, not full-corpus retrieval. Clicks reflect exposure and historical-system bias. No official learned SciDocs recommender model was trained; these are raw content-ranker results using its metric formula.",
        },
        "related-scidocs-test",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--external", action="store_true")
    external() if parser.parse_args().external else local()
