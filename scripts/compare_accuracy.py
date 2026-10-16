"""Evaluate a frozen train-selected profile against immutable original test results."""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.evaluation.benchmark import benchmark, plot
from ml.evaluation.statistical_tests import paired_bootstrap


def main():
    baseline_path = Path("experiments/results/scifact-test/results.json")
    profile_path = Path("experiments/configs/optimized-scifact.json")
    baseline_bytes, profile_bytes = baseline_path.read_bytes(), profile_path.read_bytes()
    original = json.loads(baseline_bytes)
    profile = json.loads(profile_bytes)
    assert profile["selection_split"] == "train"
    development = json.loads(Path("experiments/results/accuracy-development/selection.json").read_text())
    assert development["selected"] == profile
    test_ids = set(original["runs"]["rerank"])
    assert not test_ids.intersection(development["fusion_query_ids"])
    assert not test_ids.intersection(development["rerank_query_ids"])
    output = Path("experiments/results/scifact-optimized-test")
    result = benchmark(modes=["optimized"], output=".cache/accuracy-test")
    assert baseline_path.read_bytes() == baseline_bytes
    assert profile_path.read_bytes() == profile_bytes
    assert result["index"] == original["index"]
    assert result["query_count"] == original["query_count"] == 300
    result["configuration"] = profile
    result["protocol"] = (
        "Frozen train-selected profile; same original 300 test queries; no test-driven tuning"
    )
    result["baseline_sha256"] = hashlib.sha256(baseline_bytes).hexdigest()
    result["profile_sha256"] = hashlib.sha256(profile_bytes).hexdigest()
    result["baseline_timestamp"] = original["timestamp"]
    result["latency_note"] = (
        "Baseline latency is historical; optimized latency is a new warmed CPU pass, not a paired speed comparison."
    )
    result["scope"] = "Query-to-paper retrieval; no labeled evaluation of related-paper recommendations."
    result["comparisons"].update(original["comparisons"])
    optimized = {r["query_id"]: r for r in result["per_query"]["optimized"]}
    for mode in original["summary"]:
        rows = original["per_query"][mode]
        assert {r["query_id"] for r in rows} == set(optimized)
        for field in ["summary", "runs", "per_query"]:
            result[field][mode] = original[field][mode]
        result["comparisons"][f"optimized minus {mode}"] = paired_bootstrap(
            [r["NDCG@10"] for r in rows], [optimized[r["query_id"]]["NDCG@10"] for r in rows]
        )
    result["primary_comparison"] = "optimized minus rerank"
    base_rows = original["per_query"]["rerank"]
    deltas = [optimized[r["query_id"]]["NDCG@10"] - r["NDCG@10"] for r in base_rows]
    result["query_outcomes_vs_rerank"] = {
        "improved": sum(d > 1e-12 for d in deltas),
        "unchanged": sum(abs(d) <= 1e-12 for d in deltas),
        "worse": sum(d < -1e-12 for d in deltas),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Rewrite tables to include the historical baselines with explicit provenance in JSON.
    import csv

    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", *next(iter(result["summary"].values()))])
        writer.writeheader()
        writer.writerows(dict(mode=m, **v) for m, v in result["summary"].items())
    with (output / "per_query.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", *next(iter(result["per_query"].values()))[0]])
        writer.writeheader()
        writer.writerows(dict(mode=m, **r) for m, rows in result["per_query"].items() for r in rows)
    plot(
        {m: result["summary"][m] for m in ["bm25", "dense", "hybrid", "rerank", "optimized"]},
        output,
        latency_label="P95 ms (separate CPU runs)",
    )
    write_report(result, development)
    print(json.dumps({"summary": result["summary"], "comparisons": result["comparisons"]}, indent=2))


def write_report(result, development):
    metrics = ["NDCG@10", "Recall@10", "Recall@100", "MAP@100", "MRR@100", "Precision@10"]
    lines = [
        "# Accuracy comparison: frozen training-selected ranking",
        "",
        "Evaluated on the same 300 SciFact test queries and 5,183 documents as the original benchmark.",
        "This measures query-to-paper retrieval, not classification accuracy or related-paper relevance.",
        "",
        "| Method | " + " | ".join(metrics) + " |",
        "|---|" + "---:|" * len(metrics),
    ]
    for mode in ["bm25", "dense", "hybrid", "rerank", "optimized"]:
        row = result["summary"][mode]
        lines.append(f"| {mode} | " + " | ".join(f"{row[m]:.4f}" for m in metrics) + " |")
    primary = result["comparisons"]["optimized minus rerank"]
    baseline = result["summary"]["rerank"]["NDCG@10"]
    lo, hi = primary["ci95"]
    lines += [
        "",
        "## Primary comparison",
        "",
        f"Optimized minus original rerank nDCG@10: {primary['mean_difference']:+.4f} "
        f"({100 * primary['mean_difference'] / baseline:+.2f}% relative). "
        f"95% paired bootstrap interval [{lo:+.4f}, {hi:+.4f}], 10,000 resamples, seed 42.",
        "The interval is entirely positive on this test set."
        if lo > 0
        else "The interval does not establish a positive improvement on this test set.",
        f"Per-query outcomes versus rerank: {result['query_outcomes_vs_rerank']}.",
        "",
        "## Selection and reproducibility",
        "",
        "The search space was declared in `experiments/configs/accuracy-study.json` before test evaluation.",
        "24 fusion configurations were compared using all 809 training queries. Five cross-encoder",
        "blend weights were then compared on 200 seeded training queries. Both stages maximize mean nDCG@10.",
        "These are training selection scores, not independent validation estimates.",
        "The selected profile was frozen before test judgments were loaded; training and test IDs are disjoint.",
        "No model weights, embeddings or corpus records were changed. The original baseline artifacts remain intact.",
        "",
        "Selected profile:",
        "",
        "```json",
        json.dumps(result["configuration"], indent=2),
        "```",
        "",
        "Training blend nDCG@10 by cross-encoder weight:",
        "",
        "```json",
        json.dumps(development["blend_trials"], indent=2),
        "```",
        "",
        "From the repository root:",
        "",
        "```powershell",
        ".venv\\Scripts\\python scripts/tune_accuracy.py",
        ".venv\\Scripts\\python scripts/compare_accuracy.py",
        ".venv\\Scripts\\python scripts/verify_results.py experiments/results/scifact-optimized-test/results.json",
        "```",
        "",
        "## Latency and limits",
        "",
        result["latency_note"],
        f"Optimized P50: {result['summary']['optimized']['P50_ms']:.1f} ms; "
        f"P95: {result['summary']['optimized']['P95_ms']:.1f} ms.",
        "The first model load is excluded. Optimized ranking reranks a fixed head of 20 papers when its",
        "selected cross-encoder weight is positive; requesting more results preserves that head and adds the tail.",
        "Ranking scores are reciprocal output ranks, not probabilities.",
        "",
        "SciFact test results were previously reported. This study uses that same benchmark for a comparable",
        "final check, not a fresh external dataset. No test-driven parameter changes were made in this study.",
        "The primary comparison was declared in advance; other comparisons are exploratory and unadjusted",
        "for multiple testing. Unjudged documents count as nonrelevant. Generalization to other corpora,",
        "personalized recommendations, and the related-paper panel has not been established.",
        "",
        f"Baseline SHA-256: `{result['baseline_sha256']}`.",
        f"Frozen profile SHA-256: `{result['profile_sha256']}`.",
    ]
    Path("docs/ACCURACY_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
