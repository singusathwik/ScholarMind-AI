"""Render the saved recommendation evaluations without running or tuning a model."""

import json
from pathlib import Path


def main():
    local = json.loads(Path("experiments/results/related-scifact-test/results.json").read_text())
    external = json.loads(Path("experiments/results/related-scidocs-test/results.json").read_text())
    verification = json.loads(Path("experiments/results/related-verification.json").read_text())
    lines = [
        "# Related-paper recommendation evaluation",
        "",
        "The new semantic/lexical blend has higher point estimates in both tests below, but both",
        "95% cluster-bootstrap intervals include zero. A reliable accuracy improvement is not yet established.",
        "The original semantic method remains selectable in the paper dialog. These tests do not measure",
        "personalization or provide a calibrated probability that a recommendation is relevant.",
        "",
        "## Method and frozen selection",
        "",
        "The baseline is the original stored MiniLM title-plus-abstract cosine-neighbor ranking.",
        "The candidate adds TF-IDF cosine similarity over titles and complete abstracts: word unigrams",
        "and bigrams, English stop words, sublinear term frequency, L2 normalization, float32, and",
        "a 100,000-feature cap. Weighted RRF combines the two top-100 rankings with constant 5,",
        "semantic weight 1, and lexical weight 0.5. The selected paper is always excluded.",
        "No neural weights or search-ranking settings changed; no neural inference is needed at recommendation time.",
        "TF-IDF is fitted lazily once per server process over unlabeled corpus text.",
        "",
        "Thirteen candidates (the baseline plus twelve fusion settings) were compared on 94 training anchors.",
        "Positive links join papers annotated as evidence for the same SciFact claim. Test graph membership",
        "was used only to exclude all test papers from development, including as positive targets.",
        "The profile was frozen before either test ranking was scored. No test-driven retuning was performed.",
        "See `experiments/configs/related-study.json` and `experiments/results/related-development/selection.json`.",
        "",
        "## Local full-corpus proxy test",
        "",
        f"{local['anchors']} held-out anchor papers, {local['positive_directed_pairs']} directed positive links,",
        f"{local['comparison']['groups']} positive-graph components, and {local['documents']} corpus papers.",
        "These are co-evidence proxy labels, not direct human recommendation ratings. Unknown links are treated",
        "as nonrelevant for scoring, although many may be genuinely relevant. Most corpus papers have no",
        "usable co-evidence labels; this test does not establish accuracy across all 5,183 possible seed papers.",
        "",
        "| Metric | Original semantic | Recommended blend |",
        "|---|---:|---:|",
    ]
    for metric in local["summary"]["dense"]:
        lines.append(
            f"| {metric} | {local['summary']['dense'][metric]:.4f} | {local['summary']['hybrid'][metric]:.4f} |"
        )
    low, high = local["comparison"]["ci95"]
    lines += [
        "",
        f"nDCG@5 difference: {local['comparison']['mean_difference']:+.4f}; 95% interval [{low:+.4f}, {high:+.4f}].",
        f"Per-anchor nDCG outcomes: {local['outcomes']}.",
        "Hit@5 is the fraction of seed papers with at least one known positive among five suggestions;",
        "it is not the percentage of suggestions judged correct. Recall@5 uses all known positives as denominator.",
        "",
        "## External logged-click test",
        "",
        f"All {external['events']} official test events from {external['seed_papers']} seed papers;",
        f"{external['documents']} unique seed/candidate texts were embedded using the same pinned MiniLM revision.",
        "Each event reranks ten previously displayed candidates. This is conditional reranking, not",
        "retrieval from an unrestricted corpus. All unlabeled text in the candidate union was used for",
        "TF-IDF fitting; click labels were used only for evaluation. The SciFact-selected weights were unchanged.",
        "",
        "| Metric | Original semantic | Recommended blend |",
        "|---|---:|---:|",
    ]
    for metric in external["summary"]["dense"]:
        lines.append(
            f"| {metric} | {external['summary']['dense'][metric]:.4f} | {external['summary']['hybrid'][metric]:.4f} |"
        )
    low, high = external["comparison"]["ci95"]
    lines += [
        "",
        f"Adjusted nDCG@10 difference: {external['comparison']['mean_difference']:+.4f}; 95% interval [{low:+.4f}, {high:+.4f}].",
        "Adjusted metrics weight each event by inverse propensity of the clicked paper's original display position.",
        "Weight normalization cancels in the weighted mean. Score ties receive midpoint rank, matching the",
        "official evaluator. Pairwise accuracy counts the fraction of displayed alternatives ranked below the",
        "clicked paper, with half credit for ties; it is not overall recommendation relevance accuracy.",
        "Clicks reflect exposure and historical recommender bias. Unclicked papers are not verified irrelevant.",
        "",
        "Sources: [official SciDocs dataset and task description](https://github.com/allenai/scidocs),",
        "[official click-evaluation formula](https://github.com/allenai/scidocs/blob/master/scidocs/recomm_click_eval.py).",
        "This experiment evaluates raw content rankers using those formulas. It does not train the official",
        "SciDocs feature-based recommender, so these numbers are not a reproduction of its full leaderboard protocol.",
        "",
        "## Uncertainty and verification",
        "",
        "Both comparisons use 10,000 paired cluster-bootstrap resamples, seed 42. Local resampling preserves",
        "connected co-evidence components; external resampling preserves repeated events for each seed paper.",
        "The local nDCG@5 comparison is primary; the external transfer test is secondary. Neither interval",
        "excludes zero. Larger independent relevance judgments are needed before claiming a reliable gain.",
        "",
        f"Independent verification passed {verification['local_metric_checks']} local metric checks (pytrec_eval plus Hit@5),",
        f"{verification['external_metric_checks']} external per-event metric checks, all aggregate metrics, and",
        f"{verification['serving_parity_checks']} full serving-ranking parity checks. Top-five prefix consistency was checked too.",
        "Frozen profile hashes match both tests. Source file hashes and model revisions are saved in the artifacts.",
        "",
        "![Paired recommendation differences and uncertainty](../experiments/results/related-comparison.png)",
        "",
        "## Reproduce",
        "",
        "```powershell",
        ".venv\\Scripts\\python scripts/prepare_related_external.py",
        ".venv\\Scripts\\python scripts/tune_related.py",
        ".venv\\Scripts\\python scripts/benchmark_related.py",
        ".venv\\Scripts\\python scripts/benchmark_related.py --external",
        ".venv\\Scripts\\python scripts/verify_related.py",
        ".venv\\Scripts\\python scripts/report_related.py",
        "```",
        "",
        "Run from the repository root after the SciFact index is built. External preparation downloads about",
        "169 MB of public metadata/click files and encodes only the 8,758 needed texts, with resumable chunks.",
        "No test metrics are computed during preparation. The raw data and embedding cache are git-ignored.",
        "Keep source and profile hashes with any rerun; results may vary with dependencies or hardware.",
    ]
    Path("docs/RELATED_PAPERS_EVALUATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    for y, result in enumerate([local, external]):
        c = result["comparison"]
        mean = c["mean_difference"]
        ax.errorbar(
            mean, y, xerr=[[mean - c["ci95"][0]], [c["ci95"][1] - mean]], fmt="o", capsize=6, color="#23786b"
        )
    ax.axvline(0, color="#777777", linestyle="--")
    ax.set_yticks([0, 1], ["Local proxy: nDCG@5", "Logged clicks: adjusted nDCG@10"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Blend minus original; 95% paired cluster-bootstrap interval")
    ax.set_title("Both intervals cross zero; tasks have different candidate pools")
    fig.savefig("experiments/results/related-comparison.png", dpi=160)
    plt.close(fig)
    print("Recommendation report and uncertainty figure generated.")


if __name__ == "__main__":
    main()
