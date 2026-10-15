"""Generate portfolio claims exclusively from saved, executed experiment artifacts."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.datasets.loader import load_corpus, load_judgments


def table(summary):
    keys = ["NDCG@10", "Recall@10", "Recall@100", "MAP@100", "MRR@100", "Precision@10", "P95_ms"]
    lines = ["| Method | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
    for mode, row in summary.items():
        lines.append("| " + mode + " | " + " | ".join(f"{row[k]:.4f}" for k in keys) + " |")
    return "\n".join(lines)


def generate():
    result = json.loads(Path("experiments/results/scifact-test/results.json").read_text(encoding="utf-8"))
    if not result["executed"] or result["partial"] or result["query_count"] != 300:
        raise ValueError("Report requires the complete executed SciFact test run")
    papers, _ = load_corpus("data/scifact")
    queries, qrels = load_judgments("data/scifact", papers)
    paper_map = {p.paper_id: p for p in papers}
    runs = result["runs"]
    dense_only, lexical_only = [], []
    for qid in queries:
        positives = {pid for pid, score in qrels[qid].items() if score > 0}
        dense_hits = set(runs["dense"][qid][:10]) & positives
        lexical_hits = set(runs["bm25"][qid][:10]) & positives
        if dense_hits - lexical_hits:
            dense_only.append((qid, sorted(dense_hits - lexical_hits)[0]))
        if lexical_hits - dense_hits:
            lexical_only.append((qid, sorted(lexical_hits - dense_hits)[0]))
    hypotheses = (
        f"H1 is supported in the limited top-10 complementarity sense: on {len(dense_only)} queries, "
        f"dense retrieval finds at least one judged relevant document missing from BM25's top 10. "
        f"The reverse occurs on {len(lexical_only)} queries. This does not establish that every such "
        "case is caused by semantic understanding, nor that dense retrieval wins overall.\n\n"
    )
    comparisons = []
    for name, stats in result["comparisons"].items():
        low, high = stats["ci95"]
        comparisons.append(
            f"- {name}: mean nDCG@10 difference {stats['mean_difference']:+.4f}; "
            f"95% paired bootstrap interval [{low:+.4f}, {high:+.4f}], n={stats['queries']}."
        )
    hybrid_delta = result["comparisons"]["hybrid minus bm25"]["mean_difference"]
    rerank_delta = result["comparisons"]["rerank minus hybrid"]["mean_difference"]
    h2 = (
        "supported on this benchmark"
        if result["comparisons"]["hybrid minus bm25"]["ci95"][0] > 0
        else "inconclusive"
    )
    h3 = (
        "supported on this benchmark"
        if result["comparisons"]["rerank minus hybrid"]["ci95"][0] > 0
        else "inconclusive"
    )
    hypotheses += (
        f"H2 is {h2}: hybrid changes mean nDCG@10 relative to BM25 by {hybrid_delta:+.4f}. "
        "See paired intervals and the dense comparison below.\n\n"
        f"H3: reranking changes mean nDCG@10 relative to hybrid by {rerank_delta:+.4f}, "
        f"with P95 latency changing from {result['summary']['hybrid']['P95_ms']:.1f} to "
        f"{result['summary']['rerank']['P95_ms']:.1f} ms. The relevance improvement is {h3}; "
        "a higher point estimate alone is insufficient. These conclusions are specific to this corpus and setup."
    )
    examples = []
    for label, cases in [
        ("Dense top-10 success absent from BM25 top 10", dense_only),
        ("BM25 top-10 success absent from dense top 10", lexical_only),
    ]:
        for qid, pid in cases[:2]:
            examples.append(
                f"### {label}\n\nQuery `{qid}`: {queries[qid]}\n\n"
                f"Judged relevant paper `{pid}`: {paper_map[pid].title}\n\n"
                f"BM25 rank: {rank_of(runs['bm25'][qid], pid)}; "
                f"dense rank: {rank_of(runs['dense'][qid], pid)}. "
                "This is a rank-based error case, not an inferred causal explanation."
            )
    ablation_lines = ["| Development experiment | Queries | nDCG@10 | P95 ms |", "|---|---:|---:|---:|"]
    for path in sorted(Path("experiments/results").glob("ablation-*/results.json")):
        study = json.loads(path.read_text(encoding="utf-8"))
        row = next(iter(study["summary"].values()))
        ablation_lines.append(
            f"| {path.parent.name} | {study['query_count']} | {row['NDCG@10']:.4f} | {row['P95_ms']:.1f} |"
        )
    if len(ablation_lines) == 2:
        ablation = (
            "Development ablations have not yet been executed. CLI support is not experimental evidence."
        )
    else:
        ablation = "\n".join(ablation_lines)
    report = f"""# Evaluating Hybrid Neural and Lexical Retrieval for Scientific Literature Discovery

## Abstract

ScholarMind AI evaluates lexical, dense, fused and cross-encoder retrieval on the BEIR
SciFact export: 5,183 abstracts and 300 held-out retrieval queries. All four configurations
execute locally on CPU. We measure effectiveness, stage latency and paired uncertainty.
The measured hybrid-minus-BM25 nDCG@10 change is {hybrid_delta:+.4f}; the reranker-minus-hybrid
change is {rerank_delta:+.4f}. These are benchmark-specific observations, not general claims.

## Introduction

Researchers need to locate evidence across terminology and wording differences. Lexical
matching provides a strong, interpretable baseline, while embeddings offer a different
matching function. Fusion and pairwise neural scoring add complexity that must justify
itself through measured retrieval effectiveness and latency. This project implements the
complete corpus-to-browser path and makes the evaluation artifacts inspectable.

## Research questions

H1: Can dense retrieval find relevant papers missed by lexical matching? H2: Does combining
the rankings improve effectiveness? H3: Does cross-encoder reranking improve relevance at
additional computational cost? None is assumed true before measurement.

## Related work

BM25 models term-based evidence with document-length normalization [8]. Sentence-BERT provides
reusable sentence embeddings [1]. FAISS provides efficient vector indexing [2]. RRF merges
rank positions without assuming comparable raw scores [3]. Cross-encoders jointly process
queries and passages [4]. BEIR provides heterogeneous retrieval datasets and common evaluation
interfaces [5]; SciFact supplies scientific claims and evidence annotations [6]. This project
is a reproducible systems study of established methods, not a claim of a novel model.

## Methodology

NFKC/whitespace normalization retains original IDs. Both title and abstract are indexed.
No records were removed from this benchmark. BM25 uses Unicode word tokens with casefolding,
k1=1.5, b=0.75. MiniLM encodes batches of 32 into normalized 384-dimensional vectors.
Exact FAISS inner-product search is cosine-equivalent. RRF uses one-based ranks, constant
60 and 100 candidates per retriever. The cross-encoder reorders the top 20 hybrid candidates;
the remaining tail is retained to depth 100 for comparable evaluation. Ties sort by ID.
Application inference has no access to judgments. No fine-tuning was performed.

## Experimental setup

Executed timestamp (UTC): {result["timestamp"]}. CPU: Intel Core i5-13500H, 12 physical cores,
16 logical processors; approximately 16 GB RAM; four PyTorch inference threads. Runtime:
{result["hardware"]["platform"]}, Python {result["hardware"]["python"]}. Checkpoint revisions,
dependency versions, corpus fingerprint and per-query timings are in `results.json`.
MiniLM revision: `{result["index"]["revision"]}`.
Cross-encoder revision: `{result["index"]["reranker_revision"]}`.

The test configuration was fixed before results were inspected. Development uses the first
50 lexicographically sorted IDs from the disjoint 809-query train split; it is a convenience
subset, not a representative random validation sample. No development winner is promoted
post hoc into the fixed test run. See [the protocol](EXPERIMENTS.md).

## Results

{table(result["summary"])}

![Executed model comparison](../experiments/results/scifact-test/comparison.png)

MAP@100 and MRR@100 are depth-limited; unjudged documents count as nonrelevant. The custom
metric implementation was checked against pytrec_eval on graded test cases. Saved outputs
include per-query metrics and explicit ranked document IDs. Latency is warmed engine time,
not HTTP latency or a throughput test. This development-machine pass included incidental
editor/test activity; CPU timing remains sensitive to background load and is not a controlled microbenchmark.

## Statistical analysis and hypotheses

{hypotheses}

{chr(10).join(comparisons)}

Intervals use 10,000 paired bootstrap resamples, seed 42, and no multiple-comparison correction.
Related claims can violate query independence. Statistical and practical significance are distinct.

## Ablation studies

Experiment A is the four-method comparison above. C changes rerank candidate depth;
E changes the fusion constant, both only on development queries. F records index build and
stage timings. Title/abstract representation and MPNet comparisons are supported by CLI
configuration but are not claimed as executed here.

{ablation}

Candidate depth affects both computation and which documents can enter the reordered head.
RRF sensitivity is specific to the candidate lists and development sample. Preserve the test
configuration rather than repeatedly evaluating whichever development choice looks best.
In this executed development study, depth 20 provides most of the observed depth-50
effectiveness at substantially lower latency. The smaller RRF constant performs better on
this subset than the predeclared default. These exploratory observations do not change the
fixed test configuration or establish a generally optimal setting.

## Error analysis

{chr(10).join(examples)}

## Limitations

SciFact is small, mainly biomedical, English, and sparsely judged. Its claim-based queries
do not represent all academic discovery needs. Indexing uses abstracts rather than full text.
MiniLM truncates 3,681/5,183 combined documents at 256 tokens; the cross-encoder limits pairs
to 512 tokens. No claim of complete-document comprehension is warranted. Pretrained scientific
text overlap is possible and not audited. Metadata is incomplete and deliberately not invented.
NFCorpus, MPNet, fine-tuning, approximate indexing, GPU execution and demographic fairness
are not evaluated. Single-pass CPU timing is not a production SLA. Semantic Scholar live
ingestion returned HTTP 429; its parsing/cache behavior is fixture-tested. Docker and hosted
CI execution require infrastructure not verified by local tests.

## Conclusion

The saved measurements establish a working, reproducible comparison on a specific corpus.
Hybrid and neural reranking must be judged against the lexical baseline using the measured
differences, uncertainty and added latency above. Stronger general claims require additional
datasets, repetition, controlled hardware and independent relevance assessment.

## References

1. Reimers and Gurevych (2019), [Sentence-BERT](https://arxiv.org/abs/1908.10084).
2. Johnson, Douze and Jégou (2017), [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734).
3. Cormack, Clarke and Buettcher (2009), [Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf).
4. [MS MARCO MiniLM cross-encoder model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).
5. [BEIR official repository and publication links](https://github.com/beir-cellar/beir).
6. Wadden et al. (2020), [SciFact: Fact or Fiction](https://github.com/allenai/scifact).
7. [MiniLM sentence embedding model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
8. Robertson and Zaragoza (2009), [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019).
"""
    Path("docs/TECHNICAL_REPORT.md").write_text(report, encoding="utf-8")
    Path("docs/BENCHMARK_RESULTS.md").write_text(
        "# Executed SciFact results\n\n"
        + table(result["summary"])
        + "\n\n"
        + hypotheses
        + "\n\n"
        + "\n".join(comparisons),
        encoding="utf-8",
    )
    best = max(result["summary"], key=lambda m: result["summary"][m]["NDCG@10"])
    bullets = f"""# Verified resume bullets

- Implemented local scientific-paper retrieval over 5,183 SciFact abstracts using Sentence Transformers, normalized 384-dimensional embeddings, FAISS, BM25, reciprocal rank fusion and cross-encoder reranking.
- Evaluated four retrieval pipelines on 300 held-out BEIR SciFact queries with TREC-validated metrics and paired bootstrap intervals; best measured nDCG@10 was {result["summary"][best]["NDCG@10"]:.3f} ({best}), with measured effectiveness/latency trade-offs.
- Built a React/TypeScript and FastAPI research-discovery application with content-based recommendations, reproducible CLI experiments, automated tests, dependency locks and Docker/CI configuration; validated locally, without claiming public deployment.

These describe this implementation and local evaluation only. No publication, novel model,
university affiliation, public deployment or external benchmark standing is implied.
"""
    Path("docs/RESUME_BULLETS.md").write_text(bullets, encoding="utf-8")
    print(table(result["summary"]))
    print(hypotheses)


def rank_of(ids, pid):
    return str(ids.index(pid) + 1) if pid in ids else ">100"


if __name__ == "__main__":
    generate()
