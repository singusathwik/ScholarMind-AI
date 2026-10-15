# Experimental protocol

The test configuration is fixed in `experiments/configs/protocol.json` before examining
test results: BM25 defaults (k1=1.5, b=0.75), MiniLM, title plus abstract, normalized
IndexFlatIP, 100 lexical and 100 dense candidates, RRF k=60, rerank top 20, retrieval depth 100.
No fine-tuning or test-driven parameter selection is performed.

SciFact's BEIR `train` queries are used only for exploratory development ablations.
The ablation script uses the first 50 query IDs in lexicographic order, explicitly marked
as a partial development set. This is reproducible but not a random sample and can contain
related claims; do not interpret its intervals as independent population estimates.
The official BEIR `test` split stays fixed for the main comparison. This is a BEIR retrieval
split; it is not a claim that the original SciFact hidden verification test set is evaluated.

Experiments implemented: A (four retrieval methods), C (rerank depth 10/20/50),
E (RRF constants 10/60/100), with indexing and latency measurements supporting F.
Title-only/abstract-only representations and alternative embedding checkpoints are supported
by the index CLI but are not silently counted as executed experiments.

## Metrics

Linear-gain nDCG@10 matches `pytrec_eval`/BEIR semantics. Recall uses the count of all judged
positive documents as denominator. Precision@10 divides by 10, even for short runs.
AP accumulates precision at positive hits and divides by all known relevant documents.
MAP@100 is the mean AP of a depth-100 run; it is not untruncated MAP over the entire corpus.
MRR@100 is the first positive hit reciprocal rank within the same run. Unjudged documents
are treated as nonrelevant. Twenty seeded graded random examples are cross-checked against
`pytrec_eval` in tests. Saved rankings retain explicit order, avoiding score-scale mixing
between a reranked head and untouched tail.

Each model is warmed before per-query timing. Query encoding, retrieval/fusion, reranking,
and total engine time are saved separately. Totals exclude HTTP overhead and model downloads.
P50/P95 are empirical percentiles from one serial pass, not a service load test. CPU clocks,
background activity and thermal effects remain uncontrolled. API first-use latency includes
loading and is deliberately not described as warmed latency.

Per-query nDCG differences use paired percentile bootstrap: seed 42, 10,000 resamples,
95% intervals. Four comparisons are exploratory and unadjusted for multiple testing.
Zero-excluding intervals alone do not establish generalization to other domains.

## Commands

```sh
python scripts/benchmark_models.py --config experiments/configs/protocol.json
python scripts/run_ablations.py
python scripts/build_indexes.py --representation title --output indexes/scifact-title
python scripts/benchmark_models.py --index indexes/scifact-title --split train --output experiments/results/title-dev
```

Use unique output directories for new experiments to preserve prior results.
