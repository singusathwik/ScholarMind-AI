# Accuracy comparison: frozen training-selected ranking

Evaluated on the same 300 SciFact test queries and 5,183 documents as the original benchmark.
This measures query-to-paper retrieval, not classification accuracy or related-paper relevance.

| Method | NDCG@10 | Recall@10 | Recall@100 | MAP@100 | MRR@100 | Precision@10 |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.6519 | 0.7740 | 0.8731 | 0.6132 | 0.6233 | 0.0850 |
| dense | 0.6451 | 0.7833 | 0.9250 | 0.6031 | 0.6110 | 0.0883 |
| hybrid | 0.6816 | 0.8059 | 0.9577 | 0.6444 | 0.6539 | 0.0900 |
| rerank | 0.6871 | 0.8129 | 0.9577 | 0.6485 | 0.6640 | 0.0907 |
| optimized | 0.7052 | 0.8500 | 0.9577 | 0.6583 | 0.6730 | 0.0957 |

## Primary comparison

Optimized minus original rerank nDCG@10: +0.0180 (+2.62% relative). 95% paired bootstrap interval [+0.0079, +0.0291], 10,000 resamples, seed 42.
The interval is entirely positive on this test set.
Per-query outcomes versus rerank: {'improved': 45, 'unchanged': 244, 'worse': 11}.

## Selection and reproducibility

The search space was declared in `experiments/configs/accuracy-study.json` before test evaluation.
24 fusion configurations were compared using all 809 training queries. Five cross-encoder
blend weights were then compared on 200 seeded training queries. Both stages maximize mean nDCG@10.
These are training selection scores, not independent validation estimates.
The selected profile was frozen before test judgments were loaded; training and test IDs are disjoint.
No model weights, embeddings or corpus records were changed. The original baseline artifacts remain intact.

Selected profile:

```json
{
  "constant": 5,
  "lexical_weight": 1.0,
  "title_weight": 0.0,
  "cross_encoder_weight": 0.75,
  "blend_constant": 10,
  "rerank_depth": 20,
  "candidates": 100,
  "dataset": "scifact",
  "corpus_fingerprint": "9e76f432981aaa1c451143ef178d40c81e5c5c155588b061a9cc74d03eeddf30",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
  "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "reranker_revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
  "representation": "both",
  "selection_split": "train",
  "seed": 42
}
```

Training blend nDCG@10 by cross-encoder weight:

```json
{
  "0.0": 0.725787331989175,
  "0.25": 0.7316550927972423,
  "0.5": 0.7435849855953378,
  "0.75": 0.7575941859863603,
  "1.0": 0.7532182288458975
}
```

From the repository root:

```powershell
.venv\Scripts\python scripts/tune_accuracy.py
.venv\Scripts\python scripts/compare_accuracy.py
.venv\Scripts\python scripts/verify_results.py experiments/results/scifact-optimized-test/results.json
```

## Latency and limits

Baseline latency is historical; optimized latency is a new warmed CPU pass, not a paired speed comparison.
Optimized P50: 1291.5 ms; P95: 2123.9 ms.
The first model load is excluded. Optimized ranking reranks a fixed head of 20 papers when its
selected cross-encoder weight is positive; requesting more results preserves that head and adds the tail.
Ranking scores are reciprocal output ranks, not probabilities.

SciFact test results were previously reported. This study uses that same benchmark for a comparable
final check, not a fresh external dataset. No test-driven parameter changes were made in this study.
The primary comparison was declared in advance; other comparisons are exploratory and unadjusted
for multiple testing. Unjudged documents count as nonrelevant. Generalization to other corpora,
personalized recommendations, and the related-paper panel has not been established.

Baseline SHA-256: `575ab85773254f41432458a425a992d3eb4b7ecd6a54c965c0980e2ab3348b3a`.
Frozen profile SHA-256: `659b13e91c8b100ca49907f5dcd13171c00db8b6f9ad86dda31cbc0eecee0fe2`.
