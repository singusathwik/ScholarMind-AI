# Executed SciFact results

| Method | NDCG@10 | Recall@10 | Recall@100 | MAP@100 | MRR@100 | Precision@10 | P95_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.6519 | 0.7740 | 0.8731 | 0.6132 | 0.6233 | 0.0850 | 24.5821 |
| dense | 0.6451 | 0.7833 | 0.9250 | 0.6031 | 0.6110 | 0.0883 | 12.3839 |
| hybrid | 0.6816 | 0.8059 | 0.9577 | 0.6444 | 0.6539 | 0.0900 | 39.7497 |
| rerank | 0.6871 | 0.8129 | 0.9577 | 0.6485 | 0.6640 | 0.0907 | 1323.6752 |

H1 is supported in the limited top-10 complementarity sense: on 33 queries, dense retrieval finds at least one judged relevant document missing from BM25's top 10. The reverse occurs on 26 queries. This does not establish that every such case is caused by semantic understanding, nor that dense retrieval wins overall.

H2 is supported on this benchmark: hybrid changes mean nDCG@10 relative to BM25 by +0.0298. See paired intervals and the dense comparison below.

H3: reranking changes mean nDCG@10 relative to hybrid by +0.0055, with P95 latency changing from 39.7 to 1323.7 ms. The relevance improvement is inconclusive; a higher point estimate alone is insufficient. These conclusions are specific to this corpus and setup.

- dense minus bm25: mean nDCG@10 difference -0.0068; 95% paired bootstrap interval [-0.0454, +0.0316], n=300.
- hybrid minus bm25: mean nDCG@10 difference +0.0298; 95% paired bootstrap interval [+0.0066, +0.0526], n=300.
- rerank minus hybrid: mean nDCG@10 difference +0.0055; 95% paired bootstrap interval [-0.0195, +0.0308], n=300.
- hybrid minus dense: mean nDCG@10 difference +0.0366; 95% paired bootstrap interval [+0.0084, +0.0645], n=300.