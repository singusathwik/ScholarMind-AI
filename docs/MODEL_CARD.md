# Model card

## Intended use

Local English scientific literature discovery, transparent ranking comparisons, and
content-based related-paper suggestions. Not clinical decision support, automatic fact
verification, exhaustive systematic review, full-text understanding, or personalized recommendation.

## Components

- `rank-bm25` BM25Okapi, k1=1.5 and b=0.75; Unicode word tokenization, casefolding,
  no stemming or stopword removal. This is not an Elasticsearch-equivalent baseline.
- `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`: six-layer transformer, mean-pooled 384-dimensional
  representations, L2 normalized; 256-token input limit. The
  [checkpoint card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) describes training.
- Exact `faiss.IndexFlatIP`, equivalent to cosine similarity on normalized vectors.
- RRF with one-based rank and k=60; 100 candidates from each base retriever.
- `cross-encoder/ms-marco-MiniLM-L-6-v2`, revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`: six-layer MS MARCO cross-encoder;
  query/document pairs truncated to 512 tokens, batches of 16. No score is presented as a
  calibrated probability. See the [maintainer's checkpoint card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).

No model was fine-tuned in this project. Weights retain their upstream licenses. Index building
resolves immutable revisions and serving uses those recorded revisions. New builds can pin
them explicitly with `--revision` and `--reranker-revision`.

## Evaluation and compute

See [the technical report](TECHNICAL_REPORT.md) and machine-readable files under
`experiments/results/`. Test labels never enter model inference. The local run uses CPU,
four PyTorch threads, document batches of 32, Intel Core i5-13500H and approximately 16 GB RAM.
Index vectors occupy roughly 8 MB for this corpus; metadata and models require additional memory.
Model loading/download and the token audit are included in the measured 338.79-second embedding
preparation time; exact FAISS construction took 0.011 seconds. These are single-run local values.

## Limits and fairness

3,681/5,183 documents have text truncated by MiniLM. Query/document truncation also affects
cross-encoder scoring. Missing qrels are not proof of irrelevance. Pretrained models can
inherit publication, language, topic, geographic and demographic biases; no demographic audit
was performed. Score magnitudes are method-specific and cannot be compared across modes.
Similarity is not a probability. Recommendations are not validated reading prescriptions.
No GPU, multi-user load, multilingual retrieval or external deployment validation was performed.
