# Executed validation

Local validation date: 2026-10-15. Windows, Python 3.12.15, Node.js 25.3.0,
Intel Core i5-13500H, 16,821,166,080 bytes installed RAM. No GPU was required.

| Check | Observed result |
|---|---|
| Official BEIR SciFact download and structure validation | 5,183 documents; 300 test queries; 809 train queries; no split-ID overlap |
| NFCorpus download and structure validation | 3,633 documents; 323 test queries; full neural benchmark not run |
| Real Sentence Transformer embedding | All 5,183 documents encoded, 384 dimensions; persisted and reloaded |
| Real cross-encoder execution | All 300 held-out queries reranked |
| Offline inference with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 | All four search modes returned real papers; five recommendations returned using cached assets |
| Four-method benchmark | Completed; JSON, summary CSV, per-query CSV, plot and confidence intervals saved |
| Development ablations | Six executed runs: rerank depths 10/20/50 and RRF constants 10/60/100, each on 50 train queries |
| Python `pytest -q` | 30 passed |
| `ruff check .` | Passed |
| Frontend `npm run typecheck` | Passed |
| Frontend `npm test` | 4 passed |
| Frontend `npm run build` | Passed |
| `npx playwright test` against real API, Microsoft Edge | 2 passed on dev UI; the same 2 passed on the production Vite preview |
| npm audit after test-tool update | 0 reported vulnerabilities |
| Semantic Scholar live ingestion | Blocked by HTTP 429 after bounded retries |
| Docker execution | Not run: Docker executable unavailable |
| Hosted GitHub Actions | Configuration provided; not remotely executed |
| Public deployment | Not performed or claimed |

Browser tests exercised all four modes over real SciFact documents, pagination, paper detail
dialog, five related papers, Escape dismissal, evaluation table and 390px mobile search.
No browser page errors were observed. Actual captures are in `docs/screenshots/` and were
visually inspected. The two tests contain multiple assertions and are separate from the
four fixture-backed React tests.

The Python test suite includes randomized graded-metric equivalence against pytrec_eval,
corpus/qrels integrity, conflicting IDs, FAISS save/load/checksum rejection, deterministic
ranking, hybrid fusion, selected-paper exclusion, API validation/statuses and cached ingestion.
Synthetic test fixtures and model doubles are not treated as benchmark evidence.

The saved 300-query four-method rankings were independently re-evaluated using pytrec_eval:
all 7,200 per-query metric comparisons and aggregate effectiveness metrics agree within
1e-10. Reproduce with `python scripts/verify_results.py experiments/results/scifact-test/results.json`.
All six development runs also agree with pytrec_eval: another 1,800 per-query metric
comparisons. Compose and CI YAML parse successfully; all code cells in the four notebooks
compile, but notebook execution is not claimed.

The production build was generated locally. CI and Docker definitions are inspectable
configuration, not evidence of a cloud deployment. No fresh-machine installation or
multi-user load test has been claimed. Initial setup in this empty workspace used an isolated
Python virtual environment and npm installation; lockfiles record the resolved dependencies.
