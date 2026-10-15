# Local operation and deployment boundaries

Use Python 3.12 and Node.js 22 or newer. Run commands from the repository root unless noted.
Install the pinned Python environment from `requirements.lock.txt`; `requirements.txt`
expresses compatible version ranges for deliberate future upgrades. `npm ci` uses the JS lock.

Download models and build the index with internet access first. After caching checkpoints,
set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to enforce offline inference. Initial model
downloads may be slower or throttled without an optional Hugging Face token. Never commit tokens.

```sh
python scripts/download_dataset.py --dataset scifact
python scripts/build_indexes.py --dataset scifact
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
# separate terminal
cd frontend
npm ci
npm run dev
```

The frontend proxies `/api` to port 8000. Open http://127.0.0.1:5173.
OpenAPI is at http://127.0.0.1:8000/docs. A missing/corrupt index leaves health available
but returns HTTP 503 for retrieval. Rebuild the index rather than changing its metadata by hand.
Missing models do not cause fallback to a different retrieval method.

## Optional discovery corpus

```sh
python scripts/ingest_semantic_scholar.py "scientific information retrieval" --limit 50
python scripts/build_indexes.py --dataset semantic-scholar
```

Set `SCHOLARMIND_INDEX_DIR=indexes/semantic-scholar` and restart the backend. Health and results
label the active corpus. The evaluation dashboard continues to label its saved SciFact study.
Ingestion caches requests for 24 hours, caps each request at 100 papers and uses four attempts
with bounded exponential backoff for transient HTTP errors. A 429 after retries is surfaced;
no scraping or restriction bypass is used. The API key is optional, subject to upstream access.

## Docker

Build indexes on the host first, then run `docker compose up --build`. Open http://localhost:8080.
Indexes and results mount read-only; the model cache persists separately. First neural search
inside Docker downloads checkpoint weights into that cache. The healthcheck verifies index
readiness, not that all model weights are cached. Use one API process to avoid duplicate weights.
Docker files require an installed/running Docker engine; execution status is recorded in VALIDATION.md.

## Troubleshooting

- Python 3.14 is outside the tested range; use a 3.12 virtual environment.
- Unknown document IDs return 404. Empty queries and out-of-range top-K return 422.
- SciFact lacks author/year/URL metadata here; these are not API failures.
- Sentence Transformers safely truncates input. Token-length audit may warn about long input;
  that audit is followed by model encoding with its configured truncation, not an unbounded forward pass.
- The CPU backend serializes inference. Concurrent users wait for the active request.
- Stop with Ctrl+C. No external database or background cloud job is required.
