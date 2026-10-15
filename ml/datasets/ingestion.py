"""Optional bounded Semantic Scholar ingestion; independent of benchmark corpora."""

import hashlib
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv


def ingest(query, limit=50, output=Path("data/semantic-scholar"), client=None):
    load_dotenv()
    if not query.strip() or not 1 <= limit <= 100:
        raise ValueError("Provide query and limit between 1 and 100")
    output = Path(output)
    cache = Path(".cache/semantic-scholar")
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{query}:{limit}".encode()).hexdigest()
    cached = cache / f"{key}.json"
    if cached.exists() and time.time() - cached.stat().st_mtime < 86400:
        payload = json.loads(cached.read_text(encoding="utf-8"))
    else:
        headers = {}
        if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
            headers["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
        with client or httpx.Client(timeout=30) as connection:
            for attempt in range(4):
                try:
                    response = connection.get(
                        "https://api.semanticscholar.org/graph/v1/paper/search",
                        params={
                            "query": query,
                            "limit": limit,
                            "fields": "paperId,title,abstract,authors,year,url,fieldsOfStudy",
                        },
                        headers=headers,
                    )
                except httpx.RequestError:
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
                    continue
                if response.status_code not in {429, 500, 502, 503, 504}:
                    break
                if attempt < 3:
                    retry_after = response.headers.get("Retry-After", "")
                    if retry_after:
                        # A requested cooldown beyond our bounded wait ends this run.
                        if not retry_after.isdigit() or int(retry_after) > 30:
                            response.raise_for_status()
                        time.sleep(int(retry_after))
                        continue
                    time.sleep(2**attempt)
            response.raise_for_status()
            payload = response.json()
        cached.write_text(json.dumps(payload), encoding="utf-8")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for p in payload.get("data", []):
        if not p.get("paperId"):
            continue
        records.append(
            {
                "_id": p["paperId"],
                "title": p.get("title") or "",
                "text": p.get("abstract") or "",
                "authors": [a["name"] for a in p.get("authors", []) if a.get("name")],
                "publication_year": p.get("year"),
                "url": p.get("url"),
                "fields_of_study": p.get("fieldsOfStudy") or [],
                "source": "Semantic Scholar",
            }
        )
    (output / "corpus.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return len(records)
