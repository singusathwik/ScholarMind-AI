"""Fetch only the official recommendation files and embed test-candidate text.

No rankings or test metrics are computed here; click labels are not used in embedding.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.retrieval.dense import Encoder


def main():
    folder = Path("data/scidocs-recomm")
    folder.mkdir(parents=True, exist_ok=True)
    base = "https://ai2-s2-research-public.s3.us-west-2.amazonaws.com/specter/scidocs/"
    manifest = {}
    for remote, name in [
        ("paper_metadata_recomm.json", "metadata.json"),
        ("recomm/test.csv", "test.csv"),
        ("recomm/propensity_scores.json", "propensity_scores.json"),
    ]:
        path = folder / name
        if not path.exists():
            print(f"Downloading {remote}", flush=True)
            with httpx.stream("GET", base + remote, timeout=120) as response:
                response.raise_for_status()
                temporary = path.with_suffix(".partial")
                with temporary.open("wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
                temporary.replace(path)
        manifest[name] = {"url": base + remote, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    with (folder / "test.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = sorted({pid for r in rows for pid in [r["pid"], *r["similar_papers_avail_at_time"].split(",")]})
    assert all(pid in metadata for pid in ids)
    texts = [
        " ".join((metadata[pid].get("title") or "", metadata[pid].get("abstract") or "")).strip()
        for pid in ids
    ]
    assert all(texts)
    index_meta = json.loads(Path("indexes/scifact/manifest.json").read_text())
    manifest.update(
        model=index_meta["model"],
        revision=index_meta["revision"],
        documents=len(ids),
        test_events=len(rows),
        text="title plus abstract",
        ids=ids,
    )
    print(f"Encoding {len(ids)} unique papers for {len(rows)} external test events", flush=True)
    encoder = Encoder(index_meta["model"], index_meta["revision"])
    chunks = []
    cache = folder / "embedding-chunks"
    cache.mkdir(exist_ok=True)
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    for start in range(0, len(texts), 256):
        path = cache / f"{key}-{start}.npy"
        if path.exists():
            vector = np.load(path, allow_pickle=False)
        else:
            vector = encoder.encode(texts[start : start + 256])
            np.save(path, vector)
        chunks.append(vector)
        print(f"External embeddings: {min(start + 256, len(texts))}/{len(texts)}", flush=True)
    np.save(folder / "embeddings.npy", np.concatenate(chunks))
    manifest["embedding_sha256"] = hashlib.sha256((folder / "embeddings.npy").read_bytes()).hexdigest()
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("External text preparation complete; no test metrics examined.", flush=True)


if __name__ == "__main__":
    main()
