import csv
import hashlib
import json
import re
import unicodedata
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    source: str
    authors: list[str] = field(default_factory=list)
    publication_year: int | None = None
    url: str | None = None
    fields_of_study: list[str] = field(default_factory=list)

    def text(self, representation="both"):
        return {"title": self.title, "abstract": self.abstract, "both": f"{self.title} {self.abstract}"}[
            representation
        ].strip()


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip()


def fingerprint(papers):
    return hashlib.sha256(json.dumps([asdict(p) for p in papers], sort_keys=True).encode()).hexdigest()


def download(dataset, root=Path("data")):
    if dataset not in {"scifact", "nfcorpus"}:
        raise ValueError("Supported benchmarks: scifact, nfcorpus")
    root = Path(root)
    target = root / dataset
    if (target / "corpus.jsonl").exists():
        return target
    root.mkdir(parents=True, exist_ok=True)
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    archive = root / f"{dataset}.zip"
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with archive.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if not (root / member.filename).resolve().is_relative_to(root.resolve()):
                raise ValueError("Unsafe archive path")
        z.extractall(root)
    (target / "download.json").write_text(json.dumps({"url": url, "sha256": digest}), encoding="utf-8")
    archive.unlink()
    return target


def load_corpus(folder):
    folder = Path(folder)
    papers, seen, duplicates, empty = [], {}, 0, 0
    for line in (folder / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        pid = str(row.get("_id", row.get("paper_id", "")))
        if not pid:
            raise ValueError("Missing paper identifier")
        p = Paper(
            pid,
            normalize(row.get("title")),
            normalize(row.get("text", row.get("abstract"))),
            row.get("source", folder.name),
            row.get("authors", []),
            row.get("publication_year"),
            row.get("url") or (row.get("metadata") or {}).get("url"),
            row.get("fields_of_study", []),
        )
        if pid in seen:
            if p != seen[pid]:
                raise ValueError(f"Conflicting duplicate identifier: {pid}")
            duplicates += 1
            continue
        seen[pid] = p
        if not p.text():
            empty += 1
            continue
        papers.append(p)
    if not papers:
        raise ValueError("Empty corpus")
    papers.sort(key=lambda p: p.paper_id)
    lengths = sorted(len(p.text().split()) for p in papers)
    stats = {
        "total_records": len(papers) + duplicates + empty,
        "valid_documents": len(papers),
        "empty_removed": empty,
        "duplicates_handled": duplicates,
        "mean_whitespace_tokens": sum(lengths) / len(lengths),
        "max_whitespace_tokens": lengths[-1],
        "source": folder.name,
        "fingerprint": fingerprint(papers),
        "preprocessing": "NFKC; collapse whitespace; preserve distinct IDs even for identical text",
    }
    return papers, stats


def load_judgments(folder, papers, split="test"):
    folder = Path(folder)
    queries = {}
    for line in (folder / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        qid = str(row["_id"])
        if qid in queries:
            raise ValueError(f"Duplicate query ID: {qid}")
        queries[qid] = normalize(row["text"])
        if not queries[qid]:
            raise ValueError(f"Empty query: {qid}")
    ids = {p.paper_id for p in papers}
    qrels = {}
    with (folder / "qrels" / f"{split}.tsv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            qid, pid = row["query-id"], row["corpus-id"]
            if qid not in queries or pid not in ids:
                raise ValueError(f"Dangling relevance judgment: {qid}, {pid}")
            grade = int(row["score"])
            if grade < 0:
                raise ValueError("Negative relevance grade")
            if pid in qrels.get(qid, {}) and qrels[qid][pid] != grade:
                raise ValueError("Conflicting relevance judgment")
            qrels.setdefault(qid, {})[pid] = grade
    return {qid: queries[qid] for qid in sorted(qrels)}, qrels
