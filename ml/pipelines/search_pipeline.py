import json
from pathlib import Path
from threading import RLock
from time import perf_counter

from ml.datasets.loader import Paper
from ml.retrieval.bm25 import BM25, tokenize
from ml.retrieval.dense import Encoder
from ml.retrieval.faiss_store import VectorStore
from ml.retrieval.hybrid import rrf
from ml.retrieval.reranker import Reranker


class SearchEngine:
    def __init__(self, folder):
        self.store, self.meta = VectorStore.load(folder)
        papers = [Paper(**p) for p in json.loads((Path(folder) / "papers.json").read_text(encoding="utf-8"))]
        if [p.paper_id for p in papers] != self.store.ids:
            raise ValueError("Metadata mapping mismatch")
        self.papers = {p.paper_id: p for p in papers}
        self.representation = self.meta["representation"]
        self.bm25 = BM25(papers, self.representation)
        self.encoder = None
        self.reranker = None
        self.lock = RLock()

    def get_encoder(self):
        if self.encoder is None:
            self.encoder = Encoder(self.meta["model"], self.meta["revision"])
        return self.encoder

    def get_reranker(self):
        if self.reranker is None:
            self.reranker = Reranker(self.meta["reranker"], self.meta["reranker_revision"])
        return self.reranker

    def search(
        self,
        query,
        mode="hybrid",
        top_k=10,
        candidates=100,
        rerank_depth=20,
        constant=60,
        year=None,
        field=None,
    ):
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if min(top_k, candidates, rerank_depth) < 1 or constant < 0:
            raise ValueError("Retrieval depths must be positive and RRF constant nonnegative")
        if mode not in {"bm25", "dense", "hybrid", "rerank"}:
            raise ValueError("Unknown retrieval mode")
        with self.lock:
            start = perf_counter()
            timings = {"encoding_ms": 0.0, "retrieval_ms": 0.0, "reranking_ms": 0.0}
            # Metadata filtering happens over a full candidate ranking when requested.
            depth = len(self.papers) if year or field else max(candidates, top_k, rerank_depth)
            t = perf_counter()
            lexical = self.bm25.search(query, depth) if mode != "dense" else []
            timings["retrieval_ms"] += (perf_counter() - t) * 1000
            dense = []
            if mode != "bm25":
                t = perf_counter()
                vector = self.get_encoder().encode([query])[0]
                timings["encoding_ms"] = (perf_counter() - t) * 1000
                t = perf_counter()
                dense = self.store.search(vector, depth)
                timings["retrieval_ms"] += (perf_counter() - t) * 1000
            t = perf_counter()
            ranking = (
                lexical
                if mode == "bm25"
                else dense
                if mode == "dense"
                else rrf([lexical, dense], constant, depth)
            )
            if year or field:
                ranking = [
                    (pid, s)
                    for pid, s in ranking
                    if (not year or self.papers[pid].publication_year == year)
                    and (
                        not field
                        or field.casefold() in [f.casefold() for f in self.papers[pid].fields_of_study]
                    )
                ]
            timings["retrieval_ms"] += (perf_counter() - t) * 1000
            score_type = {
                "bm25": "BM25",
                "dense": "cosine similarity",
                "hybrid": "RRF",
                "rerank": "cross-encoder score",
            }[mode]
            if mode == "rerank":
                t = perf_counter()
                head = self.get_reranker().rank(
                    query, ranking[:rerank_depth], self.papers, self.representation
                )
                # Preserve remaining candidates for common-depth benchmark evaluation.
                ranking = head + ranking[rerank_depth:]
                timings["reranking_ms"] = (perf_counter() - t) * 1000
            timings["total_ms"] = (perf_counter() - start) * 1000
            return ranking[:top_k], timings, score_type

    def recommend(self, pid, top_k=5):
        if pid not in self.papers:
            raise KeyError(pid)
        row = self.store.ids.index(pid)
        results = self.store.search(self.store.vectors[row], top_k + 1)
        return [(other, score) for other, score in results if other != pid][:top_k]

    def shared_terms(self, first, second):
        return sorted(set(tokenize(self.papers[first].title)) & set(tokenize(self.papers[second].title)))
