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
from ml.retrieval.related import RelatedRecommender


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
        self.optimized_profile = None
        self.title_bm25 = None
        self.related_profile = None
        self.related_model = None

    def get_optimized_profile(self):
        if self.optimized_profile is None:
            path = Path(__file__).resolve().parents[2] / "experiments/configs/optimized-scifact.json"
            profile = json.loads(path.read_text(encoding="utf-8"))
            for key in ["dataset", "model", "revision", "reranker", "reranker_revision", "representation"]:
                if self.meta.get(key) != profile[key]:
                    raise ValueError("Optimized profile does not match this index")
            if self.meta.get("stats", {}).get("fingerprint") != profile["corpus_fingerprint"]:
                raise ValueError("Optimized profile does not match this corpus")
            self.optimized_profile = profile
        return self.optimized_profile

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
        if mode not in {"bm25", "dense", "hybrid", "rerank", "optimized"}:
            raise ValueError("Unknown retrieval mode")
        with self.lock:
            start = perf_counter()
            profile = self.get_optimized_profile() if mode == "optimized" else None
            if profile:
                candidates = profile["candidates"]
                rerank_depth = profile["rerank_depth"]
                constant = profile["constant"]
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
            if profile:
                titles = []
                if profile["title_weight"]:
                    if self.title_bm25 is None:
                        self.title_bm25 = BM25(list(self.papers.values()), "title")
                    titles = self.title_bm25.search(query, depth)
                ranking = rrf(
                    [lexical, dense, titles],
                    constant,
                    depth,
                    [profile["lexical_weight"], 1, profile["title_weight"]],
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
                "optimized": "optimized reciprocal rank (not probability)",
            }[mode]
            if mode == "rerank" or (profile and profile["cross_encoder_weight"] > 0):
                t = perf_counter()
                head = self.get_reranker().rank(
                    query, ranking[:rerank_depth], self.papers, self.representation
                )
                if profile:
                    weight = profile["cross_encoder_weight"]
                    head = rrf(
                        [ranking[:rerank_depth], head],
                        profile["blend_constant"],
                        rerank_depth,
                        [1 - weight, weight],
                    )
                # Preserve remaining candidates for common-depth benchmark evaluation.
                ranking = head + ranking[rerank_depth:]
                timings["reranking_ms"] = (perf_counter() - t) * 1000
            if profile:
                ranking = [(pid, 1.0 / rank) for rank, (pid, _) in enumerate(ranking, 1)]
            timings["total_ms"] = (perf_counter() - start) * 1000
            return ranking[:top_k], timings, score_type

    def recommendation_profile(self):
        if self.related_profile is None:
            path = Path(__file__).resolve().parents[2] / "experiments/configs/related-scifact.json"
            try:
                profile = json.loads(path.read_text(encoding="utf-8"))
                if any(
                    self.meta.get(k) != profile[k] for k in ["dataset", "model", "revision", "representation"]
                ):
                    return None
                if self.meta.get("stats", {}).get("fingerprint") != profile["corpus_fingerprint"]:
                    return None
                self.related_profile = profile
            except (OSError, ValueError, KeyError):
                return None
        return self.related_profile

    def recommendation_method(self, requested="auto"):
        if requested not in {"auto", "dense", "hybrid"}:
            raise ValueError("Unknown recommendation method")
        profile = self.recommendation_profile() if requested != "dense" else None
        if requested == "hybrid" and (not profile or profile["method"] != "hybrid"):
            raise ValueError("No validated recommendation profile for this index")
        return "hybrid" if profile and profile["method"] == "hybrid" else "dense"

    def recommend(self, pid, top_k=5, method="auto"):
        with self.lock:
            if pid not in self.papers:
                raise KeyError(pid)
            actual = self.recommendation_method(method)
            if self.related_model is None:
                self.related_model = RelatedRecommender(self.papers, self.store)
            return self.related_model.rank(pid, top_k, self.related_profile if actual == "hybrid" else None)

    def shared_terms(self, first, second):
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        terms = set(tokenize(self.papers[first].title)) & set(tokenize(self.papers[second].title))
        terms = [t for t in terms if t not in ENGLISH_STOP_WORDS and len(t) > 2]
        return sorted(terms, key=lambda t: (-self.bm25.model.idf.get(t, 0), t))[:8]
