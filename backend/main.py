import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from backend.config import Settings
from ml.pipelines.search_pipeline import SearchEngine

logger = logging.getLogger("scholarmind")
logging.basicConfig(level=logging.INFO, format='{"level":"%(levelname)s","event":%(message)s}')
settings = Settings()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    mode: Literal["bm25", "dense", "hybrid", "rerank", "optimized"] = "hybrid"
    top_k: int = Field(default=10, ge=1, le=50)
    publication_year: int | None = Field(default=None, ge=1600, le=2100)
    field_of_study: str | None = Field(default=None, max_length=100)
    reranking: bool = False

    @field_validator("query")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("Enter a nonempty query")
        return value.strip()


class PaperResponse(BaseModel):
    paper_id: str
    title: str
    abstract: str
    source: str
    authors: list[str]
    publication_year: int | None
    url: str | None
    fields_of_study: list[str]


class Hit(PaperResponse):
    score: float
    score_type: str


class SearchResponse(BaseModel):
    results: list[Hit]
    mode: str
    dataset: str
    timings: dict[str, float]
    returned_count: int
    total_result_count: int | None = None


def create_app(engine=None):
    @asynccontextmanager
    async def lifespan(app):
        if app.state.engine is None:
            try:
                app.state.engine = SearchEngine(settings.index_dir)
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                logger.error(json.dumps({"event": "index_unavailable", "error": str(exc)}))
        yield

    app = FastAPI(title="ScholarMind AI", version="1.0.0", lifespan=lifespan)
    app.state.engine = engine
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def get_engine():
        if app.state.engine is None:
            raise HTTPException(503, "Index unavailable. Download the corpus and build indexes first.")
        return app.state.engine

    @app.get("/api/health")
    def health():
        e = app.state.engine
        optimized_available = False
        if e:
            try:
                e.get_optimized_profile()
                optimized_available = True
            except (OSError, ValueError, KeyError):
                pass
        return {
            "status": "ready" if e else "index_unavailable",
            "index_ready": bool(e),
            "dataset": e.meta["dataset"] if e else settings.dataset,
            "documents": len(e.papers) if e else 0,
            "optimized_available": optimized_available,
            "loaded_models": {"encoder": bool(e and e.encoder), "reranker": bool(e and e.reranker)},
        }

    @app.get("/api/models")
    def models():
        e = get_engine()
        return {
            "modes": ["bm25", "dense", "hybrid", "rerank"]
            + (["optimized"] if health()["optimized_available"] else []),
            "model": e.meta["model"],
            "revision": e.meta["revision"],
            "reranker": e.meta["reranker"],
            "dataset": e.meta["dataset"],
            "max_seq_length": e.meta["max_seq_length"],
        }

    @app.post("/api/search", response_model=SearchResponse)
    def search(request: SearchRequest):
        e = get_engine()
        mode = "rerank" if request.reranking and request.mode != "optimized" else request.mode
        try:
            ranked, timings, score_type = e.search(
                request.query,
                mode,
                request.top_k,
                rerank_depth=max(20, request.top_k),
                year=request.publication_year,
                field=request.field_of_study,
            )
        except (OSError, RuntimeError) as exc:
            logger.error(json.dumps({"event": "model_unavailable", "error": str(exc)}))
            raise HTTPException(
                503, "Requested model unavailable; check model cache and server logs."
            ) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        logger.info(json.dumps({"event": "search", "mode": mode, "results": len(ranked), **timings}))
        return {
            "results": [
                dict(asdict(e.papers[pid]), score=score, score_type=score_type) for pid, score in ranked
            ],
            "mode": mode,
            "dataset": e.meta["dataset"],
            "timings": timings,
            "returned_count": len(ranked),
            "total_result_count": None,
        }

    @app.get("/api/papers/{paper_id}", response_model=PaperResponse)
    def paper(paper_id: str):
        e = get_engine()
        if paper_id not in e.papers:
            raise HTTPException(404, "Paper not found")
        return asdict(e.papers[paper_id])

    @app.get("/api/papers/{paper_id}/recommendations")
    def recommendations(paper_id: str, top_k: int = Query(default=5, ge=1, le=20)):
        e = get_engine()
        try:
            results = e.recommend(paper_id, top_k)
        except KeyError as exc:
            raise HTTPException(404, "Paper not found") from exc
        return {
            "results": [
                dict(
                    asdict(e.papers[pid]),
                    score=score,
                    score_type="cosine similarity",
                    shared_title_terms=e.shared_terms(paper_id, pid),
                )
                for pid, score in results
            ],
            "explanation": "Content similarity, not a probability or personalized prediction.",
        }

    @app.get("/api/benchmark-results")
    def benchmarks():
        if app.state.engine and app.state.engine.meta["dataset"] != "scifact":
            return {"available": False, "message": "No saved evaluation for the active corpus"}
        path = Path("experiments/results/scifact-optimized-test/results.json")
        if not path.exists():
            path = Path("experiments/results/scifact-test/results.json")
        if not path.exists():
            return {"available": False, "message": "Not yet evaluated"}
        result = json.loads(path.read_text(encoding="utf-8"))
        return {"available": True, **{k: v for k, v in result.items() if k not in {"runs", "per_query"}}}

    return app


app = create_app()
