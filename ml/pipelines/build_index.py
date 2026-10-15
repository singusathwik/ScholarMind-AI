import json
import time
from dataclasses import asdict
from pathlib import Path

from huggingface_hub import model_info

from ml.datasets.loader import load_corpus
from ml.retrieval.dense import MODEL, RERANKER, Encoder
from ml.retrieval.faiss_store import VectorStore


def build(
    dataset="scifact", representation="both", model=MODEL, output=None, revision=None, reranker_revision=None
):
    papers, stats = load_corpus(Path("data") / dataset)
    folder = Path(output or f"indexes/{dataset}")
    folder.mkdir(parents=True, exist_ok=True)
    revision = model_info(model, revision=revision).sha
    reranker_revision = model_info(RERANKER, revision=reranker_revision).sha
    start = time.perf_counter()
    encoder = Encoder(model, revision)
    texts = [p.text(representation) for p in papers]
    token_lengths = [len(encoder.model.tokenizer.encode(t, truncation=False)) for t in texts]
    vectors = encoder.encode(texts, progress=True)
    embedding_seconds = time.perf_counter() - start
    (folder / "papers.json").write_text(json.dumps([asdict(p) for p in papers]), encoding="utf-8")
    start = time.perf_counter()
    store = VectorStore([p.paper_id for p in papers], vectors)
    index_seconds = time.perf_counter() - start
    meta = {
        "model": model,
        "revision": revision,
        "reranker": RERANKER,
        "reranker_revision": reranker_revision,
        "representation": representation,
        "dataset": dataset,
        "stats": stats,
        "embedding_seconds": embedding_seconds,
        "index_seconds": index_seconds,
        "max_seq_length": encoder.model.max_seq_length,
        "truncated_documents": sum(n > encoder.model.max_seq_length for n in token_lengths),
        "mean_model_tokens": sum(token_lengths) / len(token_lengths),
    }
    store.save(folder, meta)
    return meta
