import json
from dataclasses import asdict

import numpy as np
import pytest

from ml.datasets.loader import Paper
from ml.pipelines.search_pipeline import SearchEngine
from ml.retrieval.faiss_store import VectorStore


@pytest.fixture
def papers():
    return [
        Paper("a", "Cancer immune therapy", "T cells attack tumors", "fixture"),
        Paper("b", "Bone health", "Vitamin D protects bones", "fixture"),
        Paper("c", "Cardiac disease", "Heart genetics and exercise", "fixture"),
        Paper("d", "Immune response", "Immune cells in cancer", "fixture"),
    ]


@pytest.fixture
def engine(tmp_path, papers):
    (tmp_path / "papers.json").write_text(json.dumps([asdict(p) for p in papers]))
    VectorStore([p.paper_id for p in papers], np.array([[1, 0], [0, 1], [-1, 0], [0.9, 0.1]])).save(
        tmp_path,
        {
            "model": "test-double",
            "revision": "fixture",
            "reranker": "test-double",
            "reranker_revision": "fixture",
            "representation": "both",
            "dataset": "fixture",
            "max_seq_length": 256,
        },
    )
    e = SearchEngine(tmp_path)

    class FakeEncoder:
        def encode(self, texts):
            return np.array([[1.0, 0.0] for _ in texts], dtype="float32")

    class FakeReranker:
        def rank(self, query, candidates, papers, representation):
            return [(pid, float(i)) for i, (pid, _) in reversed(list(enumerate(candidates)))]

    e.encoder = FakeEncoder()
    e.reranker = FakeReranker()
    return e
