import hashlib
import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, ids, vectors):
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or len(ids) != len(vectors) or len(set(ids)) != len(ids):
            raise ValueError("Invalid vector mapping")
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("Invalid embeddings")
        self.ids = ids
        self.vectors = vectors.copy()
        faiss.normalize_L2(self.vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(self.vectors)

    def search(self, vector, top_k):
        q = np.asarray(vector, dtype="float32").reshape(1, -1).copy()
        if q.shape[1] != self.index.d or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
            raise ValueError("Invalid query dimension or values")
        faiss.normalize_L2(q)
        # Full exact ranking gives stable ID-based ties, including at the cutoff.
        scores, rows = self.index.search(q, len(self.ids))
        return sorted(
            ((self.ids[i], float(s)) for i, s in zip(rows[0], scores[0]) if i >= 0),
            key=lambda x: (-x[1], x[0]),
        )[:top_k]

    def save(self, folder, metadata):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "embeddings.npy", self.vectors)
        faiss.write_index(self.index, str(folder / "vectors.faiss"))
        manifest = dict(
            metadata,
            ids=self.ids,
            dimension=self.index.d,
            checksums={
                name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                for name in ["embeddings.npy", "vectors.faiss", "papers.json"]
            },
        )
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder, expected_model=None):
        folder = Path(folder)
        meta = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if expected_model and meta["model"] != expected_model:
            raise ValueError("Index model mismatch; rebuild index")
        for name, digest in meta["checksums"].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Corrupt index asset: {name}")
        store = cls(meta["ids"], np.load(folder / "embeddings.npy", allow_pickle=False))
        store.index = faiss.read_index(str(folder / "vectors.faiss"))
        if store.index.d != meta["dimension"] or store.index.ntotal != len(store.ids):
            raise ValueError("Corrupt index dimensions")
        return store, meta
