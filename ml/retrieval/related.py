"""Paper-to-paper ranking using persisted dense vectors and full-text lexical similarity."""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.retrieval.hybrid import rrf


class RelatedRecommender:
    def __init__(self, papers, store):
        self.papers = papers
        self.store = store
        self.rows = {pid: i for i, pid in enumerate(store.ids)}
        self.lexical = None

    def fit_lexical(self):
        if self.lexical is None:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True,
                max_features=100000,
                dtype=np.float32,
            )
            try:
                self.lexical = vectorizer.fit_transform([self.papers[pid].text() for pid in self.store.ids])
            except ValueError as exc:
                if "empty vocabulary" not in str(exc):
                    raise
                self.lexical = csr_matrix((len(self.store.ids), 1), dtype=np.float32)

    def components(self, pid, depth=100, candidates=None):
        if pid not in self.rows:
            raise KeyError(pid)
        if depth < 1:
            raise ValueError("Recommendation depth must be positive")
        row = self.rows[pid]
        self.fit_lexical()
        if candidates is None:
            dense = [(p, s) for p, s in self.store.search(self.store.vectors[row], depth + 1) if p != pid][
                :depth
            ]
            ids = [p for p in self.store.ids if p != pid]
        else:
            if len(candidates) != len(set(candidates)) or any(p not in self.rows for p in candidates):
                raise ValueError("Candidate IDs must be unique and present in the index")
            ids = [p for p in candidates if p != pid]
            scores = self.store.vectors[[self.rows[p] for p in ids]] @ self.store.vectors[row]
            dense = sorted(zip(ids, map(float, scores)), key=lambda x: (-x[1], x[0]))[:depth]
        rows = [self.rows[p] for p in ids]
        similarities = (self.lexical[rows] @ self.lexical[row].T).toarray().ravel()
        lexical = sorted(
            ((p, float(s)) for p, s in zip(ids, similarities) if s > 0), key=lambda x: (-x[1], x[0])
        )[:depth]
        return dense, lexical

    def rank(self, pid, top_k=5, profile=None, candidates=None):
        if top_k < 1:
            raise ValueError("Recommendation count must be positive")
        if pid not in self.rows:
            raise KeyError(pid)
        if profile is None and candidates is None:
            row = self.rows[pid]
            return [(p, s) for p, s in self.store.search(self.store.vectors[row], top_k + 1) if p != pid][
                :top_k
            ]
        dense, lexical = self.components(pid, max(top_k, 100), candidates)
        if profile is None or profile.get("method") == "dense":
            return dense[:top_k]
        return rrf([dense, lexical], profile["rrf_constant"], top_k, [1, profile["lexical_weight"]])
