import re

from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"\w+", text.casefold())


class BM25:
    def __init__(self, papers, representation="both"):
        self.ids = [p.paper_id for p in papers]
        self.model = BM25Okapi([tokenize(p.text(representation)) or [""] for p in papers])

    def search(self, query, top_k):
        tokens = tokenize(query)
        if not tokens or not any(t in self.model.idf for t in tokens):
            return []
        scores = self.model.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.ids[i]))
        return [(self.ids[i], float(scores[i])) for i in order[:top_k]]
