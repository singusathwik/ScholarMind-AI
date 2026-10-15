from ml.retrieval.dense import RERANKER, configure_threads


class Reranker:
    def __init__(self, model=RERANKER, revision=None):
        from sentence_transformers import CrossEncoder

        configure_threads()
        self.model = CrossEncoder(model, revision=revision, device="cpu", max_length=512)

    def rank(self, query, candidates, papers, representation="both"):
        if not candidates:
            return []
        scores = self.model.predict(
            [(query, papers[pid].text(representation)) for pid, _ in candidates],
            batch_size=16,
            show_progress_bar=False,
        )
        return sorted(
            [(pid, float(score)) for (pid, _), score in zip(candidates, scores)], key=lambda x: (-x[1], x[0])
        )
