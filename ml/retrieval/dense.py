import os

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def configure_threads():
    import torch

    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))


class Encoder:
    def __init__(self, model=MODEL, revision=None):
        from sentence_transformers import SentenceTransformer

        configure_threads()
        self.model = SentenceTransformer(model, revision=revision, device="cpu")

    def encode(self, texts, progress=False):
        return np.asarray(
            self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=progress),
            dtype="float32",
        )
