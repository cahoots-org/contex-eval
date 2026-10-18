import os
import numpy as np
from sentence_transformers import SentenceTransformer
from ..config import EMBED_MODEL
from .base import corpus_index, make_result

class DenseRetriever:
    name = "dense"
    def __init__(self, corpus: list[dict], k: int, model=None):
        self.k = k
        self.by_id = corpus_index(corpus)
        self.ids = [c["para_id"] for c in corpus]
        # CONTEXEVAL_DEVICE lets us pin CPU (e.g. large BERT models hang on Apple MPS).
        self.model = model or SentenceTransformer(EMBED_MODEL, device=os.getenv("CONTEXEVAL_DEVICE") or None)
        self.emb = self.model.encode(
            [c["title"] + " " + c["text"] for c in corpus],
            normalize_embeddings=True, convert_to_numpy=True,
        )
    def retrieve(self, question: str):
        q = self.model.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self.emb @ q
        order = np.argsort(-scores)[: self.k]
        return make_result([self.ids[i] for i in order], self.by_id)
