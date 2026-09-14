import re
from rank_bm25 import BM25Okapi
from .base import corpus_index, make_result

def _tok(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", t.lower())

class BM25Retriever:
    name = "bm25"
    def __init__(self, corpus: list[dict], k: int):
        self.k = k
        self.by_id = corpus_index(corpus)
        self.ids = [c["para_id"] for c in corpus]
        self.bm25 = BM25Okapi([_tok(c["title"] + " " + c["text"]) for c in corpus])
    def retrieve(self, question: str):
        scores = self.bm25.get_scores(_tok(question))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k]
        return make_result([self.ids[i] for i in order], self.by_id)
