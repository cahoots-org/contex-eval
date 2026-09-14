from ..config import HIGH_TOPK
from .base import corpus_index, make_result

class ContexRetriever:
    name = "contex"
    def __init__(self, corpus: list[dict], client, threshold: float, top_k: int = HIGH_TOPK):
        self.by_id = corpus_index(corpus)
        self.client = client
        self.threshold = threshold
        self.top_k = top_k
    def retrieve(self, question: str):
        hits = self.client.query(question, top_k=self.top_k, threshold=self.threshold)
        pids = [pid for pid, _ in hits if pid in self.by_id]
        return make_result(pids, self.by_id)
