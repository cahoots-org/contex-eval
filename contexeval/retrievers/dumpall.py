from .base import corpus_index, make_result

class DumpAllRetriever:
    name = "dump-all"
    def __init__(self, corpus: list[dict]):
        self.by_id = corpus_index(corpus)
        self.pids = [c["para_id"] for c in corpus]
    def retrieve(self, question: str):
        return make_result(self.pids, self.by_id)
