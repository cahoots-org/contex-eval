# tests/test_retrievers_offline.py
from contexeval.retrievers.base import assemble_context, corpus_index
from contexeval.retrievers.dumpall import DumpAllRetriever
from contexeval.retrievers.bm25 import BM25Retriever

CORPUS = [
    {"para_id": "cats", "title": "Cats", "text": "Cats are small feline animals."},
    {"para_id": "dogs", "title": "Dogs", "text": "Dogs are loyal canine animals."},
    {"para_id": "cars", "title": "Cars", "text": "Cars are motor vehicles with wheels."},
]

def test_assemble_context_includes_titles_and_text():
    ctx = assemble_context(["cats"], corpus_index(CORPUS))
    assert "Cats" in ctx and "feline" in ctx

def test_dumpall_returns_everything():
    r = DumpAllRetriever(CORPUS).retrieve("anything")
    assert set(r.para_ids) == {"cats", "dogs", "cars"}
    assert r.context_tokens > 0

def test_bm25_ranks_relevant_first():
    r = BM25Retriever(CORPUS, k=1).retrieve("feline animal")
    assert r.para_ids == ["cats"]
    assert r.bundle_size == 1
