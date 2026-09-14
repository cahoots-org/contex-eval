"""Embedder-strength ablation: does the hybrid win survive a modern dense model?

Local retrieval comparison (no Contex needed) on data/corpus.jsonl + data/questions.jsonl:

  bm25                    rank-bm25 (a conservative BM25 proxy; actual ParadeDB Contex
                          slightly *exceeded* RRF(rank-bm25, dense) in our runs)
  dense_minilm            all-MiniLM-L6-v2  (Contex's shipped embedder, 384-dim, 2021)
  dense_strong            BAAI/bge-large-en-v1.5  (1024-dim, strong modern embedder)
  hybrid_minilm           RRF(bm25, dense_minilm)   ~ Contex today
  hybrid_strong           RRF(bm25, dense_strong)   ~ Contex if it upgraded its embedder

Key questions (paired bootstrap CIs):
  1. dense_strong - dense_minilm : how much does the embedder upgrade alone buy?
  2. hybrid_minilm - dense_strong: does Contex's *cheap* hybrid still beat a *strong* dense model?
  3. hybrid_strong - dense_strong: does fusion STILL help once the dense side is strong?

Usage:  python scripts/embedder_ablation.py <k> <label>
"""
import json
import random
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

from contexeval.retrievers.bm25 import BM25Retriever
from contexeval.scoring.retrieval import retrieval_prf

STRONG_MODEL = "BAAI/bge-large-en-v1.5"
# bge-*-en-v1.5 retrieval convention: instruct the QUERY, leave documents bare.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RRF_K = 60
TOPN = 50  # candidate depth per retriever before fusion / truncation


def rrf(rankings, k=RRF_K):
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]


def dense_rankings(model, corpus, questions, query_instruction=""):
    ids = [c["para_id"] for c in corpus]
    doc_emb = model.encode([c["title"] + " " + c["text"] for c in corpus],
                           normalize_embeddings=True, convert_to_numpy=True,
                           batch_size=64, show_progress_bar=False)
    out = []
    for q in questions:
        qe = model.encode([query_instruction + q["question"]],
                          normalize_embeddings=True, convert_to_numpy=True)[0]
        order = np.argsort(-(doc_emb @ qe))[:TOPN]
        out.append([ids[i] for i in order])
    return out


def boot_ci(diffs, n=10000, seed=13):
    rnd = random.Random(seed)
    means = []
    for _ in range(n):
        means.append(sum(diffs[rnd.randrange(len(diffs))] for _ in diffs) / len(diffs))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main(k, label):
    corpus = [json.loads(l) for l in open("data/corpus.jsonl")]
    qs = [json.loads(l) for l in open("data/questions.jsonl")]

    bm = BM25Retriever(corpus, k=TOPN)
    bm_rank = [bm.retrieve(q["question"]).para_ids for q in qs]
    mini_rank = dense_rankings(SentenceTransformer("all-MiniLM-L6-v2", device="cpu"), corpus, qs)
    strong_rank = dense_rankings(SentenceTransformer(STRONG_MODEL, device="cpu"), corpus, qs,
                                 query_instruction=BGE_QUERY_INSTRUCTION)

    def recall(rankings):
        return [retrieval_prf(r[:k], q["gold_para_ids"])[1] for r, q in zip(rankings, qs)]

    methods = {
        "bm25": recall(bm_rank),
        "dense_minilm": recall(mini_rank),
        "dense_strong(bge-large)": recall(strong_rank),
        "hybrid_minilm(~contex now)": recall([rrf([b, m]) for b, m in zip(bm_rank, mini_rank)]),
        "hybrid_strong(~contex upgraded)": recall([rrf([b, s]) for b, s in zip(bm_rank, strong_rank)]),
    }
    n = len(qs)
    print(f"\n### {label}  (n={n}, recall@{k}) ###")
    for name, v in methods.items():
        print(f"  {name:34s} {sum(v)/n:.3f}")

    def compare(a, b):
        diffs = [x - y for x, y in zip(methods[a], methods[b])]
        lo, hi = boot_ci(diffs)
        mean = sum(diffs) / n
        w = sum(1 for d in diffs if d > 0)
        l = sum(1 for d in diffs if d < 0)
        sig = "  *excludes 0*" if not (lo <= 0 <= hi) else ""
        print(f"  {a}  -  {b}:  {mean:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  W/T/L={w}/{n-w-l}/{l}{sig}")

    print("  -- key comparisons --")
    compare("dense_strong(bge-large)", "dense_minilm")
    compare("hybrid_minilm(~contex now)", "dense_strong(bge-large)")
    compare("hybrid_strong(~contex upgraded)", "dense_strong(bge-large)")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
