"""Pilot/full run: prep -> publish -> measure Contex bundle size -> set baseline k -> run -> report.

RRF score note: Contex runs with hybrid search enabled, so `contex_query` returns Reciprocal-Rank-
Fusion (RRF) fused similarity values (~0.016 = 1/(60+1)), NOT cosine similarity in [0,1].
Setting CONTEX_THRESHOLD=0.0 lets all matches through; HIGH_TOPK (100) bounds the bundle.
For the PR sweep we use RRF-scale thresholds: [0.0, 0.005, 0.01, 0.02, 0.05].
"""
import json
import sys
from contexeval import config, prep
from contexeval.contex_client import ContexClient
from contexeval.retrievers.contex import ContexRetriever
from contexeval.retrievers.dumpall import DumpAllRetriever
from contexeval.retrievers.bm25 import BM25Retriever
from contexeval.retrievers.dense import DenseRetriever
from contexeval.agent import AnswerAgent
from contexeval.runner import run
from contexeval.report import aggregate, render_table, pr_curve

# Contex uses RRF similarity (~0.016 per match); cosine-oriented thresholds (e.g. 0.5) would
# filter out ALL results.  Use 0.0 here so HIGH_TOPK (100) bounds the bundle instead.
CONTEX_THRESHOLD = 0.0

# RRF-scale thresholds for the Contex PR sweep
CONTEX_THRESHOLDS = [0.0, 0.005, 0.01, 0.02, 0.05]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main(n: int, mode: str):
    prep.main(n)  # writes corpus.jsonl + questions.jsonl
    corpus = load_jsonl(config.CORPUS_PATH)
    questions = load_jsonl(config.QUESTIONS_PATH)

    client = ContexClient()
    client.publish_corpus(corpus)  # publish once

    # Contex at RRF-safe threshold; measure average bundle size B.
    contex = ContexRetriever(corpus, client, threshold=CONTEX_THRESHOLD)
    sizes = [contex.retrieve(q["question"]).bundle_size for q in questions]
    B = max(1, round(sum(sizes) / len(sizes)))
    print(f"Contex avg bundle size B={B}; setting baseline k={B}")

    retrievers = [contex, BM25Retriever(corpus, k=B), DenseRetriever(corpus, k=B)]
    if mode == "pilot":
        retrievers.append(DumpAllRetriever(corpus))  # full dump-all only when it fits

    agent = AnswerAgent()
    agent.warmup()
    records = run(questions, retrievers, agent, config.RESULTS_PATH)

    agg = aggregate(records)
    table = render_table(agg)
    (config.DATA_DIR / "report.md").write_text(table + "\n")
    print(table)

    # PR curve: sweep Contex thresholds (RRF-scale) and baseline k for BM25/dense.
    sweeps = {}
    ks = sorted({max(1, round(B * m)) for m in (0.5, 1, 2, 4)})
    for t in CONTEX_THRESHOLDS:
        recs = [_run_one_safe(q, ContexRetriever(corpus, client, threshold=t)) for q in questions]
        sweeps.setdefault("contex", []).append(_avg_pr(recs))
    for k in ks:
        for name, R in (("bm25", BM25Retriever), ("dense", DenseRetriever)):
            recs = [_run_one_safe(q, R(corpus, k=k)) for q in questions]
            sweeps.setdefault(name, []).append(_avg_pr(recs))
    pr_curve(sweeps, config.DATA_DIR / "pr_curve.png")
    print("wrote data/report.md and data/pr_curve.png")


def _run_one_safe(q, retriever):
    from contexeval.scoring.retrieval import retrieval_prf
    res = retriever.retrieve(q["question"])
    return retrieval_prf(res.para_ids, q["gold_para_ids"])


def _avg_pr(prf_list):
    rec = sum(r for _, r, _ in prf_list) / len(prf_list)
    prec = sum(p for p, _, _ in prf_list) / len(prf_list)
    return (rec, prec)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    mode = sys.argv[2] if len(sys.argv) > 2 else "pilot"  # "pilot" | "full"
    main(n, mode)
