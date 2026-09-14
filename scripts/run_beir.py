"""Keyword-regime retrieval eval (e.g. BEIR SciFact): Contex vs dense vs BM25, recall@k.

Isolates Contex's hybrid contribution over the SAME embeddings used plainly: publishes a
BEIR corpus to a dedicated Contex project, then compares recall@k (and context-token cost)
with paired bootstrap CIs. No agent — BEIR has relevance labels but no gold answers.

Prep the data first, e.g.:
    python -m contexeval.prep_beir scifact test              # full corpus, all test queries
    python -m contexeval.prep_beir scifact test 50 500       # 50 queries, ~500-doc pool (trial)
Then:
    python scripts/run_beir.py 10 scifact                    # k=10, project 'scifact' (publishes)
    python scripts/run_beir.py 10 scifact nopublish          # reuse already-published corpus
"""
import json
import random
import sys

from contexeval import config
from contexeval.contex_client import ContexClient
from contexeval.retrievers.contex import ContexRetriever
from contexeval.retrievers.bm25 import BM25Retriever
from contexeval.retrievers.dense import DenseRetriever
from contexeval.scoring.retrieval import retrieval_prf

CONTEX_THRESHOLD = 0.0  # no-op under hybrid; top_k bounds the bundle


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def eval_method(retriever, questions):
    """Return (per-question recall list, per-question context-token list)."""
    recalls, toks = [], []
    for q in questions:
        res = retriever.retrieve(q["question"])
        _, r, _ = retrieval_prf(res.para_ids, q["gold_para_ids"])
        recalls.append(r)
        toks.append(res.context_tokens)
    return recalls, toks


def boot_ci(diffs, n=10000, seed=13):
    if not diffs:
        return (None, None)
    rnd = random.Random(seed)
    means = []
    for _ in range(n):
        s = [diffs[rnd.randrange(len(diffs))] for _ in diffs]
        means.append(sum(s) / len(s))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main(k=10, project="scifact", publish=True):
    corpus = load_jsonl(config.CORPUS_PATH)
    questions = load_jsonl(config.QUESTIONS_PATH)
    print(f"corpus={len(corpus)} docs, queries={len(questions)}, k={k}, project={project}")

    client = ContexClient(project_id=project)
    if publish:
        print("publishing corpus to Contex (one-time; slow under emulation)...")
        client.publish_corpus(corpus)

    methods = {
        "contex": ContexRetriever(corpus, client, threshold=CONTEX_THRESHOLD, top_k=k),
        "bm25": BM25Retriever(corpus, k=k),
        "dense": DenseRetriever(corpus, k=k),
    }
    per = {name: eval_method(r, questions) for name, r in methods.items()}

    print(f"\nrecall@{k} (mean) and mean context tokens:")
    for name, (rec, tok) in per.items():
        print(f"  {name:7s} recall={sum(rec)/len(rec):.3f}  ctx_tokens={sum(tok)/len(tok):.0f}")

    print("\nPaired bootstrap 95% CI on (Contex - baseline) recall, 10k resamples:")
    crec = per["contex"][0]
    for base in ["dense", "bm25"]:
        brec = per[base][0]
        diffs = [c - b for c, b in zip(crec, brec)]
        mean = sum(diffs) / len(diffs)
        lo, hi = boot_ci(diffs)
        wins = sum(1 for d in diffs if d > 0)
        losses = sum(1 for d in diffs if d < 0)
        ties = len(diffs) - wins - losses
        sig = "" if (lo <= 0 <= hi) else "  *CI excludes 0*"
        print(f"  contex-{base:5s}: mean={mean:+.3f} 95%CI=[{lo:+.3f},{hi:+.3f}] "
              f"W/T/L={wins}/{ties}/{losses} n={len(diffs)}{sig}")


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    project = sys.argv[2] if len(sys.argv) > 2 else "scifact"
    publish = not (len(sys.argv) > 3 and sys.argv[3] == "nopublish")
    main(k=k, project=project, publish=publish)
