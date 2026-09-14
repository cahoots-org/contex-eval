"""Convert a BEIR retrieval dataset (HuggingFace) into the harness corpus/questions format.

BEIR provides relevance labels (qrels) but NO gold answer strings, so questions get
`answer=""` and are meant to be scored on RETRIEVAL ONLY (recall@k) — see scripts/run_beir.py.
This complements prep.py (HotpotQA, which has gold answers for the full 3-metric run).

BEIR HF layout: `BeIR/<name>` with configs `corpus` (_id,title,text) and `queries`
(_id,title,text); `BeIR/<name>-qrels` with rows (query-id, corpus-id, score).
"""
import json
import random
import re

from .config import DATA_DIR, CORPUS_PATH, QUESTIONS_PATH, SEED


def pid(x) -> str:
    """Stable, dot-free id (Contex maps a match back via data_key.split('.',1)[0])."""
    return re.sub(r"[^a-z0-9]+", "-", str(x).lower()).strip("-")


def build(corpus_rows, queries_rows, qrels_rows, max_queries=None, max_corpus=None, seed=SEED):
    """Pure conversion. Rows are dicts (BEIR fields). Returns (corpus, questions).

    - max_queries: sample this many queries (fixed seed); None = all labeled queries.
    - max_corpus: pool = all needed gold docs + random distractors up to this many;
      None = full corpus (standard BEIR). Pooling mirrors the HotpotQA methodology.
    """
    gold = {}
    for row in qrels_rows:
        if int(row["score"]) > 0:
            gold.setdefault(str(row["query-id"]), set()).add(pid(row["corpus-id"]))
    qtext = {str(q["_id"]): q["text"] for q in queries_rows}

    qids = sorted(q for q in gold if q in qtext and gold[q])
    if max_queries is not None and max_queries < len(qids):
        r = random.Random(seed)
        qids = sorted(r.sample(qids, max_queries))
    questions = [{"qid": q, "question": qtext[q], "answer": "",
                  "gold_para_ids": sorted(gold[q])} for q in qids]

    all_docs = {}
    for c in corpus_rows:
        k = pid(c["_id"])
        all_docs[k] = {"para_id": k, "title": c.get("title") or "", "text": c["text"]}

    needed_gold = set()
    for q in questions:
        needed_gold |= set(q["gold_para_ids"])

    if max_corpus is None:
        keep = set(all_docs)
    else:
        r = random.Random(seed)
        distractors = [k for k in all_docs if k not in needed_gold]
        r.shuffle(distractors)
        n_dist = max(0, max_corpus - len(needed_gold & set(all_docs)))
        keep = (needed_gold & set(all_docs)) | set(distractors[:n_dist])

    corpus = [all_docs[k] for k in all_docs if k in keep]

    # Drop gold ids missing from the (possibly pooled) corpus; drop now-empty questions.
    present = {c["para_id"] for c in corpus}
    for q in questions:
        q["gold_para_ids"] = [g for g in q["gold_para_ids"] if g in present]
    questions = [q for q in questions if q["gold_para_ids"]]
    return corpus, questions


def main(name="scifact", split="test", max_queries=None, max_corpus=None,
         out_corpus=CORPUS_PATH, out_questions=QUESTIONS_PATH):
    from datasets import load_dataset
    corpus_rows = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    queries_rows = load_dataset(f"BeIR/{name}", "queries", split="queries")
    qrels_rows = load_dataset(f"BeIR/{name}-qrels", split=split)
    corpus, questions = build(list(corpus_rows), list(queries_rows), list(qrels_rows),
                              max_queries=max_queries, max_corpus=max_corpus)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_corpus, "w") as f:
        for c in corpus:
            f.write(json.dumps(c) + "\n")
    with open(out_questions, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
    print(f"BEIR {name}/{split}: wrote {len(corpus)} docs, {len(questions)} queries")


if __name__ == "__main__":
    import sys
    kw = {}
    if len(sys.argv) > 1: kw["name"] = sys.argv[1]
    if len(sys.argv) > 2: kw["split"] = sys.argv[2]
    if len(sys.argv) > 3: kw["max_queries"] = int(sys.argv[3])
    if len(sys.argv) > 4: kw["max_corpus"] = int(sys.argv[4])
    main(**kw)
