import json
import re
from .config import SEED, CORPUS_PATH, QUESTIONS_PATH, DATA_DIR

def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")

def pool_examples(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    corpus: dict[str, dict] = {}
    questions: list[dict] = []
    for ex in examples:
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        for title, sents in zip(titles, sentences):
            pid = slug(title)
            if pid not in corpus:
                corpus[pid] = {"para_id": pid, "title": title, "text": " ".join(sents)}
        gold = []
        for t in ex["supporting_facts"]["title"]:
            pid = slug(t)
            if pid not in gold:
                gold.append(pid)
        questions.append({
            "qid": ex["id"], "question": ex["question"],
            "answer": ex["answer"], "gold_para_ids": gold,
        })
    return list(corpus.values()), questions

def main(n: int, out_corpus=CORPUS_PATH, out_questions=QUESTIONS_PATH):
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    ds = ds.shuffle(seed=SEED).select(range(n))
    corpus, questions = pool_examples(list(ds))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_corpus, "w") as f:
        for c in corpus: f.write(json.dumps(c) + "\n")
    with open(out_questions, "w") as f:
        for q in questions: f.write(json.dumps(q) + "\n")
    print(f"wrote {len(corpus)} paragraphs, {len(questions)} questions")

if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
