import json
from .config import CONTEXT_BUDGET
from .scoring.retrieval import retrieval_prf
from .scoring.answer import answer_em, answer_f1

def run_one(question: dict, retriever, agent, budget: int = CONTEXT_BUDGET) -> dict:
    res = retriever.retrieve(question["question"])
    p, r, f = retrieval_prf(res.para_ids, question["gold_para_ids"])
    rec = {
        "qid": question["qid"], "method": retriever.name,
        "para_ids": res.para_ids, "bundle_size": res.bundle_size,
        "context_tokens": res.context_tokens,
        "precision": p, "recall": r, "f1": f,
        "prompt_tokens": None, "completion_tokens": None,
        "answer": None, "em": None, "answer_f1": None,
        "feasible": res.context_tokens <= budget,
    }
    if rec["feasible"]:
        ans = agent.answer(res.context_text, question["question"])
        rec.update(
            answer=ans.text,
            prompt_tokens=ans.prompt_tokens, completion_tokens=ans.completion_tokens,
            em=answer_em(ans.text, question["answer"]),
            answer_f1=answer_f1(ans.text, question["answer"]),
        )
    return rec

def run(questions, retrievers, agent, results_path, budget: int = CONTEXT_BUDGET, cache=None):
    cache = {} if cache is None else cache
    records = []
    with open(results_path, "w") as out:
        for q in questions:
            for retr in retrievers:
                key = (retr.name, q["qid"])
                if key in cache:
                    rec = cache[key]
                else:
                    try:
                        rec = run_one(q, retr, agent, budget)
                    except Exception as e:  # one failure skips, never kills the run
                        rec = {"qid": q["qid"], "method": retr.name, "error": str(e)}
                    cache[key] = rec
                records.append(rec)
                out.write(json.dumps(rec) + "\n")
    return records
