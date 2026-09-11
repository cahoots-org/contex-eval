from contexeval.runner import run_one
from contexeval.retrievers.base import RetrievalResult

class StubRetriever:
    name = "stub"
    def __init__(self, pids, tokens): self._pids, self._tokens = pids, tokens
    def retrieve(self, q):
        return RetrievalResult(self._pids, "CONTEXT", self._tokens)

class StubAgent:
    def answer(self, ctx, q):
        from contexeval.agent import AnswerResult
        return AnswerResult(text="ragtime", prompt_tokens=42, completion_tokens=3)

Q = {"qid": "q1", "question": "genre?", "answer": "ragtime", "gold_para_ids": ["a", "b"]}

def test_run_one_scores_retrieval_and_answer():
    rec = run_one(Q, StubRetriever(["a", "x"], 100), StubAgent(), budget=28000)
    assert rec["feasible"] is True
    assert rec["recall"] == 0.5 and rec["precision"] == 0.5
    assert rec["em"] == 1.0 and rec["answer_f1"] == 1.0
    assert rec["prompt_tokens"] == 42

def test_run_one_marks_infeasible_over_budget_and_skips_agent():
    rec = run_one(Q, StubRetriever(["a", "b"], 999999), StubAgent(), budget=28000)
    assert rec["feasible"] is False
    assert rec["answer"] is None and rec["em"] is None
    assert rec["answer_f1"] is None
    assert rec["prompt_tokens"] is None
    assert rec["completion_tokens"] is None
    assert rec["recall"] == 1.0                     # retrieval still scored
    assert rec["context_tokens"] == 999999          # projected cost recorded
