# Contex Validation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a benchmark harness that measures whether Contex routes leaner, sufficient context to an agent better than dump-all / BM25 / dense retrieval, on pooled HotpotQA, across three metrics: retrieval P/R/F1, answer EM/F1, and tokens.

**Architecture:** A plain Python package drives three local processes — Contex (docker-compose, reached over MCP), an mlx-lm agent server (OpenAI-compatible), and in-process BM25/dense indexes. Dataset prep pools HotpotQA distractor paragraphs into one shared corpus; each retriever returns a context for a fixed agent; scorers produce the three metrics; a runner + report aggregate results and a PR curve.

**Tech Stack:** Python 3.12, `datasets`, `mcp` (client SDK), `openai`, `rank-bm25`, `sentence-transformers`, `transformers` (tokenizer), `numpy`, `matplotlib`, `pytest`. Contex via Docker; agent via `mlx-lm`.

**Spec:** `docs/superpowers/specs/2026-09-11-contex-validation-harness-design.md`

## Global Constraints

- **No paid APIs.** Agent is local `mlx-community/Qwen2.5-7B-Instruct-4bit` on `http://localhost:8080/v1`. Embeddings are local `all-MiniLM-L6-v2`. Contex embeds locally too.
- **Contex is unforked.** Cloned/pinned checkout, run via its own `docker compose up -d`; config `AUTH_ENABLED=false`, `HYBRID_SEARCH_ENABLED=true`, `SIMILARITY_THRESHOLD=0.5`; MCP at `http://localhost:8001/mcp`.
- **Agent held constant** across every condition; only the retrieval method varies. Temperature 0, `max_tokens=256`. Pre-warm before timing/counting.
- **Fixed random seed** = `13` for all sampling.
- **Retrieval unit = one paragraph = one Contex node.** `para_id = slug(title)` (dot-free). Gold = the 2 supporting-fact titles.
- **Contex "auto-sizing":** call `contex_query` with `top_k = 100` (`HIGH_TOPK`) so the **threshold** is the binding constraint, not k.
- **Context budget** for dump-all feasibility = `28000` tokens (`CONTEXT_BUDGET`); the agent's window is treated as 32k.
- **Metric granularity:** retrieval scored at paragraph level against the 2 gold titles (documented scoping choice). Answer EM/F1 via the official HotpotQA normalizer.
- **Package name:** `contexeval`. Artifacts under `data/` (git-ignored). Tests under `tests/`.

---

## File Structure

- `contexeval/config.py` — constants (ports, model, seed, thresholds, paths, budgets).
- `contexeval/scoring/answer.py` — official HotpotQA `normalize_answer`, `answer_em`, `answer_f1`.
- `contexeval/scoring/retrieval.py` — `retrieval_prf`.
- `contexeval/tokens.py` — `count_tokens` (Qwen tokenizer).
- `contexeval/prep.py` — pure pooling/dedup functions + `main()` that loads HotpotQA and writes `corpus.jsonl` / `questions.jsonl`.
- `contexeval/retrievers/base.py` — `RetrievalResult`, `assemble_context`, `Retriever` protocol.
- `contexeval/retrievers/dumpall.py`, `bm25.py`, `dense.py`, `contex.py`.
- `contexeval/contex_client.py` — sync MCP client wrapper (`publish_corpus`, `query`).
- `contexeval/agent.py` — `build_prompt`, `AnswerAgent`, `AnswerResult`.
- `contexeval/runner.py` — orchestration + answer cache; writes `results.jsonl`.
- `contexeval/report.py` — aggregate table + PR-curve plot.
- `scripts/run.py` — pilot/full orchestration (measure B → set baseline k → run → report).
- `tests/…` — one test module per unit above; integration tests marked `@pytest.mark.integration`.

Interfaces other tasks rely on (defined once, used verbatim):

```python
# retrievers/base.py
@dataclass
class RetrievalResult:
    para_ids: list[str]
    context_text: str
    context_tokens: int
    @property
    def bundle_size(self) -> int: return len(self.para_ids)

# agent.py
@dataclass
class AnswerResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
```

A **corpus item** is `{"para_id": str, "title": str, "text": str}`.
A **question item** is `{"qid": str, "question": str, "answer": str, "gold_para_ids": list[str]}`.

---

### Task 1: Project scaffolding + config

**Files:**
- Create: `pyproject.toml`, `contexeval/__init__.py`, `contexeval/config.py`, `contexeval/scoring/__init__.py`, `contexeval/retrievers/__init__.py`, `tests/__init__.py`, `tests/test_config.py`, `pytest.ini`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `config` constants used everywhere — `SEED=13`, `HIGH_TOPK=100`, `CONTEXT_BUDGET=28000`, `DEFAULT_THRESHOLD=0.5`, `AGENT_MODEL`, `AGENT_BASE_URL`, `CONTEX_MCP_URL`, `CONTEX_PROJECT_ID`, `EMBED_MODEL`, `DATA_DIR`, `CORPUS_PATH`, `QUESTIONS_PATH`, `RESULTS_PATH`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from contexeval import config

def test_core_constants_present():
    assert config.SEED == 13
    assert config.HIGH_TOPK == 100
    assert config.CONTEXT_BUDGET == 28000
    assert config.DEFAULT_THRESHOLD == 0.5
    assert config.AGENT_BASE_URL.endswith("/v1")
    assert config.CONTEX_MCP_URL.endswith("/mcp")
    assert config.EMBED_MODEL == "all-MiniLM-L6-v2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError: contexeval`).

- [ ] **Step 3: Create packaging + config**

```toml
# pyproject.toml
[project]
name = "contexeval"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "datasets", "mcp", "openai", "rank-bm25",
  "sentence-transformers", "transformers", "numpy", "matplotlib",
]
[project.optional-dependencies]
dev = ["pytest"]
[tool.setuptools.packages.find]
include = ["contexeval*"]
```

```ini
# pytest.ini
[pytest]
markers =
    integration: requires live Contex and/or mlx-lm servers
addopts = -m "not integration"
```

```python
# contexeval/config.py
from pathlib import Path

SEED = 13
HIGH_TOPK = 100
CONTEXT_BUDGET = 28000
DEFAULT_THRESHOLD = 0.5

AGENT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
AGENT_BASE_URL = "http://localhost:8080/v1"
TOKENIZER_MODEL = "Qwen/Qwen2.5-7B-Instruct"

CONTEX_MCP_URL = "http://localhost:8001/mcp"
CONTEX_PROJECT_ID = "hotpotqa"
EMBED_MODEL = "all-MiniLM-L6-v2"

DATA_DIR = Path("data")
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUESTIONS_PATH = DATA_DIR / "questions.jsonl"
RESULTS_PATH = DATA_DIR / "results.jsonl"
```

Create empty `contexeval/__init__.py`, `contexeval/scoring/__init__.py`, `contexeval/retrievers/__init__.py`, `tests/__init__.py`.

- [ ] **Step 4: Install and run the test**

Run: `pip install -e ".[dev]" && pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml pytest.ini contexeval tests
git commit -m "chore: scaffold contexeval package and config"
```

---

### Task 2: Answer scoring (official HotpotQA EM/F1)

**Files:**
- Create: `contexeval/scoring/answer.py`
- Test: `tests/test_scoring_answer.py`

**Interfaces:**
- Produces: `normalize_answer(s: str) -> str`, `answer_em(pred: str, gold: str) -> float`, `answer_f1(pred: str, gold: str) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scoring_answer.py
from contexeval.scoring.answer import normalize_answer, answer_em, answer_f1

def test_normalize_strips_articles_punct_case():
    assert normalize_answer("The  Beatles!") == "beatles"

def test_em_exact_after_normalization():
    assert answer_em("the beatles", "Beatles") == 1.0
    assert answer_em("Rolling Stones", "Beatles") == 0.0

def test_f1_partial_overlap():
    # pred "Arthur Conan Doyle" vs gold "Conan Doyle" -> P=2/3, R=1, F1=0.8
    assert abs(answer_f1("Arthur Conan Doyle", "Conan Doyle") - 0.8) < 1e-6

def test_f1_yesno_mismatch_is_zero():
    assert answer_f1("yes", "no") == 0.0
    assert answer_f1("the president", "yes") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring_answer.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement (vendored from official `hotpot_evaluate_v1.py`)**

```python
# contexeval/scoring/answer.py
import re
import string
from collections import Counter

def normalize_answer(s: str) -> str:
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t): return " ".join(t.split())
    def remove_punc(t): return "".join(ch for ch in t if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def answer_em(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))

def answer_f1(pred: str, gold: str) -> float:
    npred, ngold = normalize_answer(pred), normalize_answer(gold)
    special = {"yes", "no", "noanswer"}
    if ngold in special and npred != ngold: return 0.0
    if npred in special and npred != ngold: return 0.0
    pt, gt = npred.split(), ngold.split()
    common = Counter(pt) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(gt)
    return 2 * precision * recall / (precision + recall)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scoring_answer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add contexeval/scoring/answer.py tests/test_scoring_answer.py
git commit -m "feat: official HotpotQA answer EM/F1 scoring"
```

---

### Task 3: Retrieval scoring (paragraph P/R/F1)

**Files:**
- Create: `contexeval/scoring/retrieval.py`
- Test: `tests/test_scoring_retrieval.py`

**Interfaces:**
- Produces: `retrieval_prf(retrieved: list[str], gold: list[str]) -> tuple[float, float, float]` returning `(precision, recall, f1)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scoring_retrieval.py
from contexeval.scoring.retrieval import retrieval_prf

def test_perfect_retrieval():
    assert retrieval_prf(["a", "b"], ["a", "b"]) == (1.0, 1.0, 1.0)

def test_one_gold_among_extras():
    # retrieved 4, one is gold; gold set size 2 -> P=1/4, R=1/2
    p, r, f = retrieval_prf(["a", "x", "y", "z"], ["a", "b"])
    assert abs(p - 0.25) < 1e-9 and abs(r - 0.5) < 1e-9
    assert abs(f - (2 * 0.25 * 0.5 / 0.75)) < 1e-9

def test_empty_retrieval():
    assert retrieval_prf([], ["a", "b"]) == (0.0, 0.0, 0.0)

def test_dedupes_retrieved():
    assert retrieval_prf(["a", "a", "b"], ["a", "b"]) == (1.0, 1.0, 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring_retrieval.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# contexeval/scoring/retrieval.py
def retrieval_prf(retrieved: list[str], gold: list[str]) -> tuple[float, float, float]:
    R, G = set(retrieved), set(gold)
    tp = len(R & G)
    precision = tp / len(R) if R else 0.0
    recall = tp / len(G) if G else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scoring_retrieval.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add contexeval/scoring/retrieval.py tests/test_scoring_retrieval.py
git commit -m "feat: paragraph-level retrieval P/R/F1"
```

---

### Task 4: Token counting

**Files:**
- Create: `contexeval/tokens.py`
- Test: `tests/test_tokens.py`

**Interfaces:**
- Produces: `count_tokens(text: str) -> int` (Qwen tokenizer, cached module-level).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tokens.py
from contexeval.tokens import count_tokens

def test_counts_are_positive_and_monotonic():
    a = count_tokens("hello world")
    b = count_tokens("hello world hello world hello world")
    assert a > 0 and b > a

def test_empty_string_is_zero():
    assert count_tokens("") == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tokens.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# contexeval/tokens.py
from functools import lru_cache
from transformers import AutoTokenizer
from .config import TOKENIZER_MODEL

@lru_cache(maxsize=1)
def _tok():
    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_tok().encode(text))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tokens.py -v`
Expected: PASS (first run downloads the tokenizer, small).

- [ ] **Step 5: Commit**

```bash
git add contexeval/tokens.py tests/test_tokens.py
git commit -m "feat: Qwen token counting helper"
```

---

### Task 5: Dataset prep — pooling & dedup (pure functions + CLI)

**Files:**
- Create: `contexeval/prep.py`
- Test: `tests/test_prep.py`

**Interfaces:**
- Produces:
  - `slug(title: str) -> str` (lowercase, non-alphanumerics → `-`, dot-free).
  - `pool_examples(examples: list[dict]) -> tuple[list[dict], list[dict]]` returning `(corpus, questions)`.
  - `main(n: int, out_corpus, out_questions)` — loads HotpotQA distractor validation, samples `n` with `SEED`, pools, writes JSONL.
- Each input example follows the HotpotQA `datasets` schema: `{"id","question","answer","supporting_facts":{"title":[...],"sent_id":[...]},"context":{"title":[...],"sentences":[[...],...]}}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prep.py
from contexeval.prep import slug, pool_examples

def _ex(qid, q, ans, titles, sentences, sup_titles):
    return {
        "id": qid, "question": q, "answer": ans,
        "supporting_facts": {"title": sup_titles, "sent_id": [0] * len(sup_titles)},
        "context": {"title": titles, "sentences": sentences},
    }

def test_slug_is_dot_free_and_normalized():
    assert slug("The Beatles!") == "the-beatles"
    assert "." not in slug("Dr. No (film)")

def test_pool_dedups_shared_titles_and_maps_gold():
    ex1 = _ex("q1", "Q1?", "A1",
              titles=["Alpha", "Beta"], sentences=[["a1.", "a2."], ["b1."]],
              sup_titles=["Alpha"])
    ex2 = _ex("q2", "Q2?", "A2",
              titles=["Beta", "Gamma"], sentences=[["b1."], ["g1."]],
              sup_titles=["Gamma"])
    corpus, questions = pool_examples([ex1, ex2])
    ids = {c["para_id"] for c in corpus}
    assert ids == {slug("Alpha"), slug("Beta"), slug("Gamma")}  # Beta deduped
    assert questions[0]["gold_para_ids"] == [slug("Alpha")]
    assert questions[1]["gold_para_ids"] == [slug("Gamma")]
    alpha = next(c for c in corpus if c["para_id"] == slug("Alpha"))
    assert alpha["text"] == "a1. a2."  # sentences joined
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prep.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# contexeval/prep.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prep.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add contexeval/prep.py tests/test_prep.py
git commit -m "feat: HotpotQA pooling/dedup prep + CLI"
```

---

### Task 6: Retriever base + dump-all + BM25

**Files:**
- Create: `contexeval/retrievers/base.py`, `contexeval/retrievers/dumpall.py`, `contexeval/retrievers/bm25.py`
- Test: `tests/test_retrievers_offline.py`

**Interfaces:**
- Produces:
  - `RetrievalResult` (fields per header), `assemble_context(pids, corpus_by_id) -> str`, `corpus_index(corpus) -> dict[str, dict]`.
  - `DumpAllRetriever(corpus)` with `.name = "dump-all"` and `.retrieve(question) -> RetrievalResult` (returns all paragraphs).
  - `BM25Retriever(corpus, k)` with `.name = "bm25"` and `.retrieve(question) -> RetrievalResult` (top-k).
- Consumes: `count_tokens` (Task 4).

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retrievers_offline.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# contexeval/retrievers/base.py
from dataclasses import dataclass
from ..tokens import count_tokens

@dataclass
class RetrievalResult:
    para_ids: list[str]
    context_text: str
    context_tokens: int
    @property
    def bundle_size(self) -> int:
        return len(self.para_ids)

def corpus_index(corpus: list[dict]) -> dict[str, dict]:
    return {c["para_id"]: c for c in corpus}

def assemble_context(pids: list[str], by_id: dict[str, dict]) -> str:
    return "\n\n".join(f"Title: {by_id[p]['title']}\n{by_id[p]['text']}" for p in pids)

def make_result(pids: list[str], by_id: dict[str, dict]) -> RetrievalResult:
    ctx = assemble_context(pids, by_id)
    return RetrievalResult(para_ids=pids, context_text=ctx, context_tokens=count_tokens(ctx))
```

```python
# contexeval/retrievers/dumpall.py
from .base import corpus_index, make_result

class DumpAllRetriever:
    name = "dump-all"
    def __init__(self, corpus: list[dict]):
        self.by_id = corpus_index(corpus)
        self.pids = [c["para_id"] for c in corpus]
    def retrieve(self, question: str):
        return make_result(self.pids, self.by_id)
```

```python
# contexeval/retrievers/bm25.py
import re
from rank_bm25 import BM25Okapi
from .base import corpus_index, make_result

def _tok(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", t.lower())

class BM25Retriever:
    name = "bm25"
    def __init__(self, corpus: list[dict], k: int):
        self.k = k
        self.by_id = corpus_index(corpus)
        self.ids = [c["para_id"] for c in corpus]
        self.bm25 = BM25Okapi([_tok(c["title"] + " " + c["text"]) for c in corpus])
    def retrieve(self, question: str):
        scores = self.bm25.get_scores(_tok(question))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k]
        return make_result([self.ids[i] for i in order], self.by_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retrievers_offline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add contexeval/retrievers/base.py contexeval/retrievers/dumpall.py contexeval/retrievers/bm25.py tests/test_retrievers_offline.py
git commit -m "feat: retriever base + dump-all + BM25"
```

---

### Task 7: Dense retriever (all-MiniLM-L6-v2)

**Files:**
- Create: `contexeval/retrievers/dense.py`
- Test: `tests/test_dense.py`

**Interfaces:**
- Produces: `DenseRetriever(corpus, k, model=None)` with `.name = "dense"` and `.retrieve(question) -> RetrievalResult`. Uses cosine over L2-normalized `all-MiniLM-L6-v2` embeddings.
- Consumes: `make_result`, `corpus_index` (Task 6).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dense.py
from contexeval.retrievers.dense import DenseRetriever

CORPUS = [
    {"para_id": "cats", "title": "Cats", "text": "Cats are small feline animals kept as pets."},
    {"para_id": "cars", "title": "Cars", "text": "Cars are motor vehicles for road transport."},
]

def test_dense_semantic_match():
    r = DenseRetriever(CORPUS, k=1).retrieve("a pet kitten")
    assert r.para_ids == ["cats"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dense.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# contexeval/retrievers/dense.py
import numpy as np
from sentence_transformers import SentenceTransformer
from ..config import EMBED_MODEL
from .base import corpus_index, make_result

class DenseRetriever:
    name = "dense"
    def __init__(self, corpus: list[dict], k: int, model=None):
        self.k = k
        self.by_id = corpus_index(corpus)
        self.ids = [c["para_id"] for c in corpus]
        self.model = model or SentenceTransformer(EMBED_MODEL)
        self.emb = self.model.encode(
            [c["title"] + " " + c["text"] for c in corpus],
            normalize_embeddings=True, convert_to_numpy=True,
        )
    def retrieve(self, question: str):
        q = self.model.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self.emb @ q
        order = np.argsort(-scores)[: self.k]
        return make_result([self.ids[i] for i in order], self.by_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dense.py -v`
Expected: PASS (first run downloads the 80MB model).

- [ ] **Step 5: Commit**

```bash
git add contexeval/retrievers/dense.py tests/test_dense.py
git commit -m "feat: dense retriever (all-MiniLM-L6-v2)"
```

---

### Task 8: Contex MCP client + live round-trip (de-risking task)

**Prerequisite:** Contex running. Clone into `./contex` (unforked), then:
`cd contex && docker compose up -d` (Postgres+pgvector, Redis, app). Confirm `curl -s localhost:8001/health` (or the documented health path) is OK and env has `AUTH_ENABLED=false`, `HYBRID_SEARCH_ENABLED=true`.

**Files:**
- Create: `contexeval/contex_client.py`
- Test: `tests/test_contex_roundtrip.py` (marked `integration`)

**Interfaces:**
- Produces: `ContexClient(url=CONTEX_MCP_URL, project_id=CONTEX_PROJECT_ID)` with:
  - `.publish_corpus(paragraphs: list[dict]) -> None` — publishes each paragraph as one node.
  - `.query(question: str, top_k: int, threshold: float) -> list[tuple[str, float]]` — returns `(para_id, similarity)` pairs.

- [ ] **Step 1: Write the failing integration test**

```python
# tests/test_contex_roundtrip.py
import uuid
import pytest
from contexeval.contex_client import ContexClient
from contexeval.config import CONTEX_MCP_URL

pytestmark = pytest.mark.integration

PARAS = [
    {"para_id": "scott-joplin", "title": "Scott Joplin", "text": "Scott Joplin was an American composer of ragtime music."},
    {"para_id": "the-entertainer", "title": "The Entertainer", "text": "The Entertainer is a 1902 piano rag by Scott Joplin."},
    {"para_id": "mount-fuji", "title": "Mount Fuji", "text": "Mount Fuji is the tallest mountain in Japan."},
    {"para_id": "python-language", "title": "Python (language)", "text": "Python is a high-level programming language."},
    {"para_id": "great-barrier-reef", "title": "Great Barrier Reef", "text": "The Great Barrier Reef is the world's largest coral reef system."},
]

def test_publish_and_query_maps_back_to_para_ids():
    project = f"rt-{uuid.uuid4().hex[:8]}"
    client = ContexClient(url=CONTEX_MCP_URL, project_id=project)
    client.publish_corpus(PARAS)
    hits = client.query("Who composed ragtime piano music?", top_k=100, threshold=0.3)
    ids = {pid for pid, _ in hits}
    assert ids <= {p["para_id"] for p in PARAS}          # only known ids, cleanly mapped
    assert "scott-joplin" in ids                          # relevant gold surfaced
    assert all(0.0 <= sim <= 1.0 for _, sim in hits)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_contex_roundtrip.py -v -m integration`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# contexeval/contex_client.py
import asyncio
import json
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from .config import CONTEX_MCP_URL, CONTEX_PROJECT_ID

class ContexClient:
    def __init__(self, url: str = CONTEX_MCP_URL, project_id: str = CONTEX_PROJECT_ID):
        self.url = url
        self.project_id = project_id

    async def _run(self, fn):
        async with streamablehttp_client(self.url) as (read, write, _):
            async with ClientSession(read, write) as s:
                await s.initialize()
                return await fn(s)

    @staticmethod
    def _text(result) -> str:
        # CallToolResult.content is a list of content blocks; the tool returns one text block.
        return result.content[0].text

    def publish_corpus(self, paragraphs: list[dict]) -> None:
        async def _pub(s):
            for p in paragraphs:
                await s.call_tool("contex_publish", {
                    "project_id": self.project_id,
                    "data_key": p["para_id"],
                    "data": {"para_id": p["para_id"], "title": p["title"], "text": p["text"]},
                    "data_format": "json",
                })
        asyncio.run(self._run(_pub))

    def query(self, question: str, top_k: int, threshold: float) -> list[tuple[str, float]]:
        async def _q(s):
            r = await s.call_tool("contex_query", {
                "project_id": self.project_id, "query": question,
                "top_k": top_k, "threshold": threshold,
            })
            return json.loads(self._text(r))
        payload = asyncio.run(self._run(_q))
        out: list[tuple[str, float]] = []
        seen: set[str] = set()
        for m in payload.get("matches", []):
            pid = m["data_key"].split(".", 1)[0]  # "<para_id>.root" -> "<para_id>"
            if pid not in seen:
                seen.add(pid)
                out.append((pid, float(m["similarity"])))
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_contex_roundtrip.py -v -m integration`
Expected: PASS. If `data_key` format differs from `"<para_id>.root"`, the assertion `ids <= known` fails loudly — fix `.split(".", 1)[0]` to match observed format (fallback: read `m["data"]["para_id"]`).

- [ ] **Step 5: Commit**

```bash
git add contexeval/contex_client.py tests/test_contex_roundtrip.py
git commit -m "feat: Contex MCP client + live round-trip test"
```

---

### Task 9: Contex retriever

**Files:**
- Create: `contexeval/retrievers/contex.py`
- Test: `tests/test_contex_retriever.py` (marked `integration`)

**Interfaces:**
- Produces: `ContexRetriever(corpus, client, threshold, top_k=HIGH_TOPK)` with `.name = "contex"` and `.retrieve(question) -> RetrievalResult`. Assumes `client.publish_corpus(corpus)` was already called by the caller (publish once, query many).
- Consumes: `ContexClient` (Task 8), `make_result`/`corpus_index` (Task 6), `HIGH_TOPK` (config).

- [ ] **Step 1: Write the failing integration test**

```python
# tests/test_contex_retriever.py
import uuid
import pytest
from contexeval.contex_client import ContexClient
from contexeval.retrievers.contex import ContexRetriever
from contexeval.config import CONTEX_MCP_URL

pytestmark = pytest.mark.integration

CORPUS = [
    {"para_id": "scott-joplin", "title": "Scott Joplin", "text": "Scott Joplin was an American composer of ragtime music."},
    {"para_id": "mount-fuji", "title": "Mount Fuji", "text": "Mount Fuji is the tallest mountain in Japan."},
]

def test_contex_retriever_returns_result_with_context():
    project = f"cr-{uuid.uuid4().hex[:8]}"
    client = ContexClient(url=CONTEX_MCP_URL, project_id=project)
    client.publish_corpus(CORPUS)
    r = ContexRetriever(CORPUS, client, threshold=0.3).retrieve("Who composed ragtime music?")
    assert "scott-joplin" in r.para_ids
    assert "Scott Joplin" in r.context_text
    assert r.context_tokens > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_contex_retriever.py -v -m integration`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# contexeval/retrievers/contex.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_contex_retriever.py -v -m integration`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add contexeval/retrievers/contex.py tests/test_contex_retriever.py
git commit -m "feat: Contex retriever over MCP"
```

---

### Task 10: Answer agent (prompt builder + mlx-lm)

**Prerequisite for the integration test:** `pip install mlx-lm` then
`mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080`.

**Files:**
- Create: `contexeval/agent.py`
- Test: `tests/test_agent.py` (prompt builder offline + one `integration` test)

**Interfaces:**
- Produces:
  - `build_prompt(context_text: str, question: str) -> list[dict]` (OpenAI messages).
  - `AnswerResult` (fields per header).
  - `AnswerAgent(base_url=AGENT_BASE_URL, model=AGENT_MODEL)` with `.warmup()` and `.answer(context_text, question) -> AnswerResult`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent.py
import pytest
from contexeval.agent import build_prompt, AnswerAgent

def test_prompt_contains_context_and_question():
    msgs = build_prompt("Title: Cats\nCats are feline.", "What are cats?")
    joined = " ".join(m["content"] for m in msgs)
    assert "Cats are feline." in joined and "What are cats?" in joined
    assert msgs[0]["role"] == "system"

@pytest.mark.integration
def test_agent_answers_from_context():
    agent = AnswerAgent()
    agent.warmup()
    res = agent.answer("Title: Scott Joplin\nScott Joplin composed ragtime music.",
                       "What kind of music did Scott Joplin compose?")
    assert "ragtime" in res.text.lower()
    assert res.prompt_tokens > 0 and res.completion_tokens > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# contexeval/agent.py
from dataclasses import dataclass
from openai import OpenAI
from .config import AGENT_BASE_URL, AGENT_MODEL

SYSTEM = (
    "You are a precise question-answering assistant. Using ONLY the provided context, "
    "answer the question with the shortest exact span or phrase. If the answer is a "
    "yes/no question, answer 'yes' or 'no'. Do not explain."
)

def build_prompt(context_text: str, question: str) -> list[dict]:
    user = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

@dataclass
class AnswerResult:
    text: str
    prompt_tokens: int
    completion_tokens: int

class AnswerAgent:
    def __init__(self, base_url: str = AGENT_BASE_URL, model: str = AGENT_MODEL):
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model

    def warmup(self) -> None:
        self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": "ok"}], max_tokens=1)

    def answer(self, context_text: str, question: str) -> AnswerResult:
        resp = self.client.chat.completions.create(
            model=self.model, messages=build_prompt(context_text, question),
            temperature=0, max_tokens=256)
        return AnswerResult(
            text=resp.choices[0].message.content.strip(),
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_agent.py -v` (offline prompt test passes), then
`pytest tests/test_agent.py -v -m integration` (with the server up).
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add contexeval/agent.py tests/test_agent.py
git commit -m "feat: fixed mlx-lm answer agent + prompt builder"
```

---

### Task 11: Runner + answer cache + infeasibility handling

**Files:**
- Create: `contexeval/runner.py`
- Test: `tests/test_runner.py` (offline, with stub retrievers/agent)

**Interfaces:**
- Produces:
  - `run_one(question: dict, retriever, agent, budget: int) -> dict` — one result record.
  - `run(questions, retrievers, agent, results_path, budget=CONTEXT_BUDGET, cache=None) -> list[dict]`.
- Result record keys: `qid, method, para_ids, bundle_size, context_tokens, prompt_tokens, completion_tokens, precision, recall, f1, answer, em, answer_f1, feasible`.
- Consumes: `retrieval_prf` (Task 3), `answer_em`/`answer_f1` (Task 2), `CONTEXT_BUDGET` (config).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner.py
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
    assert rec["recall"] == 1.0                     # retrieval still scored
    assert rec["context_tokens"] == 999999          # projected cost recorded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runner.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# contexeval/runner.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_runner.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add contexeval/runner.py tests/test_runner.py
git commit -m "feat: runner with scoring, caching, infeasibility handling"
```

---

### Task 12: Report — aggregate table + PR curve

**Files:**
- Create: `contexeval/report.py`
- Test: `tests/test_report.py` (offline)

**Interfaces:**
- Produces:
  - `aggregate(records: list[dict]) -> dict[str, dict]` — per-method means of `precision, recall, f1, em, answer_f1, bundle_size, context_tokens, completion_tokens`, plus `feasible_n`/`total_n`. `None`/error records are excluded from answer means but counted in `total_n`.
  - `render_table(agg: dict) -> str` — a markdown table.
  - `pr_curve(sweeps: dict[str, list[tuple[float, float]]], out_path)` — plots (recall, precision) polylines per method to a PNG.
- Consumes: nothing beyond stdlib + matplotlib.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
from contexeval.report import aggregate, render_table

RECORDS = [
    {"method": "contex", "precision": 1.0, "recall": 1.0, "f1": 1.0, "em": 1.0,
     "answer_f1": 1.0, "bundle_size": 2, "context_tokens": 200, "completion_tokens": 3, "feasible": True},
    {"method": "contex", "precision": 0.5, "recall": 1.0, "f1": 0.667, "em": 0.0,
     "answer_f1": 0.5, "bundle_size": 4, "context_tokens": 400, "completion_tokens": 5, "feasible": True},
    {"method": "dump-all", "precision": 0.01, "recall": 1.0, "f1": 0.02, "em": None,
     "answer_f1": None, "bundle_size": 500, "context_tokens": 999999, "completion_tokens": None, "feasible": False},
]

def test_aggregate_means_and_feasibility():
    agg = aggregate(RECORDS)
    assert abs(agg["contex"]["recall"] - 1.0) < 1e-9
    assert abs(agg["contex"]["precision"] - 0.75) < 1e-9
    assert agg["contex"]["em"] == 0.5                  # (1.0 + 0.0)/2
    assert agg["dump-all"]["feasible_n"] == 0
    assert agg["dump-all"]["total_n"] == 1
    assert agg["dump-all"]["em"] is None                # no feasible answers to average

def test_render_table_has_rows():
    table = render_table(aggregate(RECORDS))
    assert "contex" in table and "dump-all" in table and "|" in table
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# contexeval/report.py
from collections import defaultdict

_RETR_KEYS = ["precision", "recall", "f1", "bundle_size", "context_tokens"]
_ANS_KEYS = ["em", "answer_f1", "completion_tokens"]

def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

def aggregate(records: list[dict]) -> dict[str, dict]:
    by_method = defaultdict(list)
    for r in records:
        if "error" not in r:
            by_method[r["method"]].append(r)
    out = {}
    for method, recs in by_method.items():
        feasible = [r for r in recs if r.get("feasible")]
        row = {k: _mean([r[k] for r in recs]) for k in _RETR_KEYS}
        row.update({k: _mean([r[k] for r in feasible]) for k in _ANS_KEYS})
        row["feasible_n"] = len(feasible)
        row["total_n"] = len(recs)
        out[method] = row
    return out

def render_table(agg: dict) -> str:
    cols = ["recall", "precision", "f1", "em", "answer_f1", "bundle_size", "context_tokens", "feasible_n", "total_n"]
    lines = ["| method | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for method, row in sorted(agg.items()):
        cells = []
        for c in cols:
            v = row.get(c)
            cells.append("n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)

def pr_curve(sweeps: dict, out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for method, points in sweeps.items():
        points = sorted(points, key=lambda t: t[0])  # by recall
        ax.plot([r for r, _ in points], [p for _, p in points], marker="o", label=method)
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.legend(); ax.set_title("Precision-Recall")
    fig.savefig(out_path, bbox_inches="tight")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_report.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add contexeval/report.py tests/test_report.py
git commit -m "feat: aggregate report table + PR-curve plot"
```

---

### Task 13: Orchestration script (fair protocol) + docs + full smoke

**Files:**
- Create: `scripts/run.py`, `README.md`
- Test: manual end-to-end smoke (documented below); no new unit test (this task is wiring).

**Interfaces:**
- Consumes: everything above. Implements the fair-comparison protocol from the spec §7.

- [ ] **Step 1: Write the orchestration script**

```python
# scripts/run.py
"""Pilot/full run: prep -> publish -> measure Contex bundle size -> set baseline k -> run -> report."""
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

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def main(n: int, mode: str):
    prep.main(n)  # writes corpus.jsonl + questions.jsonl
    corpus = load_jsonl(config.CORPUS_PATH)
    questions = load_jsonl(config.QUESTIONS_PATH)

    client = ContexClient()
    client.publish_corpus(corpus)  # publish once

    # Contex at default threshold; measure average bundle size B.
    contex = ContexRetriever(corpus, client, threshold=config.DEFAULT_THRESHOLD)
    sizes = [contex.retrieve(q["question"]).bundle_size for q in questions]
    B = max(1, round(sum(sizes) / len(sizes)))
    print(f"Contex avg bundle size B={B}; setting baseline k={B}")

    retrievers = [contex, BM25Retriever(corpus, k=B), DenseRetriever(corpus, k=B)]
    if mode == "pilot":
        retrievers.append(DumpAllRetriever(corpus))  # full dump-all only when it fits

    agent = AnswerAgent(); agent.warmup()
    records = run(questions, retrievers, agent, config.RESULTS_PATH)

    agg = aggregate(records)
    table = render_table(agg)
    (config.DATA_DIR / "report.md").write_text(table + "\n")
    print(table)

    # PR curve: sweep Contex threshold and baseline k.
    sweeps = {}
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    ks = sorted({max(1, round(B * m)) for m in (0.5, 1, 2, 4)})
    for t in thresholds:
        recs = [run_one_safe(q, ContexRetriever(corpus, client, threshold=t)) for q in questions]
        sweeps.setdefault("contex", []).append(_avg_pr(recs))
    for k in ks:
        for name, R in (("bm25", BM25Retriever), ("dense", DenseRetriever)):
            recs = [run_one_safe(q, R(corpus, k=k)) for q in questions]
            sweeps.setdefault(name, []).append(_avg_pr(recs))
    pr_curve(sweeps, config.DATA_DIR / "pr_curve.png")
    print("wrote data/report.md and data/pr_curve.png")

def run_one_safe(q, retriever):
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
```

- [ ] **Step 2: Write the README (setup + run)**

`README.md` must document, in order:
1. `pip install -e ".[dev]" && pip install mlx-lm`
2. Clone Contex into `./contex`, `cd contex && docker compose up -d`; verify health; confirm `AUTH_ENABLED=false`, `HYBRID_SEARCH_ENABLED=true`.
3. `mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080`
4. Pilot: `python scripts/run.py 50 pilot` (sizes pool to fit the 32k budget; runs all four methods).
5. Full: `python scripts/run.py 200 full` (dump-all omitted from the agent loop; reported infeasible at scale).
6. Note the disclosed limitation (unlabeled cross-question paragraphs) and the paragraph-level scoring choice.

- [ ] **Step 3: Verify the whole suite (offline) passes**

Run: `pytest -v`
Expected: all non-integration tests PASS.

- [ ] **Step 4: Manual end-to-end smoke (with all servers up)**

Run: `python scripts/run.py 5 pilot`
Expected: prints a table where `dump-all` recall == 1.0 (sanity check), writes `data/report.md` and `data/pr_curve.png`. If `dump-all` recall != 1.0, stop and debug prep/scoring before scaling.

- [ ] **Step 5: Commit**

```bash
git add scripts/run.py README.md
git commit -m "feat: run orchestration (fair protocol) + docs"
```

---

## Self-Review

**Spec coverage:**
- §1 three metrics → Tasks 2 (answer EM/F1), 3 (retrieval P/R/F1), 4 + runner (tokens). ✓
- §3 HotpotQA distractor, fixed seed → Task 5. ✓
- §4 pooling, paragraph unit, `slug` identity, honest scoring → Tasks 5, 3. ✓
- §5 two-tier scale, infeasibility handling → Task 11 (`feasible`/budget), Task 13 (pilot vs full). ✓
- §6 four methods, `contex_query` w/ high top_k, same embedder for dense → Tasks 6, 7, 9. ✓
- §7 fair protocol (measure B, k=round(B), sweep, PR curve) → Task 13, Task 12. ✓
- §8 fixed local agent, warmup, temp 0 → Task 10. ✓
- §9 Contex over MCP, unforked, auth off → Task 8 prerequisite + client. ✓
- §12 caching + per-question try/except → Task 11. ✓
- §13 round-trip, scorer unit tests, dump-all recall==1.0 sanity, e2e smoke → Tasks 8, 2/3, 13. ✓

**Placeholder scan:** No "TBD/handle appropriately" — every code step is concrete. The one runtime-confirmed detail (exact `data_key` format) has an explicit fallback in Task 8 Step 4. ✓

**Type consistency:** `RetrievalResult`, `AnswerResult`, corpus item `{para_id,title,text}`, question item `{qid,question,answer,gold_para_ids}`, and result-record keys are used identically across Tasks 6–13. `retrieval_prf` returns `(precision, recall, f1)` everywhere; `_avg_pr` unpacks in that order. ✓
