# Contex Validation Harness — Design

**Date:** 2026-09-11
**Status:** Approved design; implementation plan to follow.
**Supersedes:** nothing (first spec). Builds on `HANDOFF.md`.

## 1. Purpose

Measure whether Contex (the OSS semantic context bus) actually delivers on its
core promise: routing *lean, sufficient* context to an agent beats both dumping
everything and hand-rolled keyword filtering. The deliverable is **numbers**, not
a demo. The Code Youngstown lightning talk is the milestone/forcing function, not
the product.

The promise is validated with **exactly three metrics** and nothing else:

1. **Retrieval accuracy** — did Contex route the *right* context? Precision /
   recall / F1 of retrieved paragraphs against labeled gold.
2. **Answer quality** — did the lean context let a fixed agent *do the job*
   better? Official HotpotQA answer EM/F1 with the agent in the loop.
3. **Cost** — context + generation **tokens** (the honest headline metric on a
   local model; optional illustrative $ translation for the talk only).

Together: the plumbing works (1), it matters (2), and it's cheaper (3).

## 2. Non-goals (YAGNI)

- No homemade corpus / self-authored relevance labels (self-graded-homework
  problem). Public benchmark with independent ground truth only.
- No creature-feeder demo / live front-end. Deprioritized until metrics exist.
- No forking or modifying Contex. It is run as-is and treated as a black box
  reached over MCP.
- No paid APIs anywhere (no hosted LLMs, no hosted embeddings).
- No full-Wikipedia (fullwiki) indexing.

## 3. Benchmark & ground truth (locked)

**HotpotQA, distractor setting**, validation split, via HuggingFace `datasets`
(`hotpot_qa`, config `distractor`). One example bundles every ingredient the
three metrics need, and HotpotQA ships official scorers so we don't invent
grading:

| Need | HotpotQA provides |
|---|---|
| Corpus of items to route | 10 paragraphs/question (2 gold + 8 distractors) |
| Need / query | The question |
| Relevance labels | Supporting facts → the 2 gold paragraph titles |
| Checkable answer | Gold answer span / yes-no + official EM/F1 |

**Fixed seed** for sampling so runs are reproducible.

## 4. Critical design decision: pool the corpus (locked)

Do **not** run each question against only its own 10 paragraphs — finding 2-of-10
is trivial and exercises nothing. Instead, **pool all sampled questions'
paragraphs into one shared corpus** (dedup by title). Each question is a `need`
against the *full* pool and must find its 2 gold paragraphs among the haystack.
This is essentially HotpotQA's fullwiki setting minus indexing all of Wikipedia.

### Retrieval unit & identity
- **Unit = one paragraph = one published Contex node.** Publishing a flat
  `{"title","text"}` dict with `data_format="json"` hits Contex's JSON
  leaf-object branch → exactly one node per paragraph (verified against
  `src/core/node_parsers.py`). This keeps a clean 1:1 map from a match back to a
  paragraph.
- **`para_id` = slug(title).** HotpotQA identifies paragraphs by title; the 2
  gold paragraphs are the titles named in `supporting_facts`.

### Scoring stays honest under pooling
- **Recall is exact** — the 2 gold paragraphs are labeled; a method either
  surfaced them or didn't.
- **Precision is honest** — any non-gold paragraph retrieved counts against the
  method. Cross-question paragraphs are almost all off-topic (different
  entities); treating unlabeled-as-non-relevant is standard IR practice. Residual
  risk (a stray cross-question paragraph genuinely relevant but unlabeled) is
  small and **disclosed as a known limitation**.
- **Metric granularity:** we score **paragraph-level** P/R/F1 against the 2 gold
  titles. HotpotQA's official *supporting-fact* metric is sentence-level; our
  retrieval unit is the paragraph, so paragraph-level P/R/F1 is the correct,
  clean measure under pooling. Stated explicitly as a scoping choice.

## 5. The two-tier scale design (resolves the dump-all/context-window tension)

The handoff assumed a few-thousand-doc pool that `dump-all` could still fit "in a
long-context window." That does **not** hold on a 32k local model: at
~100 tokens/paragraph, dump-all fits only ~300 paragraphs ≈ ~30 questions. A
200-question pool (~1,800 paras ≈ ~180k tokens) cannot be dumped into 32k at all.

Key observation: **only `dump-all` blows the window.** Contex / BM25 / dense each
return a small bounded set that fits easily, so the large pool is fine for them.
That turns the constraint into the story:

- **Pilot pool (~30–50 questions, sized to fit the 32k window):** run **all four**
  methods including full `dump-all`. Provides the `dump-all` recall=1.0 sanity
  check and the **context-rot** demonstration (dump-all straining the window is
  exactly where answer accuracy degrades).
- **Full pool (~150–200 questions):** run Contex / BM25 / dense head-to-head at
  scale. `dump-all` is **infeasible by construction** — reported as "exceeds 32k
  context; projected cost = N tokens; cannot produce an answer." That
  impossibility is itself the strongest statement of "you can't just dump
  everything," and the pilot already supplies dump-all's runnable data point.

Exact pilot N is pinned empirically: measure the paragraph-token distribution
during prep, size the pilot pool so full dump-all context stays within a ~28k
budget (leaving room for question + prompt + generation).

## 6. Methods under test (same pool, same agent, same questions)

Only the retrieval method varies. Common interface:
`retrieve(question) -> (para_ids, context_text, context_tokens)`.

- **`contex`** — MCP client. Publish the corpus once (idempotent/cached), then per
  question call `contex_query(project_id, query=question, top_k=HIGH,
  threshold=T)` and map returned `data_key`s → `para_id`s. `top_k` is set high so
  **threshold is the binding constraint** — this is what lets Contex "auto-size"
  the bundle rather than us hand-picking k. (The stateless `contex_query` uses the
  same matcher as persistent subscriptions; it is the right fit for a batch run.
  The subscription+resource path remains available if the talk wants a live-agent
  framing.)
- **`dump-all`** — feed the whole pool. The "too much" baseline; recall=1.0 by
  definition. Runs fully only on the pilot pool (see §5).
- **`bm25`** — `rank-bm25` over the corpus, top-k. The hand-rolled keyword filter.
- **`dense`** — sentence-transformers **all-MiniLM-L6-v2**, cosine top-k. Uses the
  **same embedder Contex uses**, so it isolates what Contex's hybrid (dense + FTS
  + RRF) + threshold adds over plain vector similarity.

## 7. Fair-comparison protocol & PR curve (locked)

Contex returns a threshold-based variable-size bundle; baselines return fixed
top-k. To compare fairly:
- Report **Contex at its default threshold (0.5)**.
- Measure Contex's **average bundle size B**; set baseline **`k = round(B)`** so no
  method gets an unfair token budget.
- **Sweep** threshold (Contex) and `k` (baselines) to produce a
  **precision–recall curve** — this is how we *show* Contex auto-sizing to the
  right amount instead of needing a hand-picked k.

## 8. The agent (locked — no paid APIs)

- **Model:** `mlx-community/Qwen2.5-7B-Instruct-4bit`, served by `mlx_lm.server`
  on port 8080 (OpenAI-compatible `/v1/chat/completions`). Confirmed fine on this
  machine (32GB RAM, Apple M4).
- **Held constant across every condition** — the only variable is the context each
  method supplies.
- Pre-warm with a dummy call so first-token weight-load isn't counted.
- Context window treated as 32k for the two-tier sizing (§5).

## 9. Contex setup (run as a black box over MCP)

- Cloned/pinned Contex checkout, unforked. Brought up via its own
  `docker compose up -d` → PostgreSQL+pgvector, Redis, Contex app; MCP over
  streamable HTTP at `http://localhost:8001/mcp`.
- Config: `AUTH_ENABLED=false` (no token dance), `HYBRID_SEARCH_ENABLED=true`,
  `SIMILARITY_THRESHOLD=0.5`.
- Embeddings are local `all-MiniLM-L6-v2` (384-d) — free, no external calls.
- Harness talks to it with the Python `mcp` client SDK.

## 10. Data flow

```
HotpotQA (HF datasets)
  → prep: sample (fixed seed) + pool + dedup → corpus.jsonl + questions.jsonl
  → publish corpus to Contex once  +  build BM25 index  +  build dense index
  → per question × method: retrieve → context → agent → answer
  → scorers: retrieval P/R/F1 · answer EM/F1 · tokens
  → results.jsonl → report table + PR-curve plot
```

## 11. Module layout

- `prep.py` — download, sample, pool, dedup, token-stat measurement, write
  `corpus.jsonl` + `questions.jsonl`.
- `retrievers/` — `contex.py`, `dumpall.py`, `bm25.py`, `dense.py` (shared
  interface).
- `agent.py` — fixed mlx-lm answer agent; token accounting.
- `scoring/` — `retrieval.py` (paragraph P/R/F1), `answer.py` (official HotpotQA
  EM/F1, vendored normalizer), `cost.py` (token counting).
- `runner.py` — questions × methods loop; per-question try/except; caching;
  `results.jsonl`.
- `report.py` — aggregate table + PR-curve plot.
- `contex_client.py` — thin MCP client wrapper (publish / query / id-mapping).

## 12. Caching & error handling

- **Cache:** corpus prep output, Contex publish (idempotent per project), dense
  embeddings, and agent answers (keyed by method + qid + context-hash). Reruns are
  cheap.
- **Errors:** per-question `try/except` — one failure logs and skips, never kills
  the run. Fixed seed for the sample.

## 13. Testing

- **Round-trip (first task):** publish 5 paragraphs → `contex_query` → confirm
  returned `data_key`s map back to the right `para_id`s.
- **Scorer unit tests:** run the answer EM/F1 scorer against known HotpotQA
  examples; assert it matches official numbers.
- **Sanity:** assert `dump-all` retrieval recall == 1.0.
- **Smoke:** end-to-end on 5 questions across all methods.

## 14. Dependencies

`datasets`, `mcp` (client SDK), `openai`, `rank-bm25`, `sentence-transformers`,
`matplotlib`, plus the model tokenizer for context-token counting. `mlx-lm` for
serving the agent. Docker (for Contex).

## 15. Decisions resolved during design

- **RAM/model:** 32GB / M4 → `Qwen2.5-7B-Instruct-4bit` pinned.
- **Contex interface:** MCP (REST is deprecated), stateless `contex_query`,
  default threshold 0.5, tunable per call, local embeddings.
- **Dense baseline:** included (cheap; isolates the hybrid contribution).
- **Scale:** two-tier pilot + full pool (§5).
- **Contex delivery:** run locally via docker-compose, unforked.

## 16. Remaining confirmables (handled in the plan, not blockers)

- Exact pilot N and full N — pinned from measured paragraph-token stats in prep.
- $ translation pricing (talk only) — defer; pick a hosted model's public
  pricing when the talk is assembled.
- Live front-end for the talk — none until numbers are in.
