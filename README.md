# contexeval — Contex retrieval benchmark harness

End-to-end evaluation of four retrieval methods (Contex, BM25, Dense, Dump-All) on HotpotQA
distractor questions, with an mlx-lm answer agent, paragraph-level P/R/F1 + EM/F1 scoring,
and a precision-recall curve comparing methods at equal context budgets.

---

## Setup

### 1. Install Python dependencies

```bash
pip install -e ".[dev]"
pip install mlx-lm        # for the local answer agent
```

### 2. Start Contex via the vendored checkout

The repo includes a patched Contex checkout at `./contex/`.

```bash
cd contex
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d
cd ..
```

Verify health (should return `{"status":"ok"}`):

```bash
curl -s http://127.0.0.1:8001/health
```

Confirm settings in the compose override:
- `AUTH_ENABLED=false` (or `CONTEX_PROTECTED_MODE=false`)
- `HYBRID_SEARCH_ENABLED=true`

The MCP endpoint is exposed at `http://127.0.0.1:8001/mcp/mcp`.

### 3. Start the mlx-lm answer agent

```bash
mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080
```

Wait until the server prints that it is ready before running the harness.

---

## Running the benchmark

### Pilot run (50 questions, all four methods including Dump-All)

```bash
python scripts/run.py 50 pilot
```

Writes `data/report.md` (markdown table) and `data/pr_curve.png`.

**Sanity check:** the `dump-all` row in the table should have `recall = 1.000`.  If it does not,
stop and debug `contexeval/prep.py` or `contexeval/scoring/retrieval.py` before scaling.

### Full run (200 questions; Dump-All omitted — infeasible at scale)

```bash
python scripts/run.py 200 full
```

`DumpAllRetriever` is excluded in `full` mode because the resulting context (all pooled
paragraphs) exceeds the 28 k-token `CONTEXT_BUDGET`, making it infeasible for the agent.
It is retained in the printed/written table as a theoretical upper-bound reference only.

---

## Scoring methodology

- **Unit:** paragraph (slug-keyed by Wikipedia title, de-duplicated across questions).
- **Pooling:** all paragraphs from all distractor contexts are pooled into a shared corpus;
  gold paragraphs are taken from `supporting_facts`. Paragraphs that appear in a question's
  distractor set but not in `supporting_facts` are treated as negatives — this is an honest
  (conservative) recall denominator.
- **Retrieval:** precision, recall, F1 over the paragraph set.
- **Answer:** exact-match (EM) and token-overlap F1 against the dataset answer string, scored
  only on feasible (within-budget) contexts.

---

## Known finding: hybrid RRF scores and threshold behaviour

Contex runs with `HYBRID_SEARCH_ENABLED=true` by default, which fuses its BM25 and dense
rankings via **Reciprocal-Rank Fusion (RRF)**.  The `similarity` values `contex_query` returns
are therefore **RRF fused scores**, roughly `1 / (RRF_K + rank)` with `RRF_K = 60` — so the
top match scores about `1 / (60 + 1) ≈ 0.016`, NOT a cosine similarity in `[0, 1]`.

**Critical empirical finding: under hybrid search, the `threshold` parameter is ignored.**

Live testing queried Contex at thresholds 0.0, 0.1, 0.3, 0.5, and 0.9.  Every threshold
returned the identical set of matches with identical RRF similarities (~0.016).  The reason:
Contex's cosine `similarity >= threshold` filter only runs in the pure-vector code path; the
RRF/hybrid path bypasses it entirely.  Only `top_k` bounds the returned bundle.

**Consequences for experiment design:**

- The spec's premise that "Contex auto-sizes the returned bundle via its similarity threshold"
  does **NOT** hold under hybrid search.  With the threshold a no-op, Contex behaves like
  fixed-`top_k` retrieval, and the Contex threshold-sweep / PR curve is degenerate — all sweep
  points are identical.
- `scripts/run.py` uses `CONTEX_THRESHOLD = 0.0` and `CONTEX_THRESHOLDS = [0.0, 0.005, 0.01,
  0.02, 0.05]`.  Under hybrid search these constants have no effect on Contex's results; only
  `HIGH_TOPK = 100` bounds the bundle.  The sweep is retained for completeness and for use in
  vector-only mode.

**Key experiment-design decision for operators:**

| Mode | `threshold` effect | Auto-sizing | Lexical (FTS) half |
|------|--------------------|-------------|-------------------|
| `HYBRID_SEARCH_ENABLED=true` (default) | **ignored** — no-op | No — fixed `top_k` | Yes |
| `HYBRID_SEARCH_ENABLED=false` (vector-only) | **applied** — cosine ≥ threshold | Yes | No |

To exercise true threshold-based auto-sizing (cosine `similarity >= threshold`), run Contex in
**vector-only mode** (`HYBRID_SEARCH_ENABLED=false`).  This enables the threshold filter at the
cost of dropping the lexical/FTS half of Contex's hybrid retrieval.

---

## Hardware / performance caveat

This harness was validated on Apple Silicon, where two factors make full runs slow:

- **Contex under emulation:** the vendored Contex docker images are `linux/amd64` and run under
  emulation on Apple Silicon, so its database and embedding steps are noticeably slower than
  native.
- **Sequential local LLM:** the mlx 7B server serves requests **one at a time** and
  prompt-processes large contexts slowly.  A ~6.5k-token `dump-all` prompt takes roughly
  **~2 minutes**; at real pilot scale `dump-all` prompts approach the 28k-token `CONTEXT_BUDGET`
  and take considerably longer.

Plan pilots accordingly: budget patience, use a smaller model (e.g. `Qwen2.5-3B-Instruct-4bit`),
and/or reduce the number of questions that exercise `dump-all`.

---

## Local Contex workaround notes

The vendored `contex/` checkout includes two local patches required to run it:

1. **Shadowed `auth_enabled` variable in `main.py`** — a local variable shadowed the
   `AUTH_ENABLED` config flag, causing startup to crash when auth was disabled.  The patch
   renames the local variable; see `contex/main.py`.

2. **`docker-compose.override.yml`** — sets `CONTEX_PROTECTED_MODE=false` (disables auth
   token checks) and leaves PostgreSQL and Redis host ports unpublished to avoid conflicts
   with other local services.  The only published port is `8001` (Contex HTTP/MCP).

---

## Disclosed limitations

- **Unlabelled negatives:** HotpotQA distractor paragraphs that appear in the pooled corpus
  but are not in `supporting_facts` are treated as negatives.  A retriever that returns them
  is penalised in precision even if they are factually relevant.
- **Scale:** at 200+ questions the pooled corpus can exceed 3 000 paragraphs.  `DumpAllRetriever`
  is excluded from the agent loop in `full` mode for this reason.
- **Single local agent:** all four methods are evaluated with the same mlx-lm Qwen 2.5 7B
  model at temperature 0.  EM/F1 scores reflect retrieval quality + model capability jointly.
