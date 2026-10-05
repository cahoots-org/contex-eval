# contexeval — Contex retrieval benchmark harness

Measures whether [Contex](https://github.com/cahoots-org/contex)'s hybrid retrieval (BM25 + dense, RRF
fusion) beats plain baselines on public benchmarks with independent labels. Methods compared on the
same corpus at the same retrieval budget `k`:

- `contex`: Contex over MCP, `HYBRID_SEARCH_ENABLED=true`
- `dense`: sentence-transformers cosine top-k, using the same embedder Contex runs
- `bm25`: `rank-bm25` top-k
- `dump-all`: the whole pool (HotpotQA only; a recall ceiling and cost reference)

## Current findings

| Setup | Benchmark | contex | dense | contex − dense (95% CI) |
|---|---|---|---|---|
| Contex v0.2.5 (ParadeDB BM25), MiniLM embedder | SciFact recall@10 | 0.841 | 0.783 | +0.058 [+0.024, +0.092] |
| Contex v0.2.5 (ParadeDB BM25), MiniLM embedder | HotpotQA recall@5 | 0.830 | 0.767 | +0.063 [+0.027, +0.103] |
| Contex v0.2.5, **gte-base** embedder | SciFact recall@10 | 0.879 | 0.890 | −0.011 [−0.041, +0.018] (tie) |
| **Contex v1** (`d25b593`, stock: gte-base, hybrid) | SciFact recall@10 | 0.825 | 0.890 | **−0.066 [−0.102, −0.029]** |
| Contex v1, `CANDIDATE_POOL_FACTOR=1` | SciFact recall@10 | 0.882 | 0.890 | ≈ tie |

1. With Contex's shipped embedder (`all-MiniLM-L6-v2`), the hybrid beats dense significantly on both
   benchmarks. The gain comes from the 8–15% of queries where the hybrid wins. Most queries come back identical.
2. With a modern embedder the advantage disappears. Swapping the embedder lifts dense recall by 9–11
   points, which is far more than fusion adds. Fusing BM25 into a strong dense retriever ties it at
   best and hurts it slightly at worst.
3. Lean retrieval beats dumping the pool on cost and feasibility: dump-all averages about 215k context
   tokens on HotpotQA n=150 and does not fit the 28k budget for any question.
4. **Contex v1 regressed below dense.** Since #236, each ranker over-fetches `top_k × 10` candidates
   *before* RRF. Fusing two 100-deep lists hands the weaker BM25 ranker more of the top 10. Fusing at
   depth `top_k` (`CANDIDATE_POOL_FACTOR=1`) recovers +0.058, confirmed both live and with
   `scripts/fusion_replay.py`.
5. Before v1, hybrid search ignored `threshold` and returned RRF scores. v1 reports cosine similarity
   and applies the threshold on the hybrid path.

Full write-ups, newest first:

- [`2026-10-04-contex-v1-scifact.md`](docs/results/2026-10-04-contex-v1-scifact.md): Contex v1 regression and its cause (RRF over-fetch)
- [`2026-09-15-contex-gtebase-real.md`](docs/results/2026-09-15-contex-gtebase-real.md): real Contex reconfigured to gte-base
- [`2026-09-14-embedder-ablation.md`](docs/results/2026-09-14-embedder-ablation.md): does the hybrid win survive a modern embedder? (no)
- [`2026-09-12-scifact-keyword-regime.md`](docs/results/2026-09-12-scifact-keyword-regime.md): SciFact across Contex versions (broken FTS → OR fix → ParadeDB)
- [`2026-09-12-hotpotqa-validation.md`](docs/results/2026-09-12-hotpotqa-validation.md): HotpotQA with an agent in the loop (EM/F1, cost)
- [`contex-hybrid-fts-bug.md`](docs/contex-hybrid-fts-bug.md): the upstream `plainto_tsquery` bug report

Open items: HotpotQA on Contex v1, and a fix for the #236 fusion depth in Contex itself.

## Setup

```bash
pip install -e ".[dev]"
pip install mlx-lm        # only for the HotpotQA answer agent
```

Contex runs from a local checkout at `./contex/` (gitignored, cloned from upstream, currently at
`d25b593`). The eval stack is its own compose project on `:8011`, so it can run next to other local
Contex instances:

```bash
cd contex && docker compose -p contexeval -f docker-compose.yml -f docker-compose.override.yml up -d --build && cd ..
curl -s http://127.0.0.1:8011/health      # {"status":"healthy"}
export CONTEX_MCP_URL=http://127.0.0.1:8011/mcp
```

`contex/docker-compose.override.yml` (local, not committed):

```yaml
services:
  postgres: { ports: !reset [] }
  redis:    { ports: !reset [] }
  contex:
    build: { platforms: !override [linux/arm64] }   # native on Apple Silicon
    ports: !override ["127.0.0.1:8011:8001"]
    environment: [AUTH_ENABLED=false, CONTEX_PROTECTED_MODE=false, HYBRID_SEARCH_ENABLED=true]
    deploy: { resources: { limits: !override { memory: 6GB, cpus: '8.0' } } }
```

Contex v1 defaults to `thenlper/gte-base`, the same model as `EMBED_MODEL` in `config.py`, so `dense`
isolates what fusion adds. The harness publishes with `contex_publish_batch` in batches of 100,
because single `contex_publish` calls are rate-limited to 60/min.

## Running

### HotpotQA, agent in the loop (`scripts/run.py`)

Start the answer agent first:

```bash
mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080
```

```bash
python scripts/run.py 50 pilot     # includes dump-all; its recall must be 1.000
python scripts/run.py 150 full     # dump-all omitted from the agent loop
python scripts/analyze.py          # paired bootstrap CIs, contex vs each baseline
```

Writes `data/results.jsonl`, `data/report.md` and `data/pr_curve.png`.

### BEIR, retrieval only (`scripts/run_beir.py`)

BEIR provides relevance labels but no gold answers, so this script scores recall@k and context tokens
and skips the agent.

```bash
python -m contexeval.prep_beir scifact test                  # full corpus, all test queries
python scripts/run_beir.py 10 scifact                        # k=10, publishes to project 'scifact'
python scripts/run_beir.py 10 scifact nopublish              # reuse an already-published project
```

### Embedder ablation (`scripts/embedder_ablation.py`)

Runs locally and does not need Contex. It compares BM25, MiniLM dense, bge-base dense, and
`RRF(bm25, dense)` for each embedder, on whatever is in `data/corpus.jsonl` and `data/questions.jsonl`.

```bash
CONTEXEVAL_DEVICE=cpu python scripts/embedder_ablation.py 10 scifact
```

`CONTEXEVAL_DEVICE` (`cpu` / `mps` / unset for auto) pins the sentence-transformers device for both
the ablation and the `dense` retriever. Large BERT models can hang on Apple MPS, in which case use `cpu`.

### Fusion replay (`scripts/fusion_replay.py`)

Pulls Contex's real vector and ParadeDB BM25 top-100 rankings for a published project straight from
Postgres, then scores RRF variants offline: fusion depth sweep, vector-only, BM25-only, and paired CIs.
Use it to test a fusion change without re-publishing.

```bash
python scripts/fusion_replay.py scifact-v1 10
```

### Tests

```bash
pytest                    # offline unit tests
pytest -m integration     # needs live Contex (and mlx-lm for agent tests)
```

## Methodology

- **Fair comparison:** every method gets the same corpus, the same queries and the same `k`. `dense`
  uses Contex's own embedder, so `contex − dense` measures exactly what BM25 + RRF adds.
- **HotpotQA pooling:** distractor paragraphs from all sampled questions go into one shared corpus
  (deduplicated by title), and each question searches the whole pool. Gold paragraphs come from
  `supporting_facts`. Unlabelled pool paragraphs count as negatives, which is conservative for precision.
- **Scoring:** paragraph-level recall/precision/F1; answers use the official HotpotQA EM/F1
  normalisation and are scored only when the context fits the budget. Cost is context plus generation
  tokens.
- **Significance:** paired bootstrap over queries, 10k resamples, seeded. A gap counts as significant
  only when its 95% CI excludes 0. Win/tie/loss counts are reported alongside.
- **Agent:** `Qwen2.5-7B-Instruct-4bit` via mlx-lm at temperature 0. EM/F1 reflect retrieval and
  model capability together.

## Threshold behaviour (pre-v1)

Before Contex v1, hybrid search returned RRF scores (about `1/(60+rank)`) and ignored `threshold`, so
`top_k` was the only retrieval budget and the Contex PR-curve sweep was flat. v1 reports cosine
similarity from the vector ranker and applies `threshold` on the hybrid path. `run_beir.py` passes
`threshold=0.0`, which keeps `top_k` as the budget.

## Hardware notes (Apple Silicon)

- Build Contex native arm64 (the override above). Emulated amd64 timed out on a 500-doc publish batch.
  Native, with 8 CPUs, publishes about 100–118 docs/min, so full SciFact takes about 50 minutes.
- pgvector's default `hnsw.ef_search=40` caps each vector query at 40 rows, with the project filter
  applied afterwards. This has no effect on a single-project DB.
- mlx-lm serves one request at a time. A 6.5k-token prompt takes around 2 minutes.

## Limitations

- Unlabelled HotpotQA pool paragraphs count as negatives even when they are relevant.
- Each configuration is a single run. Confidence comes from the bootstrap over queries, not from
  repeated runs.
- The Contex v1 and gte-base results cover SciFact only so far.
