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

1. With Contex's shipped embedder (`all-MiniLM-L6-v2`), the hybrid beats dense significantly on both
   benchmarks. The gain comes from the 8–15% of queries where the hybrid wins. Most queries come back identical.
2. With a modern embedder the advantage disappears. Swapping the embedder lifts dense recall by 9–11
   points, which is far more than fusion adds. Fusing BM25 into a strong dense retriever ties it at
   best and hurts it slightly at worst.
3. Lean retrieval beats dumping the pool on cost and feasibility: dump-all averages about 215k context
   tokens on HotpotQA n=150 and does not fit the 28k budget for any question.
4. Hybrid search ignores the `threshold` parameter. Only `top_k` bounds the bundle (see below).

Full write-ups, newest first:

- [`2026-09-15-contex-gtebase-real.md`](docs/results/2026-09-15-contex-gtebase-real.md): real Contex reconfigured to gte-base
- [`2026-09-14-embedder-ablation.md`](docs/results/2026-09-14-embedder-ablation.md): does the hybrid win survive a modern embedder? (no)
- [`2026-09-12-scifact-keyword-regime.md`](docs/results/2026-09-12-scifact-keyword-regime.md): SciFact across Contex versions (broken FTS → OR fix → ParadeDB)
- [`2026-09-12-hotpotqa-validation.md`](docs/results/2026-09-12-hotpotqa-validation.md): HotpotQA with an agent in the loop (EM/F1, cost)
- [`contex-hybrid-fts-bug.md`](docs/contex-hybrid-fts-bug.md): the upstream `plainto_tsquery` bug report

Open items: HotpotQA with real Contex(gte-base) has not been run yet (the publish takes hours), and
neither has an RRF weighting sweep.

## Setup

```bash
pip install -e ".[dev]"
pip install mlx-lm        # only for the HotpotQA answer agent
```

Contex runs from a local checkout at `./contex/` (gitignored, cloned from upstream):

```bash
cd contex && docker compose -f docker-compose.yml -f docker-compose.override.yml up -d && cd ..
curl -s http://127.0.0.1:8001/health      # {"status":"ok"}
```

The override sets `AUTH_ENABLED=false`, `CONTEX_PROTECTED_MODE=false` and `HYBRID_SEARCH_ENABLED=true`,
and publishes only port 8001. The harness talks MCP at `http://127.0.0.1:8001/mcp` (`contexeval/config.py`).

The checkout carries local patches for the gte-base run: the embedder and `embedding_dim` in
`src/core/semantic_matcher.py`, `Vector(768)` in `src/core/db_models.py` and migration 001, and a
native arm64 app build in the `Dockerfile`. Postgres stays amd64. Revert these with `git -C contex diff`
to get stock Contex (MiniLM, 384-dim). The embedder has to match `EMBED_MODEL` in `config.py`; otherwise
`dense` no longer isolates what fusion adds.

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

## Hybrid search ignores `threshold`

Under `HYBRID_SEARCH_ENABLED=true`, Contex's returned `similarity` values are RRF scores (about
`1/(60+rank)`, so roughly 0.016 at the top), not cosine similarities, and the `similarity >= threshold`
filter only runs on the vector-only path. Thresholds from 0.0 to 0.9 return identical results, so
`top_k` is the effective retrieval budget and the Contex PR-curve sweep is flat. To exercise
threshold-based sizing, run Contex with `HYBRID_SEARCH_ENABLED=false`, which drops the lexical half.

## Hardware notes (Apple Silicon)

- The stock Contex images are `linux/amd64` and run emulated. Publishing with gte-base under emulation
  ran at about 2.5 docs/min; a native arm64 app build ran at about 22 docs/min. HNSW inserts on the
  emulated Postgres still slow down as the index grows, and a full SciFact publish took about 5 hours.
- mlx-lm serves one request at a time. A 6.5k-token prompt takes around 2 minutes.

## Limitations

- Unlabelled HotpotQA pool paragraphs count as negatives even when they are relevant.
- Each configuration is a single run. Confidence comes from the bootstrap over queries, not from
  repeated runs.
- The gte-base result covers SciFact only so far.
