# Real Contex(gte-base) on SciFact — the strong-embedder baseline

**Date:** 2026-09-15
**Confirms** the embedder ablation (`2026-09-14-embedder-ablation.md`) on the **real system**: with a
strong embedder, Contex's hybrid **ties plain dense** (marginally below). This is not a simulation —
Contex was actually reconfigured to embed with **gte-base (768-dim)** and re-published.

## Setup

- **Contex embedder swapped to `thenlper/gte-base`** (768-dim, prefix-free): patched
  `semantic_matcher.py` model + `embedding_dim`, and `Vector(384)→Vector(768)` in `db_models.py` +
  migration 001. Fresh DB, sanity-checked (768-dim round-trip).
- **Contex app rebuilt native arm64** (dropped the Dockerfile's `--platform=linux/amd64` + amd64 digest
  pin) so gte-base embedding was tractable — emulated-amd64 was ~2.5 docs/min (≈35 hrs); native arm64
  was ~22 docs/min. Postgres/ParadeDB stayed amd64. (Publish still took ~5 hrs because the HNSW index
  inserts run on the emulated Postgres and slow as the index grows.)
- Full **5,183-doc SciFact** corpus, 300 test queries, k=10, hybrid on. `dense` baseline = the same
  gte-base (so `contex − dense` isolates what BM25+RRF adds over the strong embedder). Paired bootstrap.

## Result — SciFact recall@10 (n=300)

| method | recall@10 | context tokens |
|---|---|---|
| **dense (gte-base)** | **0.890** | 3,263 |
| contex (gte-base hybrid) | 0.879 | 3,490 |
| bm25 (ParadeDB pg_search) | 0.776 | 3,525 |

Paired bootstrap 95% CI:
- **contex − dense = −0.011, [−0.041, +0.018] → tie** (marginally below), W/T/L 9/275/16
- contex − bm25 = +0.103, [+0.066, +0.141] — excludes 0 (still beats keyword)

## Findings

1. **With a strong embedder, Contex's hybrid ties (slightly trails) plain dense.** The BM25 fusion's
   advantage evaporates — confirmed on the *real* Contex, not just the rank-bm25 simulation.
2. **ParadeDB's BM25 did not rescue it.** We'd flagged that Contex's real BM25 beat the rank-bm25
   simulation with MiniLM (0.841 vs 0.824), leaving open whether it might keep the hybrid ahead with a
   strong embedder. It doesn't: real ParadeDB fusion (−0.011) matches the ablation's simulated tie
   (−0.016). That hypothesis is settled.
3. **The embedder upgrade flips the SciFact verdict cleanly:**

   | Contex embedder | contex | dense | contex − dense |
   |---|---|---|---|
   | all-MiniLM-L6-v2 (weak, 384-d) | 0.841 | 0.783 | **+0.058** (sig) — hybrid wins |
   | thenlper/gte-base (strong, 768-d) | 0.879 | 0.890 | **−0.011** (tie) — edge gone |

   Upgrading the embedder is the big lever (dense +0.107); the earlier "Contex hybrid beats dense" was a
   weak-embedder artifact.

## Baseline for the parameter sweep

At Contex's **default unweighted RRF** (`RRF_K=60`), the hybrid slightly *trails* pure dense with a strong
embedder — i.e. BM25 is adding a little noise, not signal. The sweep's realistic goals:
- **Down-weight BM25 / tune RRF** to recover ≈ dense (stop the small drag), and
- test whether *any* config lets the lexical half add signal over strong dense (the oracle headroom that
  existed with MiniLM may be mostly gone once dense is strong — worth measuring).
The honest prior: with a strong embedder there is little complementary lexical signal left to fuse, so
the sweep likely lands at "hybrid ≈ dense," with BM25's value being robustness on lexical/OOV queries
rather than average recall.

## Caveats

- **SciFact only.** HotpotQA-with-gte-base was not run (another multi-hour publish); the ablation's
  simulation suggests the same direction there.
- Single run; confidence from the paired bootstrap over 300 queries.
- gte-base is prefix-free, so it's a clean drop-in for Contex (which applies no query/doc instruction).

## Reproducing

```
# Contex patched to gte-base + Vector(768), app built native arm64, fresh DB, published 5183 docs.
CONTEXEVAL_DEVICE=mps python scripts/run_beir.py 10 scifact-gtebase nopublish
```
