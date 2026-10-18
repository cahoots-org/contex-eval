# Contex validation on HotpotQA — results

**Date:** 2026-09-12
**Harness:** this repo (`contexeval`), spec `docs/superpowers/specs/2026-09-11-contex-validation-harness-design.md`.
**TL;DR:** Lean retrieval beats dumping the whole pool **decisively** (cost + feasibility). On HotpotQA specifically, **Contex is statistically tied with a plain dense retriever** and only modestly ahead of BM25 (not firmly significant). The retrieval-quality result is dataset-shaped: HotpotQA is semantic/multi-hop, which favors the dense half of Contex's hybrid.

> **UPDATE (2026-09-14):** the "tie with dense" throughout this doc's original body was the **broken-FTS era** (`plainto_tsquery` AND-matched ~nothing, so Contex's "hybrid" was silently dense-only). With **Contex v0.2.5 (ParadeDB BM25)**, Contex **significantly beats dense here too** — recall@5 **0.830 vs 0.767, 95% CI [+0.027, +0.103] (excludes 0)**. See the "Re-eval on Contex v0.2.5" section at the end. The original body is kept for the chronological bug→fix record.

## Setup

- **Benchmark:** HotpotQA distractor, validation split (7,405 questions), sampled with fixed `SEED=13`. Paragraphs pooled across sampled questions into one shared corpus (dedup by title slug) — each question is a `need` against the *full* pool, not just its own 10 paragraphs.
- **Methods (same pool, same agent, same questions; only retrieval varies):**
  - `contex` — Contex over MCP, **hybrid search on** (`HYBRID_SEARCH_ENABLED=true`).
  - `dense` — `all-MiniLM-L6-v2` cosine top-k (same embedder Contex uses).
  - `bm25` — `rank-bm25` top-k.
  - `dump-all` — the entire pool (the "too much" baseline / recall ceiling).
- **Retrieval budget:** `k=5` for all ranked methods. **Important:** under hybrid search Contex *ignores* the similarity threshold (verified: identical results at thresholds 0.0–0.9), so `top_k` is the only knob that bounds the bundle — it is the retrieval budget, and the baselines use the same `k`.
- **Agent (held constant):** `mlx-community/Qwen2.5-7B-Instruct-4bit` via `mlx-lm`, temperature 0, `max_tokens=256`.
- **Scoring:** official HotpotQA `normalize_answer` EM/F1 for answers; paragraph-level recall/precision/F1 against the 2 gold paragraphs; cost = context + generation tokens. `dump-all` is "infeasible" when its context exceeds the 32k window (28k budget) — recorded as projected cost, no answer.

## Results

### n = 150 questions (headline)

| method | recall@5 | EM | answer-F1 | ctx tokens | gen tokens | feasible |
|---|---|---|---|---|---|---|
| contex | 0.770 | 0.400 | 0.508 | 672 | 5.6 | 150/150 |
| dense | 0.767 | 0.407 | 0.516 | 647 | 5.1 | 150/150 |
| bm25 | 0.723 | 0.340 | 0.457 | 652 | 4.9 | 150/150 |
| dump-all | 1.000 | — | — | **215,295** | — | **0/150** |

**Paired bootstrap 95% CI on (Contex − baseline), 10k resamples** (`scripts/analyze.py`):

| metric | vs | mean | 95% CI | W/T/L |
|---|---|---|---|---|
| recall@5 | bm25 | +0.047 | [−0.010, +0.100] | 37/87/26 |
| recall@5 | dense | +0.003 | [−0.020, +0.027] | 7/137/6 |
| EM | bm25 | +0.060 | [−0.013, +0.133] | 20/119/11 |
| EM | dense | −0.007 | [−0.047, +0.033] | 5/139/6 |
| answer-F1 | bm25 | +0.051 | [−0.024, +0.127] | 28/102/20 |
| answer-F1 | dense | −0.008 | [−0.052, +0.036] | 9/133/8 |

Every Contex−dense interval sits on 0 (a tie); every Contex−BM25 interval is positive but includes 0 (modest, not firmly significant). See `pr_curve_n150.png` — Contex and dense overlap, both above BM25.

### n = 50 questions (earlier, smaller sample — shown for contrast)

| method | recall@5 | EM | answer-F1 | ctx tokens |
|---|---|---|---|---|
| contex | 0.77 | 0.44 | 0.53 | 641 |
| dense | 0.76 | 0.38 | 0.47 | 628 |
| bm25 | 0.72 | 0.28 | 0.38 | 635 |
| dump-all | 1.00 | — | — | 68,714 |

At n=50, Contex *appeared* to beat dense (EM 0.44 vs 0.38). The paired bootstrap showed those gaps were not significant, and at n=150 the Contex−dense gap **collapsed to zero** — a textbook small-sample artifact. This is exactly why the rigor pass mattered.

## Findings

1. **"Don't dump everything" is validated decisively.** At n=150 `dump-all` needs 215k tokens (~330× the lean methods) and is *infeasible* — it exceeds the context window and answers 0/150. Lean retrieval answers at ~650 tokens with no quality penalty. This is the strong, defensible result.
2. **On HotpotQA, Contex ≈ dense and only modestly ≥ BM25.** Contex's hybrid did not beat a plain dense retriever here. Mechanistically expected: HotpotQA is semantic/multi-hop, so the dense component of Contex's hybrid (dense + FTS + RRF) carries it and it tracks pure dense.
3. **The comparison at equal cost is fair and clean:** all ranked methods used ~650 context tokens; the differences are in *which* paragraphs they surfaced, not budget.

## Caveats / what this does NOT show

- **Single dataset, and one that favors dense.** HotpotQA rewards semantic matching. Contex's keyword/FTS half should matter more on exact-match corpora (IDs, error codes, config keys). Untested here.
- **Auto-sizing untested.** Under hybrid the threshold is a no-op, so Contex ran as fixed top-k like the baselines. Its "return exactly enough, no `k` to tune" property only exists in vector-only mode (`HYBRID_SEARCH_ENABLED=false`), which we did not run.
- **Single run per n** (no cross-seed variance); confidence comes from the paired bootstrap over questions, not from repeated sampling.
- **Retrieval quality only.** The benchmark does not measure Contex's operational value (eliminating manual context-assembly / continuous routing), which is a separate claim.

## Reproducing

```
python scripts/run.py 150 pilot     # writes data/report.md + data/pr_curve.png + data/results.jsonl
python scripts/analyze.py           # paired bootstrap CIs from data/results.jsonl
```

Raw records for this run: `results_n150.jsonl`. Environment/workarounds documented in the top-level `README.md`.

## Re-eval on Contex v0.2.5 (ParadeDB BM25), retrieval-only — also a significant win

Re-ran HotpotQA (same n=150, k=5, fresh `hotpot-paradedb` project) against Contex v0.2.5 (ParadeDB
`pg_search` BM25), retrieval-only (recall@5 + paired bootstrap):

| method | recall@5 | context tokens |
|---|---|---|
| **contex (ParadeDB)** | **0.830** | 668 |
| dense | 0.767 | 647 |
| bm25 | 0.723 | 652 |

- **contex − dense = +0.063, 95% CI [+0.027, +0.103]** — excludes 0 (W/T/L 23/121/6)
- **contex − bm25  = +0.107, 95% CI [+0.060, +0.153]** — excludes 0 (W/T/L 38/102/10)

The earlier HotpotQA run had Contex at 0.770 ≈ dense 0.767 (a *tie*) — but that was the broken-FTS
version (`plainto_tsquery` AND-matched ~nothing), so the "hybrid" was silently dense-only. HotpotQA's
entity-heavy multi-hop questions (names, titles, places) DO carry lexical signal, so once ParadeDB BM25
is fused in, Contex jumps 0.770 → 0.830 and **significantly beats dense**. The prior "tie on semantic
data" was a symptom of the same bug, not a property of the data.

**Cross-dataset conclusion:** with a real BM25 ranker, Contex's hybrid significantly out-retrieves both
pure dense (same embeddings) and pure BM25 on BOTH public benchmarks — SciFact (+0.058) and HotpotQA
(+0.063), both 95% CIs excluding 0. This is the Contex-specific retrieval win, and it is robust across a
lexical and a semantic dataset.

**Effect-size honesty:** the win is *reliable*, not *dominant* — here it rides on ~23 flipped queries of
150 (W/T/L 23/121/6, ~15%); the other ~80% are identical to dense. And the `dense` baseline is
`all-MiniLM-L6-v2` — Contex's own (modest, 2021) embedder. **⚠️ The embedder ablation
(`2026-09-14-embedder-ablation.md`) answers the "does it survive a modern embedder?" question: NO.** A
2023 model (bge-base) alone significantly beats Contex's hybrid here too (−0.090, CI excludes 0), and
even an upgraded hybrid loses to modern-dense-alone (−0.037). This win holds only against Contex's weak
embedder. See the SciFact doc's "Honest caveats & effect size" for the full framing.
