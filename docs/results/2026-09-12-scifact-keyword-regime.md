# Contex on a keyword/lexical regime (BEIR SciFact) — results

**Date:** 2026-09-12
**Question:** On HotpotQA (semantic) Contex tied a plain dense retriever. Does a **lexical-favoring**
benchmark — where BM25/FTS matters — let Contex's hybrid pull ahead of pure dense? i.e. is there a
retrieval-quality win that is actually *Contex-specific* rather than generic hybrid retrieval?

**Answer: No.** On SciFact too, Contex is statistically indistinguishable from plain dense (and BM25).
Its hybrid layer adds ~nothing measurable on retrieval quality, on either a semantic or a lexical public benchmark.

## Setup

- **Dataset:** BEIR **SciFact**, test split — full **5,183-doc corpus**, **300 queries** with relevance labels.
  Retrieval-only (BEIR has no gold answers), so we score **recall@10** + context-token cost. No agent.
- **Methods (same corpus, same `k=10`, hybrid):** `contex` (MCP, hybrid on), `dense`
  (`all-MiniLM-L6-v2` — the *same* embeddings Contex uses), `bm25` (`rank-bm25`). So `contex` vs
  `dense` isolates exactly what Contex's FTS + RRF fusion adds over those embeddings used plainly.
- Prep via `contexeval/prep_beir.py`; run via `scripts/run_beir.py 10 scifact-full`.

## Result (n = 300 queries, full 5,183-doc corpus, k=10)

| method | recall@10 | mean context tokens |
|---|---|---|
| contex | 0.792 | 3,314 |
| dense | 0.783 | 3,349 |
| bm25 | 0.776 | 3,525 |

**Paired bootstrap 95% CI on (Contex − baseline) recall, 10k resamples:**

| vs | mean | 95% CI | W/T/L |
|---|---|---|---|
| dense | +0.008 | [−0.010, +0.028] | 8/287/5 |
| bm25 | +0.016 | [−0.027, +0.060] | 31/246/23 |

Both intervals include 0. Contex is marginally above both on the point estimate, but the gaps are
noise — 287 of 300 queries are identical between Contex and dense.

## Finding

Across **two** standard public benchmarks now — HotpotQA (semantic) and SciFact (lexical) — **Contex's
hybrid retrieval is on par with a plain dense retriever using the same embeddings, and comparable to
BM25.** There is no measurable retrieval-quality advantage from Contex's hybrid layer on either regime.

The honest implication: **the retrieval-ranking axis does not differentiate Contex.** Contex's value is
not "it ranks better than dense/BM25" — it's operational (schema-free publish/subscribe, no manual
retrieval pipeline, continuous re-routing as data changes). A static labeled-retrieval benchmark cannot
measure that, and these two runs confirm it also cannot manufacture a ranking win where there isn't one.

## Caveat

SciFact claims are full-sentence prose, so semantic embeddings do well and the "lexical advantage"
is muted. The sharpest lexical test is **short exact-token / identifier** retrieval over near-duplicate
distractors (e.g. `SERVICE_TIMEOUT_MS` vs `cfg_timeout_prod`) — the shape of Contex's *own* golden eval.
That regime might still favor FTS, but (a) a public dataset of that shape is what would make it rigorous,
and (b) even a win there would validate "hybrid > dense on exact-match," not "Contex > other hybrids."
Two public benchmarks tying is already strong evidence that retrieval ranking is not Contex's edge.

## Reproducing

```
python -m contexeval.prep_beir scifact test          # 5183 docs, 300 test queries
python scripts/run_beir.py 10 scifact-full           # publish + recall@10 + bootstrap CIs
```

## Follow-up: Contex's hybrid ties dense because of an FTS bug (not "hybrid can't help")

Prompted by the question "isn't this just a hybrid-tuning problem?", we dug deeper. It is a
Contex-implementation problem — and a working hybrid *does* win here.

**1. Complementary signal exists.** Oracle union (gold in BM25-top-10 ∪ dense-top-10) = **0.873**
recall vs 0.79 for either alone (+0.08 headroom). BM25 rescues 24 dense misses; dense rescues 23
BM25 misses (of 300). So there is real lexical signal a hybrid could add.

**2. Contex's FTS returns almost nothing on multi-term queries.** Contex builds its lexical query
with `plainto_tsquery('english', query)`, which **AND's every term** (`src/core/lexical_search.py`).
Verified against the live `scifact-full` project: single terms match fine (`to_tsquery('cell')` →
2,529 docs), but full-sentence SciFact claims match **0–1 docs** (all terms required). So the lexical
half contributes ~nothing and the "hybrid" collapses to dense-only — exactly matching the 287/300
Contex≈dense ties. The one hybrid knob Contex exposes (`RRF_K`) cannot fix this: RRF only reorders the
candidate union; it cannot add candidates FTS never returned.

**3. A correct hybrid wins.** Simulating Contex's own RRF fusion (`rrf_fuse`, k=60) but feeding it a
real partial-match lexical retriever (`rank-bm25`) instead of the AND-broken FTS:

| method | recall@10 |
|---|---|
| dense | 0.783 |
| bm25 | 0.776 |
| RRF(bm25, dense) | **0.824** |

Paired bootstrap: **RRF-hybrid − dense = +0.041, 95% CI [+0.003, +0.080] (excludes 0), W/T/L 25/259/16.**

**Conclusion.** Contex's hybrid *design* is sound — a correctly-implemented version beats dense on
SciFact, significantly. Its *implementation* has a real defect: `plainto_tsquery` requires all query
terms, neutralizing the lexical half for multi-word queries, so the shipped hybrid ties dense. **Fix =
change Contex's FTS query construction to BM25-style / OR partial matching** (a code change in
`src/core/lexical_search.py`), not a parameter tune. This is a concrete, independently-surfaced bug in
Contex — arguably the most actionable output of the whole validation.

## Re-eval on Contex v0.2.x (2026-09-13): the OR fix is necessary but insufficient

Contex shipped the fix for #138 (OR the `plainto_tsquery` lexemes, exactly as suggested). Re-ran the
SAME SciFact setup (identical 5,183-doc index — reused via the DB volume, only the FTS *query* changed;
300 queries, k=10, hybrid). Result — the fix did **not** deliver the simulated hybrid win, and slightly
regressed recall + cost:

| method | recall@10 | context tokens | vs prior |
|---|---|---|---|
| contex (v0.2.x, OR fix) | **0.768** | **4,640** | was 0.792 / 3,314 (broken AND) |
| bm25 | 0.776 | 3,525 | — |
| dense | 0.783 | 3,349 | — |

Paired bootstrap: contex−dense = **−0.015 [−0.043, +0.013]** (tie, W/T/L 8/275/17); contex−bm25 =
−0.007 [−0.052, +0.037]. So Contex still ties dense — now marginally *below*, and ~40% more expensive.

**Why (confirmed):** the OR query now matches **2,920 / 1,999 of 5,183** docs for sample queries (vs 0
under AND) — i.e. it matches ~half the corpus on any shared common term. Postgres `ts_rank_cd` ranks
that broad set weakly (no real IDF weighting), so RRF fusion of dense + noisy-lexical *displaces* good
dense hits (recall ↓) and pulls in longer docs (cost ↑).

**Reconciliation with the earlier simulation.** Our simulated "correct hybrid" hit 0.824 using
**rank-bm25** as the lexical retriever (proper BM25/IDF ranking). Contex fuses using `ts_rank_cd` over a
plain OR match. Same RRF, same dense — the only difference is the lexical **ranker**, so the ranker is
decisively the bottleneck: AND→OR fixed *which docs match*, but not *how well they're ranked*.

**Recommendation (follow-up to #138):** the minimal OR fix stops the silent degrade-to-vector-only, but
to realize the hybrid gain Contex needs a real BM25 ranker (e.g. ParadeDB `pg_search`/`pg_bm25`) or an
IDF-weighted / min-should-match lexical query — not `ts_rank_cd` over a broad OR. This was flagged as the
"more thorough option" in the original report; the SciFact re-eval now shows it's the *necessary* one.

## Re-eval on Contex v0.2.5 (ParadeDB / pg_search BM25) — the hybrid win lands

v0.2.5 switched Contex's lexical retriever to **ParadeDB pg_search BM25** (`paradedb.score` via the
`@@@` operator over `description`/`data_original`), replacing `ts_rank_cd`. Fresh DB, re-published the
same SciFact corpus (5,183 docs) so the BM25 index is built over it; same 300 queries, k=10, hybrid.

| method | recall@10 | context tokens |
|---|---|---|
| **contex (ParadeDB BM25)** | **0.841** | 3,421 |
| dense | 0.783 | 3,349 |
| bm25 (rank-bm25 baseline) | 0.776 | 3,525 |

Paired bootstrap (10k resamples):
- **contex − dense = +0.058, 95% CI [+0.024, +0.092] — excludes 0** (W/T/L 25/266/9)
- **contex − bm25  = +0.065, 95% CI [+0.033, +0.100] — excludes 0** (W/T/L 28/267/5)

**Contex's hybrid now significantly beats both pure dense and pure keyword**, at comparable token cost —
and it exceeded the `RRF(rank-bm25, dense) = 0.824` simulation (0.841), i.e. ParadeDB's BM25 + Contex's
RRF fusion did even better than the rank-bm25 proxy predicted.

### The full arc (why this is the Contex-specific validation)

| Contex version | lexical retriever | recall@10 | vs dense (95% CI) |
|---|---|---|---|
| original (as first tested) | `plainto_tsquery` (AND all terms → matched ~0) | 0.792 | tie |
| v0.2.x (#138 fix) | OR'd `plainto_tsquery` + `ts_rank_cd` (matched ~½ corpus, weak rank) | 0.768 | tie / slightly below |
| **v0.2.5** | **ParadeDB `pg_search` BM25** | **0.841** | **+0.058 (excludes 0)** |

This is the retrieval-quality result that *is* Contex-specific: with a real BM25 ranker fused into its
hybrid, Contex measurably and significantly out-retrieves both the dense baseline (Contex's own
embeddings) and pure BM25 on a public benchmark — the thing neither the broken-AND nor the OR-`ts_rank_cd`
versions could do. The validation harness drove the whole loop: measure → surface the FTS bug (#138) →
the OR fix proved insufficient → ParadeDB BM25 → significant, reproducible hybrid win.

## Honest caveats & effect size (read before quoting the number)

- **"Significant" ≠ "dominant."** The win is *reliable* but *narrow in reach*: ~266/300 queries are
  identical to dense; the effect rides on ~25 flipped queries (~8%). "+0.058 recall@10" is a true, clean
  number — it means "reliably better on the minority of queries where lexical signal exists," not "beats
  dense across the board." The CIs justify *significant*, not *dominant*.
- **The dense baseline is deliberately modest — and it is Contex's own embedder.** `dense` =
  `all-MiniLM-L6-v2` (384-dim, 2021). This is not a hand-picked weak baseline to flatter Contex; it is the
  embedder Contex actually ships, so `contex − dense` isolates fusion gain *at Contex's real operating
  point*. **⚠️ ANSWERED by the embedder ablation (`2026-09-14-embedder-ablation.md`): the win does NOT
  survive a modern embedder.** A 2023 model (bge-base) *alone* significantly beats this hybrid
  (−0.050, CI excludes 0); the win documented above holds only against Contex's weak MiniLM embedder. The
  highest-leverage retrieval change for Contex is upgrading the embedder, not the hybrid.
- **Regime note.** SciFact here is the **full 5,183-doc BEIR corpus** (no subsampling). The HotpotQA
  companion result runs the **distractor-pool** regime (paragraphs pooled across sampled questions) —
  standard for HotpotQA but not the full-wiki corpus, and at a different `k` / #-gold-per-query. Both are
  defensible; the two datasets are *not* identical setups, so "two public benchmarks" ≠ "one protocol."
