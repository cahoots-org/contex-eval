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
