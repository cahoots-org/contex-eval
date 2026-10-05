# Contex v1 on SciFact: hybrid regressed below dense; cause is the RRF over-fetch (#236)

**Date:** 2026-10-04
**Contex:** upstream `d25b593` (v1.0.1+, 179 commits past the v0.2.5 runs). Stock config: gte-base
via ONNX, hybrid on, `RRF_K=60`.
**Answer:** Contex v1 hybrid retrieval is **significantly worse than plain dense** on SciFact. It
scores 0.825 against 0.890, while v0.2.5 with the same embedder scored 0.879. Replaying Contex's own
rankings offline puts the whole drop on one change: since #236, each ranker feeds 100 candidates into
RRF instead of `top_k`. Fusing at depth `top_k` gives 0.882 again.

## Result: live Contex v1, SciFact recall@10 (n=300, 5,183 docs)

| method | recall@10 | context tokens |
|---|---|---|
| dense (gte-base, plain title+text) | **0.890** | 3,263 |
| contex v1 (hybrid) | 0.825 | 3,515 |
| bm25 (rank-bm25) | 0.776 | 3,525 |

Paired bootstrap, 10k resamples:
- contex − dense = **−0.066, [−0.102, −0.029]**, excludes 0 (W/T/L 7/263/30)
- contex − bm25 = +0.049, [+0.015, +0.085], excludes 0

On the same embedder, v0.2.5 tied dense (−0.011, CI included 0;
[`2026-09-15-contex-gtebase-real.md`](2026-09-15-contex-gtebase-real.md)). v1 is a regression.

## Diagnosis

The loss splits into two parts.

**1. The vector side costs about 3 points, and this is not new.** Run with
`HYBRID_SEARCH_ENABLED=false`, Contex scores 0.860. The JSON parser turns each doc into a single node
whose embedding text is `"<id>\nroot | para_id: <id> | title: … | text: …"`. Embedding that exact
string locally with sentence-transformers gte-base also gives 0.860. Plain `title text` gives 0.890.
The field-label scaffolding is what costs recall. The ONNX embedder, the HNSW index and the #219
doc-key prefix are not the cause: without the prefix the local score is 0.853. Leaf JSON objects
convert the same way they did in v0.2.5, so v0.2.5 paid this cost too.

**2. Fusion now subtracts recall where it used to add it.** `scripts/fusion_replay.py` pulls each
query's top-100 vector ranking and top-100 ParadeDB BM25 ranking straight from the Contex database
and replays RRF offline:

| RRF input depth per ranker | recall@10 |
|---|---|
| vector only | 0.860 |
| BM25 only (ParadeDB) | 0.768 |
| **10 (= top_k; ≈ v0.2.5)** | **0.882** |
| 20 | 0.867 |
| 30 | 0.861 |
| 50 | 0.856 |
| **100 (v1: `top_k × candidate_pool_factor`)** | **0.825** |
| 100, lexical-only hits dropped (pre-#187) | 0.835 |

Depth 100 reproduces the live v1 number exactly. Paired bootstrap:
- depth 10 − depth 100 = **+0.058, [+0.025, +0.092]**, excludes 0
- depth 10 − vector-only = +0.022, [−0.005, +0.051]
- depth 100 − vector-only = −0.036, [−0.075, +0.004]

**Mechanism.** #236 over-fetches `top_k × 10` candidates so that collapsing to one result per
document still fills `top_k`. The over-fetch happens *before* fusion, so RRF now fuses two 100-deep
lists. In RRF a document's score is the sum of `1/(60+rank)` over the rankers that returned it.
With deep lists, many mediocre BM25 hits at ranks 20–100 also sit somewhere in the vector top-100,
and the two modest contributions add up to more than one strong vector rank. The weaker ranker
(BM25 alone: 0.768) gets more say in the top 10 as the lists get deeper. The sweep falls steadily
as depth grows. Restoring #187's drop of lexical-only hits recovers only a point, so the depth is
the cause.

**Confirmed live.** Contex already exposes the over-fetch as `CANDIDATE_POOL_FACTOR` (default 10).
Restarting the same published project with `CANDIDATE_POOL_FACTOR=1`, so each ranker returns `top_k`,
scores **0.882** live: +0.058 [+0.025, +0.092] over stock v1, matching the replay exactly. That is back
to v0.2.5's level (0.879) and within a point of dense (0.890). For corpora with one node per document
(SciFact, HotpotQA), `CANDIDATE_POOL_FACTOR=1` is a complete workaround today.

## Fix direction (for Contex)

Fuse at depth `top_k` and over-fetch only for the collapse step. One way is to cut each ranker's
list to `top_k` before `rrf_fuse`, then load extra candidates for documents that collapse. Another is
to over-fetch whole documents rather than nodes. Re-run `fusion_replay.py` after publishing to check
any variant without a full re-eval.

## Side finding: the HNSW result cap (no recall effect here)

`vector_search` runs `ORDER BY embedding <=> q LIMIT 100` with a `project_id` filter. With pgvector's
default `hnsw.ef_search = 40` the index scan returns **at most 40 rows**, and the project filter is
applied afterwards (confirmed with EXPLAIN ANALYZE: `rows=40` under `LIMIT 100`). Setting
`hnsw.ef_search=200` and `hnsw.iterative_scan=strict_order` restores 100 rows. It made no difference
to SciFact recall@10 (0.860 / 0.825 either way), because this DB holds almost only one project. In a
multi-project database, other projects' rows take up those 40 slots, so it is worth fixing alongside.

## Caveats

- SciFact only. HotpotQA on v1 has not been run yet. Its docs are also single flat JSON objects, so
  the same mechanism should apply.
- The dense baseline embeds plain `title text`, while Contex embeds its node format. Part of the
  remaining gap (0.882 vs 0.890) is that formatting cost.
- Single run. Confidence comes from the paired bootstrap over 300 queries.

## Reproducing

Eval stack: `contex/` at upstream `d25b593`, built native arm64. Compose project `contexeval` runs
the app on `:8011` with 8 CPUs; see README.

```
CONTEX_MCP_URL=http://127.0.0.1:8011/mcp python scripts/run_beir.py 10 scifact-v1   # ~1 h incl. publish
python scripts/fusion_replay.py scifact-v1 10                                        # fusion diagnosis
```
