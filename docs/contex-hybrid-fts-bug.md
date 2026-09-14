# Bug report (for cahoots-org/contex): hybrid search silently degrades to vector-only because the FTS query AND's every term

## Summary

With `HYBRID_SEARCH_ENABLED=true`, Contex's lexical (full-text) retriever builds its query with
`plainto_tsquery('english', query)`, which requires **every** query term to be present in a document
(`term1 & term2 & …`). For any multi-word need — i.e. most real queries — almost no document contains
all terms, so the FTS side returns 0–1 candidates. The RRF fusion then has nothing to fuse and the
"hybrid" result is effectively identical to pure vector search. Hybrid search therefore provides **no
benefit over vector-only** for multi-term queries, and the one exposed knob (`RRF_K`) cannot fix it
(RRF only re-orders the candidate union; it cannot add candidates FTS never returned).

We measured a **statistically significant retrieval improvement that a correct hybrid would deliver but
Contex currently does not** (details below).

## Affected code

`src/core/lexical_search.py` — `PgFtsLexical` uses:

```sql
SELECT node_key,
       ts_rank_cd(search_text, plainto_tsquery('english', :q)) AS score
  FROM ...
 WHERE search_text @@ plainto_tsquery('english', :q)
 LIMIT :top_k
```

`plainto_tsquery('english', 'a b c')` produces `'a' & 'b' & 'c'` (all terms required). Fusion is
`src/core/rank_fusion.py::rrf_fuse` (standard RRF, k=`RRF_K`, default 60); config knobs are only
`HYBRID_SEARCH_ENABLED` and `RRF_K` (`src/core/config.py`).

## Reproduction (SQL, against a populated project)

Corpus: BEIR SciFact (5,183 docs) published to a project, `HYBRID_SEARCH_ENABLED=true`.

```sql
-- The FTS index is fine — single terms match plenty:
select count(*) from embeddings
 where project_id = :proj and search_text @@ to_tsquery('english','cell');        -- 2529
select count(*) from embeddings
 where project_id = :proj and search_text @@ to_tsquery('english','treatment');   -- 911

-- But a real (multi-term) query AND's every term and matches almost nothing:
select count(*) from embeddings
 where project_id = :proj
   and search_text @@ plainto_tsquery('english',
       'All hematopoietic stem cells segregate their chromosomes randomly');       -- 0
select count(*) from embeddings
 where project_id = :proj
   and search_text @@ plainto_tsquery('english',
       '0-dimensional biomaterials show inductive properties');                    -- 0
```

So for typical queries the FTS half returns nothing → hybrid == vector-only.

## Impact, measured (BEIR SciFact, 300 test queries, k=10)

- **Contex hybrid recall@10 = 0.792**; **vector-only (same `all-MiniLM-L6-v2` embeddings) = 0.783.**
  Contex ties pure vector search and returns the *same* docs on 287/300 queries — i.e. the lexical
  half contributes nothing.
- **There is real complementary lexical signal being lost.** Oracle union (gold in BM25-top-10 ∪
  dense-top-10) = **0.873** (+0.08 over either alone); a real lexical retriever rescues 24 queries the
  vector side misses.
- **A correct hybrid captures it.** Feeding Contex's own `rrf_fuse` (k=60) a partial-match lexical
  retriever (`rank-bm25`) instead of the AND-broken FTS yields **recall@10 = 0.824**, i.e. **+0.041 over
  vector-only, paired-bootstrap 95% CI [+0.003, +0.080] (excludes 0), W/T/L = 25/259/16.**

The hybrid *design* is sound; the FTS query construction is the bottleneck.

## Suggested fix

Make the lexical query match documents that contain a *subset* of the terms (as BM25 does), and let
`ts_rank_cd` + RRF handle precision. Minimal, low-risk change: reuse `plainto_tsquery`'s tokenization
but OR the lexemes instead of AND-ing them:

```sql
-- build an OR tsquery from plainto's parsed (stemmed, stopword-filtered) terms
WITH q AS (
  SELECT replace(plainto_tsquery('english', :q)::text, ' & ', ' | ')::tsquery AS tsq
)
SELECT node_key, ts_rank_cd(search_text, q.tsq) AS score
  FROM ..., q
 WHERE search_text @@ q.tsq
 LIMIT :top_k
```

This keeps stemming/stopword handling identical, broadens candidates to partial matches, and preserves
`ts_rank_cd` scoring. (A more thorough option is a real BM25 index via a Postgres extension such as
ParadeDB `pg_search`/`pg_bm25`, but the OR change alone recovers most of the lost signal.)

Trade-off to consider: OR broadens the candidate set, so keep the `LIMIT`/`top_k` and rely on
`ts_rank_cd` ordering + RRF fusion for precision. A "minimum-should-match" variant is possible but the
plain OR already yields the significant improvement above.

## How this was found

Independent validation harness (HotpotQA + BEIR SciFact over the MCP interface). Full methodology,
data, and the analysis scripts are at
`https://github.com/robmillersoftware/contex-eval` — see `docs/results/2026-09-12-scifact-keyword-regime.md`.
