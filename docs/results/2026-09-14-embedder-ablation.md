# Embedder-strength ablation — does the hybrid win survive a modern embedder?

**Date:** 2026-09-14
**Answer: No.** The earlier "Contex hybrid beats dense" result was an artifact of Contex shipping a weak
2021 embedder (all-MiniLM-L6-v2, 384-dim). Swap in a modern embedder and **plain dense alone significantly
beats Contex's hybrid**, and fusing BM25 into a strong dense retriever stops helping (and can hurt).

## Method

Local retrieval comparison (`scripts/embedder_ablation.py`), no Contex — same corpora/queries as the
main runs. `hybrid_*` are `RRF(rank-bm25, dense)`, Contex's own fusion; actual ParadeDB Contex tracked
this simulation (0.841 vs 0.824 on SciFact), so it stands in faithfully for Contex-current. Strong
embedder = **BAAI/bge-base-en-v1.5** (768-dim, 2023). Paired bootstrap, 10k resamples, seeded.

> bge-**base**, not bge-large (bge-large hung on Apple MPS and was impractically slow on CPU). This is
> **conservative**: a stronger embedder would only *widen* the gap below, strengthening the conclusion.

## Results

**SciFact — recall@10 (n=300):**

| method | recall@10 |
|---|---|
| bm25 | 0.776 |
| dense_minilm *(Contex's shipped embedder)* | 0.783 |
| **dense_bge-base *(modern embedder, alone)*** | **0.874** |
| hybrid_minilm *(≈ Contex now)* | 0.824 |
| hybrid_bge *(≈ Contex if it upgraded)* | 0.858 |

**HotpotQA — recall@5 (n=150):**

| method | recall@5 |
|---|---|
| bm25 | 0.723 |
| dense_minilm | 0.767 |
| **dense_bge-base** | **0.877** |
| hybrid_minilm *(≈ Contex now)* | 0.787 |
| hybrid_bge *(≈ Contex if it upgraded)* | 0.840 |

**Paired bootstrap 95% CIs:**

| comparison | SciFact | HotpotQA |
|---|---|---|
| dense_bge − dense_minilm (embedder upgrade alone) | +0.091 [+0.054,+0.129] ✅ | +0.110 [+0.073,+0.150] ✅ |
| hybrid_minilm − dense_bge (Contex-now vs modern dense) | −0.050 [−0.087,−0.013] ✅ | −0.090 [−0.127,−0.053] ✅ |
| hybrid_bge − dense_bge (upgraded hybrid vs modern dense) | −0.016 [−0.049,+0.018] (ns) | −0.037 [−0.063,−0.010] ✅ |

(✅ = 95% CI excludes 0.)

## Findings

1. **The embedder upgrade dominates.** MiniLM → bge-base buys **+9 to +11 recall points** — far more than
   the BM25 hybrid ever contributed (~+6). The highest-leverage retrieval change for Contex is a better
   embedder, not the hybrid.
2. **A modern dense model alone significantly beats Contex's current hybrid** on both datasets. The
   "Contex-specific hybrid win" documented earlier holds *only against Contex's own weak embedder*.
3. **Fusion's value evaporates (then reverses) as dense improves.** With a strong dense side, adding BM25
   is neutral on SciFact (−0.016, ns) and **significantly harmful on HotpotQA** (−0.037). Standard IR
   result: BM25 rescues a weak retriever and adds noise to a strong one.

## Actionable takeaways for Contex

- **Upgrade the default embedder** (all-MiniLM-L6-v2 → a modern model). This is the single biggest
  retrieval lever by a wide margin (+9–11 pts), and it dwarfs the hybrid.
- **The BM25 hybrid earns its keep mainly as a rescue for weak embeddings.** With a modern embedder it is
  neutral-to-negative, so it should probably be **optional / down-weighted** (RRF is unweighted today), not
  on by default — otherwise it becomes a small liability.
- **Don't position Contex on retrieval-ranking superiority.** It doesn't survive a modern embedder. Lead
  on the operational/cost story (lean routing vs dump-all) and the rigor of this validation instead.

## Why this matters

This is the value of adversarial-against-yourself evaluation: the earlier win was real *as measured* but
fragile to an obvious challenge ("use a better embedder"). Running that challenge **before** publishing
turned a claim that would have collapsed under a skeptic's afternoon into an honest, defensible position.

## Reproducing

```
python -m contexeval.prep_beir scifact test && CONTEXEVAL_DEVICE=cpu python scripts/embedder_ablation.py 10 SciFact
python -m contexeval.prep 150            && CONTEXEVAL_DEVICE=cpu python scripts/embedder_ablation.py 5  HotpotQA
```
