# Contex Validation Harness — "Does lean context actually win?"

A research/validation project for [Contex](https://github.com/) (the OSS semantic context bus). Handoff brief — written to be picked up cold in a fresh session.

> **What this is (locked):** A small, credible benchmark harness that measures whether Contex actually delivers on its core promise. Not a toy, not a demo — a real validation with numbers. The [Code Youngstown](https://www.meetup.com/code-youngstown/) lightning talk is the *milestone/forcing function*, **not** the deliverable. The deliverable is the metrics. If the numbers are good, they become the talk's closing punch ("here's the pain, here's Contex, and here's the data proving it helps").

> **Supersedes the old `creature-feeder.md` demo brief.** That was a presenter-driven game to *illustrate* Contex's routing. This pivots to *validating* Contex. The creature demo is deprioritized — at most it's an optional, playful front-end for the talk once the metrics exist. Don't build it first; don't let it drive decisions.

## The point of Contex (this is what we're validating)

Contex exists to kill the manual work of context-assembling for agents: fetching data, reshaping it into something a model can read, and — the hard part — giving the agent **enough** context without giving it **too much**. Contex does that continuously via semantic matching: producers publish schema-free data, consumers subscribe to *needs* in plain English, matching happens on meaning.

That promise has **two faces of one mechanic**:
- **Routing** (data's-eye view): each item flows to whichever need it's relevant to.
- **Curation / "not too much"** (agent's-eye view): each agent only ever receives what's relevant, bounded — no firehose.

Both faces are the *same* semantic-matching event seen from two sides. The validation measures both, plus what they cost.

## The claim, as three metrics (locked)

We prove (or fail to prove) the promise with exactly three numbers. Nothing else.

1. **Retrieval accuracy** — did Contex route the *right* context to the need? Precision / recall / F1 of retrieved items against labeled ground-truth relevance.
2. **Answer quality** — did the lean context let an agent actually *do the job* better? Task correctness (exact-match / F1) with an agent in the loop, fed each method's context.
3. **Cost** — tokens (context + generation). A local model has no per-token price, so **tokens is the honest headline metric**; optionally translate to "what this would've cost on a hosted model" using public pricing as an illustrative figure only.

Together: the plumbing works (1), it *matters* (2), and it's cheaper (3). Three numbers, hard to hand-wave.

## Why a public benchmark, not homemade data (locked)

The original instinct was to author a relatable dev-exhaust corpus (Slack/GitHub/email). Rejected as the *primary* dataset because of the **self-graded-homework problem**: if you author the corpus AND the needs AND the relevance labels AND the answers, a skeptic rightly says "you built the test so Contex would pass." For a validation that has *never actually been done*, that makes the numbers worthless. Use an established benchmark with independent ground truth instead.

## The benchmark: HotpotQA, distractor setting (locked)

Chosen because a single example already bundles **every** ingredient the three metrics need, *and* ships official scorers — so we don't invent the grading:

| What we need | What HotpotQA gives |
|---|---|
| A corpus of items to route | 10 paragraphs per question: 2 gold + 8 distractors |
| A need / query | The question |
| Relevance labels (→ retrieval accuracy) | Sentence-level **supporting facts** (gold evidence) + official supporting-fact F1 |
| A checkable answer (→ answer quality) | Gold answer span / yes-no + official answer EM/F1 |

~100–200 questions is enough (standard for RAG-eval labeled sets). Pre-derived retrieval-eval versions exist (e.g. **StratRAG**, HotpotQA-distractor with 2 gold + 13 distractors) if a more turnkey packaging is wanted.

Sources: [HotpotQA overview](https://www.emergentmind.com/topics/hotpotqa), [StratRAG (derived)](https://arxiv.org/html/2604.22757v1), [RAG eval guidance](https://labelyourdata.com/articles/llm-fine-tuning/rag-evaluation), [Anthropic: context rot / effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents).

## Critical design decision: pool the corpus (locked)

**Do NOT run each question against only its own 10 paragraphs.** Finding 2 relevant items out of 10 is a trivial retrieval task — dump-all, keyword, and Contex would all score nearly the same, and it wouldn't exercise the thing Contex is *for* (routing at scale, out of "too much").

Instead: **pool the paragraphs across all sampled questions into one shared Contex bus.** ~200 questions × ~10 paragraphs ≈ **a few thousand documents** (dedup). Each question subscribes as a `need` against the *full* pool and must find its 2 gold paragraphs among thousands. Real haystack, multi-hop intact, and it's genuinely testing routing-at-scale. (This is essentially HotpotQA's "fullwiki" setting, minus indexing all of Wikipedia.)

**Pool size target: a few thousand docs** — deliberately chosen so `dump-all` is still *runnable* as a baseline (fits a long-context window, just wastefully) rather than infeasible. Big enough that dumping is painful; small enough that dumping is measurable. Don't push to full-Wikipedia scale or you lose `dump-all` as a comparison point.

### Scoring stays honest under pooling
- **Recall is exact.** The 2 gold paragraphs are labeled; Contex either surfaced them or didn't.
- **Precision is honest.** Any non-gold paragraph retrieved counts against the method. Cross-question paragraphs are almost all off-topic (different entities), so treating unlabeled-as-non-relevant is safe. Residual risk — a stray cross-question paragraph that's genuinely relevant but unlabeled — is small; **disclose it** as a known limitation. Standard IR practice.
- **Bonus:** pooling makes the cost story *stark*. `dump-all` on thousands of docs is wildly expensive, so Contex's token win goes from modest to dramatic, and you can show the **context-rot effect** (answer accuracy *dropping* as you dump more) instead of merely asserting it.

## Baselines (what Contex is measured against)

Same pooled corpus, same fixed agent, same questions — only the retrieval method changes:
- **`dump-all`** — feed the whole pool. The "too much" baseline. By definition scores **100% retrieval recall** (contains every gold para) at **maximal cost** → this is the reference ceiling and a built-in sanity check.
- **`BM25 top-k`** — the "hand-rolled keyword filter" baseline.
- **`dense top-k`** *(optional)* — isolates what Contex's hybrid + threshold adds over plain vector similarity.

## Fair-comparison protocol for retrieval (locked)

Contex returns a **threshold**-based bundle (variable size); baselines return **top-k** (fixed size). To compare fairly:
- Report **Contex at its default threshold**.
- Set baseline **`k` = Contex's average bundle size** (so no method gets an unfair token budget).
- Also **sweep** threshold (Contex) and `k` (baselines) to produce a **precision-recall curve** — this is how you *show* Contex auto-sizing to the right amount instead of needing a hand-picked `k`.

## The agent model: local via mlx-lm (locked — no paid APIs)

Budget constraint: **no Claude / hosted APIs.** Serve one local model on Apple Silicon with `mlx-lm`.

- **Model:** `mlx-community/Qwen2.5-7B-Instruct-4bit`. This is a *batch* run (not the live demo), so latency is irrelevant — we can afford a 7B for answer quality that reflects **context** quality, not model weakness. **Held constant across every condition**; the only variable is the context each method supplies.
- **RAM gate (OPEN — confirm before pinning):** 7B-4bit assumes **16GB+**. On an 8GB machine, drop to a 3–4B (`Qwen2.5-3B-Instruct-4bit`).
- **Serving:** `mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080` → OpenAI-compatible `/v1/chat/completions`. All conditions point an OpenAI client at `localhost:8080`. `mlx_lm.server` serves sequentially — fine for a batch run.
- Pre-warm with a dummy call so first-token weight-load isn't counted.

## Architecture — the harness (six pieces)

Contex is imported as a **published GitHub dependency** (not forked). The harness is a thin test rig around a black box.

1. **Dataset prep** — download HotpotQA distractor dev set; sample ~200 questions (**fixed seed**); pool + dedup paragraphs into the shared corpus (~few thousand docs). Keep per-question: question, 2 gold paragraph IDs, gold answer, gold supporting facts.
2. **Contex adapter** — publish the whole pooled corpus into Contex once as schema-free items. For each question, create a subscription with the question as the `need`; read back the matched bundle (= Contex's routed context). *(Confirm the actual publish/subscribe/read API against the repo — see open questions.)*
3. **Baseline retrievers** — `dump-all`, `BM25 top-k`, optional `dense top-k`, over the same corpus.
4. **Answer agent** — the fixed mlx-lm model: `(retrieved context + question) → answer`. Constant across all conditions.
5. **Scorers** — retrieval: precision/recall/F1 on gold paragraphs (HotpotQA supporting-fact scoring); answer: EM/F1 (official); cost: context + generation tokens (+ optional $ translation).
6. **Runner + report** — loop questions × methods → aggregate table + precision-recall plot.

**Data flow:** `benchmark → pooled corpus → [Contex | dump-all | BM25 | dense] → context → fixed agent → answer → scorers → report`

**Error handling:** per-question `try/except` (one failure skips, doesn't kill the run); **cache** embeddings/publishes so reruns are cheap; fixed seed for the sample.

**Testing:** end-to-end smoke test on ~5 questions; unit-test the scorers against known HotpotQA examples to confirm they match official numbers; assert `dump-all` = 100% retrieval recall (the built-in sanity check).

## Open questions for the build session
- **RAM → final model size.** Confirm the laptop's RAM; pin 7B-4bit (16GB+) vs 3–4B (8GB).
- **Contex's actual API surface.** Confirm from the GitHub repo: how to publish items, create a subscription/need, read the matched bundle, and what the **default similarity threshold** is (and whether it's tunable for the sweep). Python client vs HTTP endpoints.
- **Exact pool size + question count** — "a few thousand docs" ≈ 200–300 questions after dedup; pin a number that keeps `dump-all` runnable.
- **Include `dense top-k`?** Optional; adds the "what does Contex add over plain vectors" story. Decide based on effort.
- **$ translation pricing** — which hosted model's public pricing to cite for the illustrative cost figure (talk only).
- **Talk scope** — date TBD; decide how much (if any) live front-end to build once metrics exist. Default: none until numbers are in.

## Locked decisions (summary)
- **Validation is the deliverable; the talk is the milestone.** Metrics first.
- **Three metrics only:** retrieval accuracy, answer quality, cost (tokens).
- **Public benchmark, not homemade data** (avoids self-graded homework). **HotpotQA distractor.**
- **Pool the corpus** into one shared bus (~few thousand docs); per-question needs against the full pool. NOT per-question 10-doc episodes.
- **Baselines:** dump-all, BM25 top-k, optional dense top-k. `dump-all` = recall sanity check.
- **Fair protocol:** Contex at default threshold; baseline `k` = Contex avg bundle size; sweep for PR curve.
- **Local model via mlx-lm** (`Qwen2.5-7B-Instruct-4bit`, RAM-gated), held constant. **No paid APIs.**
- **Contex imported from GitHub**, not forked.

## Pointers
- Contex library: GitHub (published) — import as dependency; confirm API surface.
- Benchmark: HotpotQA distractor dev set (+ official scorer); optional StratRAG packaging.
- Local serving: `mlx-lm` (`pip install mlx-lm`; `mlx_lm.server`), models from the `mlx-community` HF org.
- Prior/superseded demo brief: `../contex-demo/creature-feeder.md` (context only; deprioritized).
