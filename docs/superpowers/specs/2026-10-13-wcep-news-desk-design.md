# WCEP News Desk — Design

**Date:** 2026-10-13
**Status:** Draft for review.
**Supersedes:** the static-retrieval benchmarks (SciFact, HotpotQA) as the main line of work. Those
answered their question: on one-shot retrieval, fixed Contex v1 ties dense.

## 1. Purpose

Measure Contex on the job it now claims: agents subscribe once to a standing need, data keeps
arriving, and each agent reads current context without querying. The test is a global news desk.
Three years of news is replayed in date order into one Contex project, and hundreds of analyst
agents each follow one story. Every simulated day, each agent reports what changed in its story.

The questions, in order:

1. **Delivery.** Does each agent's subscription hold the articles that matter for its story that
   day?
2. **Briefing quality.** Does the agent's daily briefing match what a human editor wrote about that
   story that day?
3. **Cost.** What does each method spend to get there: searches run, tokens read, and wall time?
4. **Freshness under load.** How long does a new article take to reach the subscriptions it
   belongs in, and how does that grow with the number of subscriptions and the ingest rate?

Expected result, stated up front: Contex's delivery should roughly tie dense re-query, because a
subscription read is a search re-run when data changes. If Contex wins, it wins on cost: agents run
no searches, and the server re-matches only the subscriptions an article can change. Questions 3
and 4 are where the result is decided.

## 2. Non-goals

- No self-authored labels, needs, or answers. Needs are generated mechanically from Wikipedia
  entity names, and gold comes from WCEP editors.
- No live front-end or demo.
- No paid model APIs. The agent is the local `MlxAgent`.
- No tuning Contex for the benchmark. It runs stock at the current upstream commit, recorded per
  run.

## 3. Dataset: WCEP

[WCEP](https://github.com/complementizer/wcep-mds-dataset) (Ghalandari et al., ACL 2020). It comes
as one Google Drive folder with `train/val/test.jsonl.gz` (490 MB), and needs no registration.

| | |
|---|---|
| Events | 10,200, 2016-08-25 to 2019-08-28 |
| Per event | date, category, a 30–40 word summary written by a WCEP editor, `wiki_links` (linked entities) |
| Articles | 649,575 in total: 12,204 cited by the editor (`origin: WCEP`) and 637,371 added automatically from Common Crawl (`origin: CommonCrawl`) |
| Entities | 15,452 distinct; 412 are linked by 10 or more events |

All three splits are pooled. Splits don't matter here because nothing is trained.

**Label quality.** Only the editor-cited articles and the summaries are human labels. The Common
Crawl articles were attached to events by the dataset authors' similarity model. Using them as gold
would reward whichever method resembles that model, so they serve only as replay volume and as
distractors. A secondary metric counts them, and the report labels it as machine-labelled.

## 4. Stories, needs, and gold

- **Stories.** Each entity linked by at least 10 events is one story: 412 subscriptions, under
  Contex's cap of 1000 per project. The very generic entities (`United_States`,
  `President_of_the_United_States`) stay in and are reported separately, because hard needs are part
  of the test.
- **Need text.** It is generated from the entity's Wikipedia title only:
  `"news about {title}"`, for example `"news about Brexit"`. Nobody hand-writes needs.
- **Gold for each (story, day).** The events dated that day whose `wiki_links` include the entity.
  - Gold articles are those events' editor-cited articles.
  - The gold briefing is the concatenation of those events' editor summaries.
  - A (story, day) pair with no event is a **quiet day**, where the correct briefing is "nothing
    new".
- **Known limitation.** An event can link an entity in passing. Those pairs stay in as labelled,
  with no filtering by us, and the report says so.

## 5. Replay

- **Order.** Articles are sorted by date and published to one Contex project in daily batches. The
  date comes from the article's `publish_data`, or the event date when that's missing.
- **Text.** Each article is published as `title + lead`, capped at 1,000 characters. Every method
  sees the same text. This keeps the embedding load near one node per article.
- **Clock.** Contex's sliding window (`max_age_seconds`) is measured against the server's wall
  clock, so the replay runs on a scaled clock.
  - One simulated day takes S wall seconds.
  - Day d's articles are published with `published_at = t0 + (d - d0)·S`.
  - Subscriptions use `max_age_seconds = W·S` for a window of W days.
  - The driver sleeps when it gets ahead of the clock. When it falls behind, the delay is logged as
    ingest lag and not hidden.
  - S is set from ingest throughput measured on a small run.
- **Window.** W = 3 days. The briefing asks what is new, and three days gives a story's follow-ups
  time to land.
- **Subscription expiry.** Subscriptions expire an hour after the last read, and each read resets
  that. Agents read every simulated day, which keeps them alive as long as S stays well under an
  hour. The driver checks this anyway.

## 6. Methods

Every method uses the same stories, the same window, the same `k = 10`, the same agent, and the
same prompt.

| Method | Each simulated day, per story |
|---|---|
| `contex-sub` | Read the subscription resource. The agent runs no search. |
| `contex-query` | Call `contex_query` with the need and the window. Same engine, no subscription. This separates the subscription mechanism from Contex's ranking. |
| `dense` | Cosine top-k over the articles in the window, with the same embedder Contex runs. |
| `bm25` | `rank-bm25` top-k over the articles in the window. This is the keyword-alert baseline. |
| `snapshot` | Top-k fetched once on the story's first day and never refreshed. A floor. |
| `dump-window` | Every article in the window. Used only as a token-cost reference, never run through the agent. |

`dense` and `bm25` search only the articles inside the window (a few thousand), so both stay cheap
at full scale. They reuse `contexeval/retrievers`.

## 7. Metrics

1. **Delivery**, for every (story, day) with gold:
   - Recall@10 against the editor-cited articles.
   - Event hit rate: at least one article from a gold event in the top 10, where articles of all
     origins count. This one is reported as machine-labelled.
   - Quiet-day load: the number of in-window articles returned on quiet days. This is
     informational only, because a fixed k always returns k.
2. **Briefing quality**, on a fixed sample of 300 (story, day) pairs with gold, plus 100 quiet
   days:
   - The agent writes a briefing of at most 60 words from its context, or "nothing new".
   - Scored with ROUGE-1/2/L and BERTScore against the gold briefing, which are WCEP's own metrics.
   - Quiet days are scored on whether the agent correctly said "nothing new".
   - The sample is seeded and the same for every method.
3. **Cost:**
   - Agent side: searches issued and context tokens read per agent turn.
   - Server side: CPU seconds of the Contex and Postgres containers (cgroup `cpu.stat`) per run.
     Contex's Prometheus metrics cover HTTP only. `contex-sub` and `contex-query` ingest the same
     articles, so the gap between their CPU totals is the cost of subscriptions against per-day
     queries.
   - Wall time for both sides.
4. **Freshness under load:**
   - Lag: wall time from publishing a gold article to its first appearance in a subscription,
     measured by polling the subscriptions that have gold in that day's batch every 0.25 s.
   - Publish latency, measured end to end, including the wait for subscriptions to update.
   - Both are swept over the number of subscriptions (50, 200, 412) and the ingest rate (×1, ×4,
     ×16 articles per wall second) on one fixed replay slice.

Every comparison between methods gets a paired bootstrap 95% CI over (story, day) pairs, reusing
`scripts/analyze.py`.

## 8. Scale tiers

| Tier | Slice | Stories | Articles | Purpose |
|---|---|---|---|---|
| T0 | 1 week | 20 | ~4k | Smoke test; measures ingest throughput, which sets S |
| T1 | 3 months | stories with gold in the slice | ~50k | First full result |
| T2 | all 3 years | 412 | 650k | Headline, if T1 throughput puts it within a day of wall time |

The load sweep in metric 4 runs on the T1 slice.

## 9. Module layout

New code only. It reuses `contex_client`, `agent`, `retrievers`, `scoring`, `tokens`, and
`analyze`.

- `contexeval/prep_wcep.py`: reads the gz files and writes `data/wcep/articles.jsonl` (date-sorted),
  `stories.jsonl`, and `gold.jsonl`, keyed by (story, day).
- `contexeval/replay.py`: the scaled-clock publisher plus per-day agent turns for each method, and
  the source of the cost and lag logs.
- `scripts/run_newsdesk.py`: the CLI. Takes a tier, methods, and sweep parameters, and writes to
  `results/newsdesk/<run>/`.
- Tests: `prep_wcep` gold derivation on a hand-built 3-event fixture; the replay clock mapping;
  quiet-day detection.

## 10. Open questions for the plan

- **Ingest throughput** of the current upstream Contex on 1,000-character articles, natively on
  arm64. This decides S and whether T2 fits.
- **Fallback window.** If W = 3 days misses most follow-ups at T0, try W = 1, which is a "today
  only" briefing.
