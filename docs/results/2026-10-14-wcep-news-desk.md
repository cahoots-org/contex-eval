# WCEP news desk: push subscriptions match polling on freshness at 45% less server CPU

**Date:** 2026-10-14
**Contex:** upstream `feca2de`, run natively with GPU embedding (see Setup).
**Answer:** Hundreds of agents each follow one news story while three months of news is replayed into Contex.
- **Freshness:** Contex's push subscriptions deliver what polling the same engine delivers: recall@10 0.185 vs 0.179, a tie, with similar delivery lag.
- **Cost:** at that same freshness, push uses **45% less server CPU** and replaces every agent search with a read of a stored bundle.
- **Retrieval:** what a subscription holds ties plain dense and BM25 re-queried each day (+0.004 and +0.008, both CIs span 0), and a fetch-once snapshot falls to zero within days.
- **Limits:** the 3-month runs showed that reading every subscription on a fixed schedule, instead of on push, costs more than querying (+29% server CPU). Subscriptions refreshed one at a time also lag the data under load.

## The question

Contex claims agents can subscribe once to what they need and stay current as data changes, with no
fetching and no cache invalidation. Static retrieval benchmarks (SciFact, HotpotQA) can't test that,
and Contex ties dense on them anyway. This benchmark tests the claim directly. A large stream of
real news arrives over time, many agents each hold a standing need, and every result is scored
against labels written by human editors.

## Data and labels

[WCEP](https://github.com/complementizer/wcep-mds-dataset) (Ghalandari et al., ACL 2020) has 10,200
news events from 2016-08 to 2019-08. Each event has a summary written by a Wikipedia Current Events
editor and its linked Wikipedia entities. The dataset holds 649,575 articles in all.

- **Stories:** each entity linked by at least 10 events is a story, 412 in all. The need text is
  `"news about {title}"`, generated mechanically. Nobody hand-wrote any needs.
- **Gold:** for a (story, day), the gold set is the articles editors cited for that day's events
  that link the story.
  - Only the 12,204 editor-cited articles count as gold.
  - The other 637k articles were attached to events by the dataset authors' similarity model, so
    they serve only as volume and distractors. If they were gold, they would favour embedding
    methods.
- **Article dates:** an article's date comes from `publish_data`, or from `time` if that's missing,
  but only within 7 days of its event. Times further away are archive or crawl dates, so the
  article takes its event's date.
- **Scoring:**
  - Recall@10 counts only gold articles dated inside the 3-day window, since those are the only
    ones any method could deliver.
  - Gold dated outside the window is reported separately as "undeliverable".
  - "Event hit" (any article from a gold event in the top 10) uses machine labels and is reported
    only as context.

## Setup

- **Replay:** articles are published to one Contex project in date order. Each article is its title
  plus one body paragraph, capped at 1,000 characters, and every method sees the same text.
- **Scaled clock:** Contex measures its sliding windows against the server's wall clock, so one
  simulated day lasts S seconds of wall time.
  - Articles are stamped with their day's start time on that clock.
  - Windows are W = 3 days (`max_age_seconds = 3·S`).
  - Every day is read at 0.9·S. A day whose read lands late is excluded and reported; none were late.
- **Retrieval settings:** k = 10 for every method, and Contex calls pass `threshold=0` so every method
  returns its plain top 10.
- **Contex runs natively with GPU embedding.**
  - In Docker on this Mac, Contex embeds on CPU at about 3 articles/s, and a 3-month replay would
    take about 30 hours per pass. ONNX Runtime's CoreML provider failed to embed 200 texts in 30
    minutes.
  - `scripts/contex-mps.patch` adds an opt-in `TorchEmbedder`: the same gte-base model with the same
    truncation, pooling and normalisation, running on MPS (Apple's GPU backend). It uses a lock,
    because Contex encodes from worker threads and concurrent MPS calls crash the process.
  - Its embeddings match ONNX's to cosine ≥ 0.99976 on 200 WCEP articles.
  - A one-week trial gave the same recall in Docker and natively (0.235 both). 97% of top-10 entries
    matched.
  - Postgres stays in the compose stack. `scripts/contex_native.sh` starts Contex, and
    `scripts/contex-native.env` holds its settings.
- **Sweep:** Contex removes articles that have aged out of a subscription only when it re-matches
  that subscription, either on a relevant publish or on its periodic sweep. The sweep runs every
  S/10 (`RECONCILE_SWEEP_SECONDS`, recorded in each run's `meta.json`), and the run script refuses
  slower settings.
- **Server CPU:** Contex is measured as host process CPU time, which excludes GPU time. Postgres is
  measured as container cgroup CPU time.
- **Statistics:** paired bootstrap over (story, day) pairs, 10k resamples.

## 1. Push vs polling at equal freshness

This is how Contex is meant to be used. Each agent listens on its subscription
(`subscriptions/listen`). Contex pushes `resources/updated` when the bundle changes, and the agent
reads the bundle only then. The polling agent stays equally fresh by calling `contex_query` after
every batch of new data. Both run against the same engine.

The slice is 14 days from 2018-03-18, with the 162 stories that have gold in it. Each day's articles
arrive in 6 batches, 40 s apart, at S = 300 s (`--mode push` and `--mode pull --pull-every 1`).

| | push | poll after every batch |
|---|---|---|
| Server CPU: Contex + Postgres | **7,829 s** (1,021 + 6,808) | 14,333 s (1,496 + 12,837) |
| Agent-side operations | 10,054 bundle reads | 15,552 hybrid searches |
| Recall@10 held at 0.9·S (353 story-days) | **0.185** | 0.179 |
| Event hit (machine-labelled) | 0.850 | 0.853 |
| Delivery lag of gold articles, p50 / p90 | 21.7 s / 106 s | 15.1 s / 111 s |
| Gold reaching an agent's top 10 at all | 95 / 436 | 99 / 436 |

- **Cost:** push uses 45% less server CPU and halves the Postgres work. Polling runs every agent's
  search after every batch whether or not anything relevant arrived. Push re-matches only the
  subscriptions a new article could change, and agents read only bundles that changed (about 3.9
  per agent per day here, against 6 polls).
- **Freshness:** about equal.
  - Recall: push − poll = +0.006, 95% CI [+0.000, +0.014].
  - Delivery lag ties at p90.
  - At the median, push is about 7 s slower, because Contex re-matches the affected subscriptions
    one after another under a per-project lock (see §3).
- **This comparison favours polling.** Here, data arrives in 6 discrete batches a day, so polling
  only has to keep pace with those. With data arriving continuously, matching push's freshness
  would mean polling more often, and the gap would widen.

## 2. Three months, 338 stories

The slice is 90 days from 2018-03-18, the 90-day window with the most gold, with the 338 stories
that have gold in it. That's about 89k articles, 1,915 scored story-days with deliverable gold, and
S = 180 s. In these runs every agent reads once per simulated day, at 0.9·S:
- **contex-sub:** reads its subscription.
- **contex-query:** calls `contex_query`.
- **dense and bm25:** re-run over the same 3-day window offline.
- **snapshot:** keeps the dense top 10 from day one.

| method | recall@10 | event hit (machine) |
|---|---|---|
| contex-query | **0.191** | 0.840 |
| contex-sub | 0.183 | 0.831 |
| dense (gte-base) | 0.179 | 0.802 |
| bm25 | 0.175 | 0.819 |
| snapshot | 0.001 | 0.018 |

contex-sub − baseline:

| vs | mean | 95% CI |
|---|---|---|
| dense | +0.004 | [−0.010, +0.019] |
| bm25 | +0.008 | [−0.007, +0.024] |
| contex-query | **−0.008** | [−0.013, −0.003] |
| snapshot | **+0.182** | [+0.166, +0.199] |

- **Live context matters.** A snapshot fetched on day one holds almost none of the later stories,
  so every live method beats it by 0.18.
- **On retrieval quality, Contex ties plain dense and BM25**, as it does on SciFact and HotpotQA.
- **Recall is low in absolute terms for every method.** Editor-cited gold is about one article per
  event, and broad stories (Trump, China) have hundreds of on-topic articles in each window. 163
  gold articles were dated outside their window and are reported as undeliverable.

**What a fixed-schedule subscription read costs.** These runs read every subscription once a day
instead of on push, which wastes the mechanism:

| | contex-sub | contex-query |
|---|---|---|
| Server CPU: Contex + Postgres | 22,588 s (3,628 + 18,960) | 17,455 s (2,171 + 15,284) |
| Ingest time per simulated day, median | 58 s | 27.5 s |
| Day-end read for 338 agents, median | **2.5 s** | 24 s |

Subscriptions cost 29% more server CPU here. Every subscription is kept current whenever data
arrives, but each one is read only once a day. That wasted refresh work is exactly what push avoids
(§1). Agent-side reads are still about 10× faster.

## 3. Problems found in Contex

1. **Subscription bundles lag the data under load.**
   - In the 3-month run, against what a fresh query returned at the same moment, subscription
     bundles were missing 13.1% of the same day's articles and 14.9% of the previous day's.
   - They held older articles instead (23k extra 2-day-old articles), and 0.5% of what they held had
     already left the window.
   - `reconcile_project` re-matches the affected subscriptions one after another while holding a
     per-project lock, and the periodic sweep takes the same lock. With 338 subscriptions on an
     index of about 89k articles, a full pass takes longer than the data takes to change.
   - The same queueing is behind push's slower median in §1.
2. **Ingest slows with the number of subscriptions.**
   - On the same 3-month slice, native GPU ingest ran at a median 33.9 articles/s with no
     subscriptions (the query pass) and 15.9 articles/s with 338 subscriptions.
   - The re-matching that follows each publish competes with ingest for the same CPU and database.
3. **The compose file sets no `shm_size` for Postgres.**
   - Docker gives containers 64 MB of `/dev/shm` by default.
   - With 16 concurrent `contex_query` calls, parallel query workers fail with `could not resize
     shared memory segment … No space left on device`, and the tool call errors out. That crashed
     the first 3-month query pass on day 48.
   - `shm_size: 1gb` fixes it.
4. **A listening agent doesn't keep its subscription alive.**
   - Only reads renew the one-hour lease, so a pushed-to agent on a quiet story loses its
     subscription.
   - Contex does send a final event, and re-creating the subscription resumes the same URI, so this
     is workable. But a push-only client has to know to do it.
   - No subscription expired in these runs.

## Caveats

- **Native GPU embedding:** the main runs used the MPS patch, so absolute CPU and throughput numbers
  describe this Mac running natively, not the Docker image. Contex CPU excludes GPU time. The
  push-vs-poll comparison uses the same setup on both sides.
- **Scaled clock:** S is 180–300 s, and the sweep interval scales with it.
- **Labels:**
  - Editor-cited gold is sparse, and an event can link an entity only in passing.
  - Event hit depends on the dataset authors' similarity matching.
  - "Quiet" days (no event linking a story) are not reliable labels for broad stories, so the
    planned briefing and quiet-day scoring was not run.
- **Sample size:** one replay per configuration. Confidence comes from paired bootstrap over
  (story, day).
- **Not run:**
  - the full 3-year replay, about 18 hours per pass at GPU speed
  - the sweep of subscription count against ingest rate
  - polling every 3 batches and once a day, which matter only if push lost on cost

## Reproducing

```
pip install gdown && gdown --folder https://drive.google.com/drive/folders/1T5wDxu4ajFwEq77dG88oE95e8ppREamg -O data/wcep/raw
python -m contexeval.prep_wcep data/wcep/raw data/wcep
# Postgres from the compose stack with "127.0.0.1:5442:5432" and shm_size: 1gb in contex/docker-compose.override.yml
git -C contex apply ../scripts/contex-mps.patch && ./scripts/contex_native.sh &
export CONTEX_MCP_URL=http://127.0.0.1:8011/mcp
# §1 (RECONCILE_SWEEP_SECONDS=18 in scripts/contex-native.env)
python scripts/run_newsdesk.py --native --mode push --start 2018-03-18 --days 14 --sec-per-day 300 --run pp-push
python scripts/run_newsdesk.py --native --mode pull --pull-every 1 --start 2018-03-18 --days 14 --sec-per-day 300 --run pp-pull-k1
python scripts/analyze_newsdesk.py results/newsdesk/pp-push results/newsdesk/pp-pull-k1
# §2
python scripts/run_newsdesk.py --native --mode sub   --start 2018-03-18 --days 90 --sec-per-day 180 --no-lag --run t1-sub
python scripts/run_newsdesk.py --native --mode query --start 2018-03-18 --days 90 --sec-per-day 180 --run t1-query
python scripts/run_newsdesk.py --local --start 2018-03-18 --days 90 --run t1-local
python scripts/analyze_newsdesk.py results/newsdesk/t1-sub results/newsdesk/t1-query results/newsdesk/t1-local
```
