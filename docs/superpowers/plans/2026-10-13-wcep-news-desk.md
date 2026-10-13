# WCEP News Desk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replay WCEP news into Contex on a scaled clock, with one subscription per entity story,
and score what each method delivers and briefs against WCEP editor labels, along with cost and
freshness under load.

**Architecture:**
- `prep_wcep` turns the WCEP dumps into date-sorted articles, stories and per-(story, day) gold.
- `replay` publishes each simulated day to Contex on a wall clock and records what each Contex
  method (subscription read or per-day query) holds at the end of the day.
- Windowed local baselines (dense, BM25, snapshot) run offline over the same days.
- Briefings and scoring run offline afterwards, from the recorded article ids.

**Tech Stack:** Python 3.12, `mcp==2.0.0`, sentence-transformers (`thenlper/gte-base`),
`rank-bm25`, `mlx-lm` (agent), `rouge-score`, `bert-score`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-13-wcep-news-desk-design.md`

## Global Constraints

- Contex runs as the isolated compose project `contexeval` on `:8011`. Never touch `:8001`.
- Contex is stock upstream (currently `feca2de`). Record `git -C contex rev-parse --short HEAD` in
  every run's `meta.json`.
- `k = 10` for every method; Contex calls pass `threshold=0.0` so every method returns its plain
  top-k.
- Window `W = 3` days: day d sees articles dated d-2, d-1 and d.
- Need text is exactly `f"news about {title}"`, where `title` is the unquoted Wikipedia slug with
  underscores replaced by spaces.
- Story threshold: an entity linked by at least `10` events across all splits.
- Published text: `(title + "\n\n" + body)[:1000]`, sent as `data_format="text"`.
- Gold articles: `origin == "WCEP"` only. `CommonCrawl` articles are never gold.
- Seed `13` for every sample.
- Briefing sample: `300` gold pairs + `100` quiet pairs. Briefings are at most 60 words.
- Embedder for local dense: `contexeval.config.EMBED_MODEL` (`thenlper/gte-base`, same as Contex).
- Contex env for replays: `RECONCILE_SWEEP_SECONDS = sec_per_day / 4`, set in the local
  `contex/docker-compose.override.yml`. Disclose it in the results doc.
- Contex limits to respect: 1000 subscriptions per project, `top_k ≤ 100`, ≤1000 items per publish
  batch, subscription TTL 3600 s (renewed on read).

## Review Focus

1. **Gold article dated outside its event's window.** Editor-cited articles can be dated days
   before the event. Recall must count only *deliverable* gold (dated inside the window), and
   report undeliverable gold separately. It must not be scored as a miss. Pinned in Task 5.
2. **Replay falling behind the clock.** If ingest is slower than `sec_per_day`, each day's
   `behind_s` must be logged and the run must continue. It must not silently skip days or sleep a
   negative time. Pinned in Task 3.
3. **Missing or odd article fields.** A `publish_data` of `None`, `"None"` or an unparseable value,
   or a `None` title or text, must fall back to the event date or empty string, not crash. Pinned in
   Task 1.
4. **A subscription read failing (expired or errored).** It must raise with the story named, never
   be recorded as an empty delivery. Pinned in Task 2 (client raises) and Task 4 (driver
   propagates).
5. **Days with more than 1000 articles.** These are published in chunks of ≤500 so Contex's batch
   cap and its affected-subscription filter both hold. Pinned in Task 3.

---

## File Structure

| File | Responsibility |
|---|---|
| `contexeval/prep_wcep.py` (new) | WCEP gz dumps → `articles.jsonl`, `stories.jsonl`, `gold.jsonl` |
| `contexeval/contex_client.py` (modify) | Add `ContexSession`: one long-lived async MCP session (publish batch, subscribe, read bundle, query) |
| `contexeval/replay.py` (new) | `Clock`, `window_days`, `chunks`, and the async Contex replay driver |
| `contexeval/retrievers/windowed.py` (new) | Offline windowed dense / BM25 / snapshot deliveries |
| `contexeval/stats.py` (new) | `boot_ci` moved out of `scripts/analyze.py` so both analyzers share it |
| `contexeval/scoring/delivery.py` (new) | Per-(story, day) recall against deliverable gold, and event hit |
| `contexeval/scoring/briefing.py` (new) | Quiet-day check, ROUGE, BERTScore |
| `contexeval/agent.py` (modify) | `system` parameter on `build_prompt` / `answer` |
| `scripts/run_newsdesk.py` (new) | CLI: one Contex replay (`--mode sub|query`), or `--local` for offline baselines |
| `scripts/brief_newsdesk.py` (new) | Sample pairs, run the agent per method, score briefings |
| `scripts/analyze_newsdesk.py` (new) | Delivery means and CIs, lag percentiles, cost, briefing scores |
| `tests/test_prep_wcep.py`, `tests/test_replay.py`, `tests/test_windowed.py`, `tests/test_delivery.py`, `tests/test_briefing.py`, `tests/test_contex_session.py` (new) | Tests |

Data lives in `data/wcep/` (`raw/` holds the downloaded gz files). Runs write to
`results/newsdesk/<run>/`. Both are already gitignored (`data/**/*.jsonl`, `/results/`), apart from
`raw/*.gz`, which Task 1 adds.

---

### Task 1: WCEP prep

**Files:**
- Create: `contexeval/prep_wcep.py`
- Test: `tests/test_prep_wcep.py`
- Modify: `.gitignore` (add `data/wcep/`)

**Interfaces:**
- Produces:
  - `parse_day(s: str | None) -> str | None`
  - `story_title(link: str) -> str`
  - `prep(clusters: list[dict], lo: str = "2016-08-25", hi: str = "2019-08-28") -> tuple[list[dict], list[dict], list[dict]]`
    returns `(articles, stories, gold)`.
    - Article: `{"id", "day", "text", "origin", "event"}`, sorted by `(day, id)`.
    - Story: `{"story", "title", "need", "n_events"}`.
    - Gold: `{"story", "day", "articles": [ids], "summaries": [str], "events": [int]}`.
  - CLI: `python -m contexeval.prep_wcep data/wcep/raw data/wcep`

- [ ] **Step 1: Download the data (one-off, not committed)**

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1T5wDxu4ajFwEq77dG88oE95e8ppREamg -O data/wcep/raw
ls data/wcep/raw   # train.jsonl.gz  val.jsonl.gz  test.jsonl.gz
```

Add `data/wcep/` to `.gitignore`.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_prep_wcep.py
from contexeval.prep_wcep import parse_day, prep, story_title, MIN_EVENTS

def _art(id, origin="CommonCrawl", pub="Aug 23, 2016", title="T", text="body"):
    return {"id": id, "origin": origin, "publish_data": pub, "title": title, "text": text}

def _cluster(id, date, links, arts, summary="s"):
    return {"id": id, "date": date, "wiki_links": links, "articles": arts, "summary": summary}

def test_parse_day_formats_and_garbage():
    assert parse_day("Aug 23, 2016") == "2016-08-23"
    assert parse_day("2016-08-25") == "2016-08-25"
    assert parse_day(None) is None
    assert parse_day("None") is None
    assert parse_day("sometime") is None

def test_story_title_unquotes():
    assert story_title("/wiki/Rodrigo_Duterte") == "Rodrigo Duterte"
    assert story_title("/wiki/Caf%C3%A9_Society") == "Café Society"

def _many(link, n, start=0):
    # n single-article events linking `link`, so it clears MIN_EVENTS
    # dated 2017-01-01 so they don't share a (story, day) gold key with the clusters under test
    return [_cluster(start + i, "2017-01-01", [link], [_art(f"x{start+i}", pub=None)]) for i in range(n)]

def test_gold_uses_editor_cited_only_and_skips_rare_entities():
    target = _cluster(999, "2017-01-10", ["/wiki/Brexit", "/wiki/Rare"], [
        _art("cited", origin="WCEP", pub="Jan 09, 2017"),
        _art("auto", origin="CommonCrawl", pub="Jan 10, 2017"),
    ], summary="Parliament votes.")
    clusters = _many("/wiki/Brexit", MIN_EVENTS - 1) + [target]
    articles, stories, gold = prep(clusters)
    assert [s["story"] for s in stories] == ["/wiki/Brexit"]
    assert stories[0]["need"] == "news about Brexit"
    g = next(g for g in gold if g["events"] == [999])
    assert g["story"] == "/wiki/Brexit" and g["day"] == "2017-01-10"
    assert g["articles"] == ["cited"]
    assert g["summaries"] == ["Parliament votes."]

def test_article_fields_fallbacks_and_dedup():
    clusters = _many("/wiki/Brexit", MIN_EVENTS) + [
        _cluster(500, "2017-02-01", ["/wiki/Brexit"], [
            _art("a", pub=None, title=None, text=None),
            _art("b", pub="Jan 01, 1999"),          # out of range -> event date
            _art("c", pub="Feb 03, 2017", text="z" * 5000),
        ]),
        _cluster(501, "2017-03-01", ["/wiki/Brexit"], [_art("a", pub="Mar 01, 2017")]),  # dup id
    ]
    by = {a["id"]: a for a in prep(clusters)[0]}
    assert by["a"]["day"] == "2017-02-01" and by["a"]["text"] == "" and by["a"]["event"] == 500
    assert by["b"]["day"] == "2017-02-01"
    assert by["c"]["day"] == "2017-02-03" and len(by["c"]["text"]) == 1000
    days = [a["day"] for a in prep(clusters)[0]]
    assert days == sorted(days)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_prep_wcep.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'contexeval.prep_wcep'`

- [ ] **Step 4: Implement**

```python
# contexeval/prep_wcep.py
"""WCEP (Ghalandari et al., ACL 2020) -> news-desk replay inputs.

Stories are entities linked by >= MIN_EVENTS events. Gold for (story, day) is the events on that
day linking the story: their editor-cited (origin WCEP) articles and editor summaries.
Common Crawl articles were attached by the dataset authors' similarity model, so they are replay
volume, never gold.

Usage: python -m contexeval.prep_wcep data/wcep/raw data/wcep
"""
import gzip
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

MIN_EVENTS = 10
TEXT_CHARS = 1000
SPLITS = ("train", "val", "test")


def parse_day(s):
    for fmt in ("%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except (TypeError, ValueError):
            pass
    return None


def story_title(link: str) -> str:
    return unquote(link.removeprefix("/wiki/")).replace("_", " ")


def prep(clusters, lo="2016-08-25", hi="2019-08-28"):
    counts = Counter(link for c in clusters for link in set(c["wiki_links"]))
    story_ids = {link for link, n in counts.items() if n >= MIN_EVENTS}

    articles: dict[str, dict] = {}
    gold: dict[tuple, dict] = defaultdict(lambda: {"articles": [], "summaries": [], "events": []})
    for c in clusters:
        for a in c["articles"]:
            if a["id"] in articles:  # an article can sit in several clusters; first wins
                continue
            day = parse_day(a.get("publish_data"))
            if day is None or not lo <= day <= hi:
                day = c["date"]
            text = ((a.get("title") or "") + "\n\n" + (a.get("text") or "")).strip()[:TEXT_CHARS]
            articles[a["id"]] = {"id": a["id"], "day": day, "text": text,
                                 "origin": a["origin"], "event": c["id"]}
        cited = [a["id"] for a in c["articles"] if a["origin"] == "WCEP"]
        for link in sorted(set(c["wiki_links"]) & story_ids):
            g = gold[(link, c["date"])]
            g["articles"] += [i for i in cited if i not in g["articles"]]
            g["summaries"].append(c["summary"])
            g["events"].append(c["id"])

    stories = [{"story": l, "title": story_title(l), "need": f"news about {story_title(l)}",
                "n_events": counts[l]} for l in sorted(story_ids)]
    gold_rows = [{"story": s, "day": d, **g} for (s, d), g in sorted(gold.items())]
    return sorted(articles.values(), key=lambda a: (a["day"], a["id"])), stories, gold_rows
```

Note that in the fallback test, `title=None, text=None` gives `"".strip() == ""`, which the test
expects.

```python
# contexeval/prep_wcep.py (continued)
def _read(raw_dir: Path):
    for split in SPLITS:
        with gzip.open(raw_dir / f"{split}.jsonl.gz", "rt") as f:
            for line in f:
                yield json.loads(line)


def _write(path: Path, rows):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    articles, stories, gold = prep(list(_read(raw)))
    _write(out / "articles.jsonl", articles)
    _write(out / "stories.jsonl", stories)
    _write(out / "gold.jsonl", gold)
    print(f"{len(articles)} articles, {len(stories)} stories, {len(gold)} gold (story, day) pairs")
```

`prep` takes a list because it makes two passes. With text capped at 1,000 characters, 650k
articles fit in memory.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_prep_wcep.py -v`
Expected: 4 passed

- [ ] **Step 6: Run prep on the real data and sanity-check the counts**

Run: `python -m contexeval.prep_wcep data/wcep/raw data/wcep`
Expected: about 649k articles (fewer after dedup), **412 stories**, and several thousand gold pairs.
If the story count isn't 412, stop and investigate, because the spec's numbers came from the same
counting rule.

- [ ] **Step 7: Commit**

```bash
git add contexeval/prep_wcep.py tests/test_prep_wcep.py .gitignore
git commit -m "feat(prep): WCEP -> news-desk articles, stories, and editor gold"
```

---

### Task 2: Long-lived async Contex session

**Files:**
- Modify: `contexeval/contex_client.py` (append the class; leave `ContexClient` untouched)
- Test: `tests/test_contex_session.py`

**Interfaces:**
- Produces: `ContexSession(url: str = CONTEX_MCP_URL, project_id: str = CONTEX_PROJECT_ID)`, an
  async context manager with:
  - `async publish_batch(items: list[dict]) -> None`. Each item is `{"data_key", "data", "data_format", "published_at"}`.
  - `async subscribe(need: str, top_k: int, max_age_seconds: int) -> str`. Returns the resource URI.
  - `async read_bundle(uri: str) -> list[str]`. Returns document ids, best first. Raises
    `RuntimeError` naming the URI on any failure.
  - `async query(need: str, top_k: int, max_age_seconds: int) -> list[str]`. Returns document ids,
    best first.
  - Module-level `doc_ids(matches: list[dict]) -> list[str]`, the pure helper both of the above use.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_contex_session.py
import asyncio
import uuid
import pytest
from contexeval.contex_client import ContexSession, doc_ids

def test_doc_ids_prefers_document_and_dedups_in_order():
    ms = [{"data_key": "a/p1", "document": "a"}, {"data_key": "b"}, {"data_key": "a/p0", "document": "a"}]
    assert doc_ids(ms) == ["a", "b"]

@pytest.mark.integration
def test_publish_subscribe_read_and_query_roundtrip():
    async def go():
        async with ContexSession(project_id=f"nd-{uuid.uuid4().hex[:8]}") as c:
            uri = await c.subscribe("news about volcanoes", top_k=10, max_age_seconds=3600)
            assert await c.read_bundle(uri) == []
            await c.publish_batch([
                {"data_key": "v1", "data": "Volcano erupts\n\nLava flows in Iceland.", "data_format": "text", "published_at": None},
                {"data_key": "f1", "data": "Football final\n\nThe cup was won.", "data_format": "text", "published_at": None},
            ])
            for _ in range(40):  # reconcile runs in the background since Contex #457
                if "v1" in await c.read_bundle(uri):
                    break
                await asyncio.sleep(0.25)
            assert (await c.read_bundle(uri))[0] == "v1"
            assert (await c.query("news about volcanoes", top_k=10, max_age_seconds=3600))[0] == "v1"
            with pytest.raises(RuntimeError, match="contex://subscriptions/nope"):
                await c.read_bundle("contex://subscriptions/nope")
    asyncio.run(go())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_contex_session.py -v -m "integration or not integration"`
Expected: FAIL with `ImportError: cannot import name 'ContexSession'`

- [ ] **Step 3: Implement (append to `contexeval/contex_client.py`)**

```python
from contextlib import AsyncExitStack


def doc_ids(matches: list[dict]) -> list[str]:
    out: list[str] = []
    for m in matches:
        d = m.get("document") or m["data_key"]
        if d not in out:
            out.append(d)
    return out


class ContexSession:
    """One MCP session held for a whole replay. ContexClient opens a session per call, which is
    fine for one-shot evals but too slow for hundreds of reads per simulated day."""

    def __init__(self, url: str = CONTEX_MCP_URL, project_id: str = CONTEX_PROJECT_ID):
        self.url = url
        self.project_id = project_id

    async def __aenter__(self):
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(streamable_http_client(self.url))
        self.s = await self._stack.enter_async_context(ClientSession(read, write))
        await self.s.initialize()
        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()

    async def _call(self, tool: str, args: dict) -> str:
        r = await self.s.call_tool(tool, {"project_id": self.project_id, **args})
        if r.is_error:
            raise RuntimeError(f"{tool} failed: {r.content[0].text}")
        return r.content[0].text

    async def publish_batch(self, items: list[dict]) -> None:
        await self._call("contex_publish_batch", {"items": items})

    async def subscribe(self, need: str, top_k: int, max_age_seconds: int) -> str:
        return json.loads(await self._call("contex_create_subscription", {
            "needs": [need], "top_k": top_k, "threshold": 0.0, "max_age_seconds": max_age_seconds,
        }))["resource_uri"]

    async def read_bundle(self, uri: str) -> list[str]:
        try:
            r = await self.s.read_resource(uri)
            bundle = json.loads(r.contents[0].text)
        except Exception as e:
            raise RuntimeError(f"reading {uri} failed: {e}") from e
        return doc_ids([m for ms in bundle.values() for m in ms])

    async def query(self, need: str, top_k: int, max_age_seconds: int) -> list[str]:
        return doc_ids(json.loads(await self._call("contex_query", {
            "query": need, "top_k": top_k, "threshold": 0.0, "max_age_seconds": max_age_seconds,
        }))["matches"])
```

If `read_resource` rejects a plain `str` URI under `mcp==2.0.0`, wrap it with `pydantic.AnyUrl(uri)`.
The integration test is what catches this.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_contex_session.py -v` (offline test only)
Expected: 1 passed, 1 deselected

Run, with the `:8011` stack up and `CONTEX_MCP_URL=http://127.0.0.1:8011/mcp`:
`pytest tests/test_contex_session.py -v -m integration`
Expected: 1 passed. If a bundle comes back with `document` missing, or with node keys that don't
match the published `data_key`, fix `doc_ids` here before moving on.

- [ ] **Step 5: Commit**

```bash
git add contexeval/contex_client.py tests/test_contex_session.py
git commit -m "feat(client): long-lived async ContexSession for replays (publish, subscribe, read, query)"
```

---

### Task 3: Replay clock, windows, chunking

**Files:**
- Create: `contexeval/replay.py` (pure helpers only in this task)
- Test: `tests/test_replay.py`

**Interfaces:**
- Produces:
  - `Clock(t0: float, d0: str, sec_per_day: float)` (frozen dataclass) with:
    - `slot_start(day) -> float`
    - `read_time(day) -> float`, which is `slot_start + 0.9 * sec_per_day`
    - `published_at(day) -> str` (ISO-8601 UTC)
    - `max_age_seconds(window_days) -> int`
  - `window_days(day: str, w: int) -> list[str]`: the w days ending at `day`, oldest first.
  - `chunks(xs: list, n: int = 500) -> list[list]`
  - `async sleep_until(t: float) -> float`: returns seconds behind (0.0 if on time); never sleeps a
    negative amount.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_replay.py
import asyncio
import time
from datetime import datetime
from contexeval.replay import Clock, chunks, sleep_until, window_days

def test_clock_maps_days_to_wall_slots():
    c = Clock(t0=1000.0, d0="2017-01-01", sec_per_day=60.0)
    assert c.slot_start("2017-01-01") == 1000.0
    assert c.slot_start("2017-01-03") == 1120.0
    assert c.read_time("2017-01-03") == 1120.0 + 54.0
    assert datetime.fromisoformat(c.published_at("2017-01-02")).timestamp() == 1060.0
    assert c.max_age_seconds(3) == 180

def test_window_at_read_time_covers_exactly_w_days():
    c = Clock(t0=0.0, d0="2017-01-01", sec_per_day=60.0)
    day = "2017-01-05"
    start = c.read_time(day) - c.max_age_seconds(3)  # what Contex computes server-side
    inside = [d for d in ["2017-01-02", "2017-01-03", "2017-01-04", "2017-01-05"]
              if c.slot_start(d) >= start]
    assert inside == window_days(day, 3) == ["2017-01-03", "2017-01-04", "2017-01-05"]

def test_chunks_respect_batch_cap():
    xs = list(range(1203))
    cs = chunks(xs, 500)
    assert [len(c) for c in cs] == [500, 500, 203] and sum(cs, []) == xs

def test_sleep_until_reports_lag_and_never_sleeps_negative():
    t = time.time()
    behind = asyncio.run(sleep_until(t - 5))
    assert 4.9 < behind < 6 and time.time() - t < 0.5
    assert asyncio.run(sleep_until(time.time() + 0.05)) == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_replay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'contexeval.replay'`

- [ ] **Step 3: Implement**

```python
# contexeval/replay.py
"""Scaled-clock replay of the news desk into Contex.

Contex's sliding windows (max_age_seconds) are measured against the server's wall clock, so
simulated day d occupies the wall slot [t0 + (d - d0)·S, t0 + (d - d0 + 1)·S). Articles are
published at their slot start; agents read at 90% of the slot.
"""
import asyncio
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

READ_AT = 0.9


@dataclass(frozen=True)
class Clock:
    t0: float
    d0: str
    sec_per_day: float

    def slot_start(self, day: str) -> float:
        return self.t0 + (date.fromisoformat(day) - date.fromisoformat(self.d0)).days * self.sec_per_day

    def read_time(self, day: str) -> float:
        return self.slot_start(day) + READ_AT * self.sec_per_day

    def published_at(self, day: str) -> str:
        return datetime.fromtimestamp(self.slot_start(day), timezone.utc).isoformat()

    def max_age_seconds(self, window_days: int) -> int:
        return math.ceil(window_days * self.sec_per_day)


def window_days(day: str, w: int) -> list[str]:
    d = date.fromisoformat(day)
    return [(d - timedelta(days=i)).isoformat() for i in range(w - 1, -1, -1)]


def chunks(xs: list, n: int = 500) -> list[list]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


async def sleep_until(t: float) -> float:
    delay = t - time.time()
    if delay > 0:
        await asyncio.sleep(delay)
        return 0.0
    return -delay
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_replay.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add contexeval/replay.py tests/test_replay.py
git commit -m "feat(replay): scaled clock, day windows, batch chunking"
```

---

### Task 4: Contex replay driver and CLI

**Files:**
- Modify: `contexeval/replay.py` (add the driver)
- Create: `scripts/run_newsdesk.py`
- Test: `tests/test_replay.py` (add driver tests using a fake session)

**Interfaces:**
- Consumes: `ContexSession` (Task 2): `publish_batch`, `subscribe`, `read_bundle`, `query`. Also
  `Clock`, `chunks`, `sleep_until` (Task 3).
- Produces:
  - `async run_contex(session, mode: str, clock: Clock, days: list[str], by_day: dict[str, list[dict]], stories: list[dict], gold_ids: dict[tuple[str, str], set[str]], k: int, w: int, out) -> None`
    - `mode` is `"sub"` or `"query"`.
    - `out` is an object with `delivery(row)`, `lag(row)` and `day(row)` methods.
  - Delivery row: `{"method": "contex-sub"|"contex-query", "story", "day", "ids": [..]}`.
  - Lag row (sub mode only): `{"story", "day", "article", "lag_s": float|None}`. `None` means it
    hadn't appeared by read time.
  - Day row: `{"day", "n_articles", "behind_s", "publish_s", "read_s"}`.
  - CLI writes `results/newsdesk/<run>/{deliveries,lag,days}.jsonl` and `meta.json`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_replay.py (append)
import pytest
from contexeval.replay import run_contex

class FakeSession:
    def __init__(self, fail_read=False):
        self.published, self.subs, self.fail_read = [], {}, fail_read
    async def subscribe(self, need, top_k, max_age_seconds):
        self.subs[f"contex://subscriptions/{need}"] = need
        return f"contex://subscriptions/{need}"
    async def publish_batch(self, items):
        self.published.append(items)
    async def read_bundle(self, uri):
        if self.fail_read:
            raise RuntimeError(f"reading {uri} failed: expired")
        return [i["data_key"] for b in self.published for i in b][:2]
    async def query(self, need, top_k, max_age_seconds):
        return ["q"]

class Sink:
    def __init__(self): self.d, self.l, self.days = [], [], []
    def delivery(self, r): self.d.append(r)
    def lag(self, r): self.l.append(r)
    def day(self, r): self.days.append(r)

STORIES = [{"story": "/wiki/A", "need": "news about A"}, {"story": "/wiki/B", "need": "news about B"}]
BY_DAY = {"2017-01-01": [{"id": "a1", "text": "x"}], "2017-01-02": [{"id": f"b{i}", "text": "y"} for i in range(1203)]}

def _run(session, mode, gold=None):
    clock = Clock(t0=time.time(), d0="2017-01-01", sec_per_day=0.2)
    sink = Sink()
    asyncio.run(run_contex(session, mode, clock, sorted(BY_DAY), BY_DAY, STORIES,
                           gold or {}, k=10, w=3, out=sink))
    return sink

def test_sub_mode_publishes_in_chunks_and_records_every_story_day():
    s = FakeSession()
    sink = _run(s, "sub", gold={("/wiki/A", "2017-01-01"): {"a1"}})
    assert [len(b) for b in s.published] == [1, 500, 500, 203]
    assert s.published[0][0]["published_at"].startswith("20")
    assert {(r["story"], r["day"]) for r in sink.d} == {(st["story"], d) for st in STORIES for d in BY_DAY}
    assert all(r["method"] == "contex-sub" for r in sink.d)
    assert sink.l == [{"story": "/wiki/A", "day": "2017-01-01", "article": "a1", "lag_s": sink.l[0]["lag_s"]}]
    assert sink.l[0]["lag_s"] is not None
    assert [r["day"] for r in sink.days] == sorted(BY_DAY) and all(r["behind_s"] >= 0 for r in sink.days)

def test_query_mode_never_subscribes():
    s = FakeSession()
    sink = _run(s, "query")
    assert s.subs == {} and {r["method"] for r in sink.d} == {"contex-query"}
    assert sink.l == []

def test_failed_read_raises_with_story_named():
    with pytest.raises(RuntimeError, match="sub read failed for story /wiki/"):
        _run(FakeSession(fail_read=True), "sub")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_replay.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_contex'`

- [ ] **Step 3: Implement the driver (append to `contexeval/replay.py`)**

```python
READ_CONCURRENCY = 16
LAG_POLL_S = 0.25


async def _gather_limited(coros, n=READ_CONCURRENCY):
    sem = asyncio.Semaphore(n)
    async def one(c):
        async with sem:
            return await c
    return await asyncio.gather(*(one(c) for c in coros))


async def _watch_lag(session, uri, story, day, articles, published_t, deadline, out):
    """Poll one subscription until each gold article in today's batch shows up, or read time."""
    pending = set(articles)
    while pending and time.time() < deadline:
        seen = set(await session.read_bundle(uri)) & pending
        for a in sorted(seen):
            out.lag({"story": story, "day": day, "article": a, "lag_s": time.time() - published_t})
        pending -= seen
        await asyncio.sleep(LAG_POLL_S)
    for a in sorted(pending):
        out.lag({"story": story, "day": day, "article": a, "lag_s": None})


async def run_contex(session, mode, clock, days, by_day, stories, gold_ids, k, w, out):
    max_age = clock.max_age_seconds(w)
    uris = {}
    if mode == "sub":
        for st in stories:
            uris[st["story"]] = await session.subscribe(st["need"], top_k=k, max_age_seconds=max_age)

    async def fetch(st):
        try:
            if mode == "sub":
                return await session.read_bundle(uris[st["story"]])
            return await session.query(st["need"], top_k=k, max_age_seconds=max_age)
        except Exception as e:
            raise RuntimeError(f"{mode} read failed for story {st['story']}: {e}") from e

    for day in days:
        behind = await sleep_until(clock.slot_start(day))
        arts = by_day.get(day, [])
        t = time.time()
        for batch in chunks(arts):
            await session.publish_batch([{"data_key": a["id"], "data": a["text"], "data_format": "text",
                                          "published_at": clock.published_at(day)} for a in batch])
        published_t = time.time()
        watchers = []
        if mode == "sub":
            today = {a["id"] for a in arts}
            for st in stories:
                g = gold_ids.get((st["story"], day), set()) & today
                if g:
                    watchers.append(asyncio.create_task(_watch_lag(
                        session, uris[st["story"]], st["story"], day, g, published_t,
                        clock.read_time(day), out)))
        await sleep_until(clock.read_time(day))
        await asyncio.gather(*watchers)
        r0 = time.time()
        results = await _gather_limited([fetch(st) for st in stories])
        for st, ids in zip(stories, results):
            out.delivery({"method": f"contex-{mode}", "story": st["story"], "day": day, "ids": ids})
        out.day({"day": day, "n_articles": len(arts), "behind_s": behind,
                 "publish_s": published_t - t, "read_s": time.time() - r0})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_replay.py -v`
Expected: 7 passed

- [ ] **Step 5: Write the CLI**

```python
# scripts/run_newsdesk.py
"""One news-desk replay.

  Contex replay:   python scripts/run_newsdesk.py --mode sub   --start 2017-01-01 --days 7 --stories 20 --sec-per-day 60 --run t0-sub
  Local baselines: python scripts/run_newsdesk.py --local      --start 2017-01-01 --days 7 --stories 20 --run t0-local

Each Contex replay uses a fresh project. Server CPU (contex + postgres containers) is read from
cgroup cpu.stat before and after, into meta.json.
"""
import argparse
import asyncio
import json
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from contexeval.config import CONTEX_MCP_URL
from contexeval.contex_client import ContexSession
from contexeval.replay import Clock, run_contex

DATA = Path("data/wcep")
CONTAINERS = ["contexeval-contex-1", "contexeval-postgres-1"]
K, W = 10, 3


def load(name):
    return [json.loads(l) for l in open(DATA / name)]


def cpu_seconds() -> dict[str, float]:
    out = {}
    for c in CONTAINERS:
        stat = subprocess.run(["docker", "exec", c, "cat", "/sys/fs/cgroup/cpu.stat"],
                              capture_output=True, text=True, check=True).stdout
        out[c] = int(next(l.split()[1] for l in stat.splitlines() if l.startswith("usage_usec"))) / 1e6
    return out


def select(args):
    days = [(date.fromisoformat(args.start) + timedelta(days=i)).isoformat() for i in range(args.days)]
    first = (date.fromisoformat(days[0]) - timedelta(days=W - 1)).isoformat()  # pre-roll the window
    by_day = defaultdict(list)
    for a in load("articles.jsonl"):
        if first <= a["day"] <= days[-1]:
            by_day[a["day"]].append(a)
    gold = [g for g in load("gold.jsonl") if days[0] <= g["day"] <= days[-1]]
    n_gold = defaultdict(int)
    for g in gold:
        n_gold[g["story"]] += 1
    # Stories with gold in the slice, most gold first; --stories caps the count.
    stories = sorted((s for s in load("stories.jsonl") if n_gold[s["story"]]),
                     key=lambda s: (-n_gold[s["story"]], s["story"]))[: args.stories]
    replay_days = sorted(set(by_day) | set(days))
    return replay_days, days, by_day, stories, gold


class Sink:
    def __init__(self, d: Path):
        self.f = {n: open(d / f"{n}.jsonl", "w") for n in ("deliveries", "lag", "days")}
    def _w(self, n, r):
        self.f[n].write(json.dumps(r) + "\n"); self.f[n].flush()
    def delivery(self, r): self._w("deliveries", r)
    def lag(self, r): self._w("lag", r)
    def day(self, r): self._w("days", r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sub", "query"])
    p.add_argument("--local", action="store_true")
    p.add_argument("--start", required=True)
    p.add_argument("--days", type=int, required=True)
    p.add_argument("--stories", type=int, default=1000)
    p.add_argument("--sec-per-day", type=float, default=60.0)
    p.add_argument("--run", required=True)
    args = p.parse_args()
    if not (args.local or args.mode):
        sys.exit("pass --mode or --local")
    if args.sec_per_day * W >= 3600:
        sys.exit("window would outlive the 3600 s subscription TTL between reads; lower --sec-per-day")

    out = Path("results/newsdesk") / args.run
    out.mkdir(parents=True, exist_ok=True)
    replay_days, days, by_day, stories, gold = select(args)
    meta = {**vars(args), "k": K, "w": W, "n_stories": len(stories),
            "n_articles": sum(len(v) for v in by_day.values()),
            "contex_commit": subprocess.run(["git", "-C", "contex", "rev-parse", "--short", "HEAD"],
                                            capture_output=True, text=True).stdout.strip()}

    if args.local:
        from contexeval.retrievers.windowed import local_deliveries
        sink = Sink(out)
        for row in local_deliveries(by_day, days, stories, k=K, w=W):
            sink.delivery(row)
    else:
        gold_ids = {(g["story"], g["day"]): set(g["articles"]) for g in gold}
        project = f"newsdesk-{args.run}-{uuid.uuid4().hex[:6]}"
        cpu0, t_start = cpu_seconds(), time.time()
        clock = Clock(t0=time.time() + 5, d0=replay_days[0], sec_per_day=args.sec_per_day)

        async def go():
            async with ContexSession(url=CONTEX_MCP_URL, project_id=project) as s:
                await run_contex(s, args.mode, clock, replay_days, by_day, stories, gold_ids, K, W, Sink(out))
        asyncio.run(go())
        cpu1 = cpu_seconds()
        meta |= {"project": project, "wall_s": time.time() - t_start,
                 "server_cpu_s": {c: cpu1[c] - cpu0[c] for c in CONTAINERS}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
```

The replay starts W−1 days before `--start` so the first scored day has a full window. Deliveries
on those pre-roll days get written but are ignored by analysis, which scores only gold days inside
`--start .. +days`.

`--local` imports `local_deliveries`, which Task 5 creates. Until then, `--local` fails with an
ImportError and `--mode` works.

- [ ] **Step 6: Live smoke (T0 subset, 3 days, 5 stories)**

The `:8011` stack must be up and on `feca2de`. Add `RECONCILE_SWEEP_SECONDS=15` to the `contex`
environment in `contex/docker-compose.override.yml`, then rebuild:

```bash
cd contex && docker compose -p contexeval -f docker-compose.yml -f docker-compose.override.yml up -d --build && cd ..
docker compose -p contexeval ps --format '{{.Name}}'   # confirm the CONTAINERS names in run_newsdesk.py
CONTEX_MCP_URL=http://127.0.0.1:8011/mcp python scripts/run_newsdesk.py --mode sub --start 2017-01-10 --days 3 --stories 5 --sec-per-day 60 --run smoke-sub
```

Expected:
- `results/newsdesk/smoke-sub/deliveries.jsonl` has 5 × 5 rows (3 days + 2 pre-roll).
- `days.jsonl` shows `behind_s` near 0.
- `meta.json` has nonzero `server_cpu_s`.

If `behind_s` grows each day, ingest is slower than 60 s/day at this volume. Note the
`publish_s` values; Task 7 uses them.

- [ ] **Step 7: Commit**

```bash
git add contexeval/replay.py scripts/run_newsdesk.py tests/test_replay.py
git commit -m "feat(replay): Contex replay driver (sub/query) with lag watch and server CPU"
```

---

### Task 5: Windowed local baselines and delivery scoring

**Files:**
- Create: `contexeval/retrievers/windowed.py`, `contexeval/scoring/delivery.py`, `contexeval/stats.py`, `scripts/analyze_newsdesk.py`
- Modify: `scripts/analyze.py` (import `boot_ci` from `contexeval.stats`, and delete its copy)
- Test: `tests/test_windowed.py`, `tests/test_delivery.py`

**Interfaces:**
- Consumes: `window_days` (Task 3). Article and story row shapes (Task 1). Delivery row shape
  (Task 4).
- Produces:
  - `local_deliveries(by_day, days, stories, k, w, embed=None) -> Iterator[dict]`. Yields delivery
    rows for `dense`, `bm25` and `snapshot`. `embed(texts) -> np.ndarray` defaults to normalized
    gte-base.
  - `score_delivery(ids: list[str], gold_articles: list[str], window: set[str], article_event: dict[str, int], gold_events: list[int]) -> dict`.
    Returns `{"recall": float|None, "n_deliverable": int, "n_undeliverable": int, "event_hit": float}`.
  - `boot_ci(diffs: list[float], n: int = 10000, seed: int = 13) -> tuple[float, float]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_windowed.py
import numpy as np
from contexeval.retrievers.windowed import local_deliveries

BY_DAY = {
    "2017-01-01": [{"id": "old", "text": "volcano erupts in iceland"}],
    "2017-01-04": [{"id": "v", "text": "volcano lava iceland"}, {"id": "f", "text": "football cup final"},
                   {"id": "e", "text": "election vote count"}],  # 3 docs so BM25 idf for "volcano" is > 0
}
STORIES = [{"story": "/wiki/Volcano", "need": "news about volcano"}]

def fake_embed(texts):
    vocab = ["volcano", "lava", "iceland", "football", "cup", "news", "about"]
    m = np.array([[t.count(w) for w in vocab] for t in texts], dtype=float) + 1e-9
    return m / np.linalg.norm(m, axis=1, keepdims=True)

def test_window_excludes_old_articles_and_snapshot_freezes():
    rows = list(local_deliveries(BY_DAY, ["2017-01-01", "2017-01-04"], STORIES, k=1, w=3, embed=fake_embed))
    got = {(r["method"], r["day"]): r["ids"] for r in rows}
    assert got[("dense", "2017-01-01")] == ["old"]
    assert got[("dense", "2017-01-04")] == ["v"]          # "old" is outside 01-02..01-04
    assert got[("bm25", "2017-01-04")] == ["v"]
    assert got[("snapshot", "2017-01-04")] == ["old"]     # frozen at the first day
```

```python
# tests/test_delivery.py
from contexeval.scoring.delivery import score_delivery
from contexeval.stats import boot_ci

EVENT = {"g1": 7, "g2": 7, "early": 7, "auto": 7, "x": 9}

def test_recall_counts_only_deliverable_gold():
    s = score_delivery(["g1", "x"], gold_articles=["g1", "g2", "early"],
                       window={"g1", "g2", "x", "auto"}, article_event=EVENT, gold_events=[7])
    assert s["recall"] == 0.5 and s["n_deliverable"] == 2 and s["n_undeliverable"] == 1
    assert s["event_hit"] == 1.0

def test_no_deliverable_gold_gives_none_recall_but_event_hit_still_scored():
    s = score_delivery(["auto"], gold_articles=["early"], window={"auto"},
                       article_event=EVENT, gold_events=[7])
    assert s["recall"] is None and s["event_hit"] == 1.0

def test_boot_ci_brackets_mean():
    lo, hi = boot_ci([0.1, 0.2, 0.3, 0.2, 0.1] * 10)
    assert lo < 0.18 < hi
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_windowed.py tests/test_delivery.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# contexeval/stats.py
import random


def boot_ci(diffs, n=10000, seed=13):
    """95% bootstrap CI on the mean of paired differences."""
    if not diffs:
        return (None, None)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]
```

In `scripts/analyze.py`, delete `random.seed(13)` and the local `boot_ci`, and add
`from contexeval.stats import boot_ci`.

```python
# contexeval/scoring/delivery.py
def score_delivery(ids, gold_articles, window, article_event, gold_events):
    """Recall over gold articles dated inside the window (deliverable); event hit over any origin."""
    deliverable = [a for a in gold_articles if a in window]
    got = set(ids)
    recall = (sum(a in got for a in deliverable) / len(deliverable)) if deliverable else None
    hit = float(any(article_event.get(i) in set(gold_events) for i in ids))
    return {"recall": recall, "n_deliverable": len(deliverable),
            "n_undeliverable": len(gold_articles) - len(deliverable), "event_hit": hit}
```

```python
# contexeval/retrievers/windowed.py
"""Offline per-day baselines over the same W-day window Contex sees. No wall clock needed."""
import os
import numpy as np
from rank_bm25 import BM25Okapi
from ..config import EMBED_MODEL
from ..replay import window_days
from .bm25 import _tok


def _gte():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device=os.getenv("CONTEXEVAL_DEVICE") or None)
    return lambda texts: model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                      batch_size=64, show_progress_bar=len(texts) > 1000)


def local_deliveries(by_day, days, stories, k, w, embed=None):
    embed = embed or _gte()
    arts = [a for d in sorted(by_day) for a in by_day[d]]
    idx = {a["id"]: i for i, a in enumerate(arts)}
    emb = embed([a["text"] for a in arts])
    q = embed([s["need"] for s in stories])
    snapshot = {}
    for day in days:
        ids = [a["id"] for d in window_days(day, w) for a in by_day.get(d, [])]
        if not ids:
            for s in stories:
                for m in ("dense", "bm25"):
                    yield {"method": m, "story": s["story"], "day": day, "ids": []}
                yield {"method": "snapshot", "story": s["story"], "day": day, "ids": snapshot.get(s["story"], [])}
            continue
        rows = np.array([idx[i] for i in ids])
        dense_scores = emb[rows] @ q.T                      # (n_window, n_stories)
        bm25 = BM25Okapi([_tok(arts[r]["text"]) for r in rows])
        for j, s in enumerate(stories):
            dense = [ids[i] for i in np.argsort(-dense_scores[:, j], kind="stable")[:k]]
            sc = bm25.get_scores(_tok(s["need"]))
            bm = [ids[i] for i in np.argsort(-sc, kind="stable")[:k]]
            snapshot.setdefault(s["story"], dense)
            yield {"method": "dense", "story": s["story"], "day": day, "ids": dense}
            yield {"method": "bm25", "story": s["story"], "day": day, "ids": bm}
            yield {"method": "snapshot", "story": s["story"], "day": day, "ids": snapshot[s["story"]]}
```

`snapshot` freezes at each story's first day that has any window articles. For the CLI, that's the
first day of `--days`, because pre-roll days aren't passed in `days`.

```python
# scripts/analyze_newsdesk.py
"""Delivery, lag, cost, and (if present) briefing scores across news-desk runs.

Usage: python scripts/analyze_newsdesk.py results/newsdesk/t1-sub results/newsdesk/t1-query results/newsdesk/t1-local
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from contexeval.replay import window_days
from contexeval.scoring.delivery import score_delivery
from contexeval.stats import boot_ci

DATA = Path("data/wcep")
W = 3


def rows(p):
    return [json.loads(l) for l in open(p)] if p.exists() else []


def main(run_dirs):
    articles = {a["id"]: a for a in rows(DATA / "articles.jsonl")}
    by_day = defaultdict(set)
    for a in articles.values():
        by_day[a["day"]].add(a["id"])
    event = {i: a["event"] for i, a in articles.items()}
    gold = {(g["story"], g["day"]): g for g in rows(DATA / "gold.jsonl")}

    scores = defaultdict(dict)   # method -> (story, day) -> score dict
    for d in map(Path, run_dirs):
        for r in rows(d / "deliveries.jsonl"):
            g = gold.get((r["story"], r["day"]))
            if g is None:
                continue
            window = set().union(*(by_day[x] for x in window_days(r["day"], W)))
            scores[r["method"]][(r["story"], r["day"])] = score_delivery(
                r["ids"], g["articles"], window, event, g["events"])

    print("Delivery (gold story-days):")
    for m, s in sorted(scores.items()):
        rec = [v["recall"] for v in s.values() if v["recall"] is not None]
        hit = [v["event_hit"] for v in s.values()]
        und = sum(v["n_undeliverable"] for v in s.values())
        print(f"  {m:13s} recall@10={statistics.mean(rec):.3f} (n={len(rec)})  "
              f"event_hit={statistics.mean(hit):.3f} (machine-labelled)  undeliverable_gold={und}")

    print("\nPaired 95% CI, contex-sub minus baseline (recall):")
    for base in [m for m in scores if m != "contex-sub"]:
        keys = [k for k in scores.get("contex-sub", {}) if k in scores[base]
                and scores["contex-sub"][k]["recall"] is not None]
        diffs = [scores["contex-sub"][k]["recall"] - scores[base][k]["recall"] for k in keys]
        if diffs:
            lo, hi = boot_ci(diffs)
            print(f"  vs {base:13s} {statistics.mean(diffs):+.3f} [{lo:+.3f}, {hi:+.3f}] n={len(diffs)}")

    for d in map(Path, run_dirs):
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        lags = [r["lag_s"] for r in rows(d / "lag.jsonl")]
        days = rows(d / "days.jsonl")
        if lags or days:
            got = sorted(l for l in lags if l is not None)
            pct = lambda q: got[min(len(got) - 1, int(q * len(got)))] if got else None
            print(f"\n{d.name}: stories={meta.get('n_stories')} sec/day={meta.get('sec_per_day')} "
                  f"server_cpu_s={meta.get('server_cpu_s')} wall_s={meta.get('wall_s')}")
            print(f"  lag p50={pct(.5)} p95={pct(.95)} undelivered_by_read={lags.count(None)}/{len(lags)}")
            print(f"  max behind_s={max((r['behind_s'] for r in days), default=None)}")
        for b in rows(d / "briefing_scores.jsonl"):
            print(f"  briefing {b}")


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 4: Run all offline tests**

Run: `pytest -v`
Expected: all pass, including the existing `analyze`-adjacent tests, since `boot_ci` keeps the same
signature and behavior.

- [ ] **Step 5: Smoke the local baselines and analysis on the Task 4 smoke slice**

```bash
python scripts/run_newsdesk.py --local --start 2017-01-10 --days 3 --stories 5 --run smoke-local
python scripts/analyze_newsdesk.py results/newsdesk/smoke-sub results/newsdesk/smoke-local
```

Expected: recall lines for `contex-sub`, `dense`, `bm25` and `snapshot`, and a lag block for
`smoke-sub`.

- [ ] **Step 6: Commit**

```bash
git add contexeval/retrievers/windowed.py contexeval/scoring/delivery.py contexeval/stats.py \
        scripts/analyze_newsdesk.py scripts/analyze.py tests/test_windowed.py tests/test_delivery.py
git commit -m "feat(newsdesk): windowed dense/BM25/snapshot baselines and delivery scoring"
```

---

### Task 6: Briefings

**Files:**
- Modify: `contexeval/agent.py` (`system` param), `pyproject.toml` (add `rouge-score`, `bert-score`)
- Create: `contexeval/scoring/briefing.py`, `scripts/brief_newsdesk.py`
- Test: `tests/test_briefing.py`; update `tests/test_agent.py` only if it asserts on the exact
  `build_prompt` signature.

**Interfaces:**
- Consumes: delivery rows (Tasks 4 and 5). `MlxAgent.answer(context_text, question, system=...)`.
- Produces:
  - `is_quiet(text: str) -> bool`
  - `rouge(cands: list[str], refs: list[str]) -> list[dict]`
  - `sample_pairs(gold_keys: list[tuple], quiet_keys: list[tuple], n_gold=300, n_quiet=100, seed=13) -> list[tuple]`
  - `brief_newsdesk.py` writes `<run>/briefs.jsonl` and `<run>/briefing_scores.jsonl` (one summary
    row per method).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_briefing.py
from contexeval.agent import build_prompt
from contexeval.scoring.briefing import is_quiet, rouge, sample_pairs

def test_is_quiet_tolerates_case_and_punctuation():
    assert is_quiet("Nothing new.") and is_quiet("  nothing new")
    assert not is_quiet("Nothing new was expected, but the parliament voted.")

def test_rouge_identical_is_one_and_disjoint_is_zero():
    r = rouge(["the cat sat", "dogs bark"], ["the cat sat", "volcano erupts"])
    assert r[0]["rougeL"] == 1.0 and r[1]["rouge1"] == 0.0

def test_sample_pairs_is_seeded_and_sized():
    g = [("s", f"d{i}") for i in range(500)]
    q = [("s", f"q{i}") for i in range(500)]
    a, b = sample_pairs(g, q), sample_pairs(g, q)
    assert a == b and len(a) == 400 and len(set(a) & set(q)) == 100

def test_build_prompt_takes_system():
    assert build_prompt("ctx", "q", system="SYS")[0] == {"role": "system", "content": "SYS"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_briefing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'contexeval.scoring.briefing'`

- [ ] **Step 3: Implement**

In `contexeval/agent.py`, change only these signatures (bodies use `system` in place of `SYSTEM`):

```python
def build_prompt(context_text: str, question: str, system: str = SYSTEM) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"},
    ]
```

`AnswerAgent.answer` and `MlxAgent.answer` each gain `system: str = SYSTEM` and pass it to
`build_prompt`.

`pyproject.toml`: add `"rouge-score", "bert-score"` to `dependencies`, then run
`pip install -e ".[dev]"`.

```python
# contexeval/scoring/briefing.py
import random
import re
from rouge_score import rouge_scorer

_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def is_quiet(text: str) -> bool:
    return re.sub(r"[^a-z ]", "", text.lower()).strip() == "nothing new"


def rouge(cands, refs):
    return [{k: v.fmeasure for k, v in _SCORER.score(r, c).items()} for c, r in zip(cands, refs)]


def bertscore(cands, refs) -> list[float]:
    from bert_score import score
    return score(cands, refs, lang="en", verbose=False)[2].tolist()


def sample_pairs(gold_keys, quiet_keys, n_gold=300, n_quiet=100, seed=13):
    rng = random.Random(seed)
    return (rng.sample(sorted(gold_keys), min(n_gold, len(gold_keys)))
            + rng.sample(sorted(quiet_keys), min(n_quiet, len(quiet_keys))))
```

```python
# scripts/brief_newsdesk.py
"""Briefings on a fixed seeded sample of (story, day) pairs, one per method, scored vs editors.

Usage: CONTEXEVAL_AGENT=mlx python scripts/brief_newsdesk.py results/newsdesk/t1-sub results/newsdesk/t1-query results/newsdesk/t1-local --out results/newsdesk/t1-brief --start 2017-01-01 --days 90
"""
import argparse
import json
import statistics
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from contexeval.agent import MlxAgent
from contexeval.scoring.briefing import bertscore, is_quiet, rouge, sample_pairs
from contexeval.tokens import count_tokens

DATA = Path("data/wcep")
BRIEF_SYSTEM = (
    "You are a news analyst following one story. Using ONLY the articles provided, say what "
    "happened in this story on the given date, in at most 60 words. If the articles contain "
    "nothing new about the story on that date, reply exactly: nothing new"
)


def rows(p):
    return [json.loads(l) for l in open(p)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--days", type=int, required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    articles = {x["id"]: x for x in rows(DATA / "articles.jsonl")}
    stories = {s["story"]: s for s in rows(DATA / "stories.jsonl")}
    days = {(date.fromisoformat(a.start) + timedelta(days=i)).isoformat() for i in range(a.days)}
    gold = {(g["story"], g["day"]): g for g in rows(DATA / "gold.jsonl") if g["day"] in days}

    deliv = defaultdict(dict)  # method -> (story, day) -> ids
    for d in a.runs:
        for r in rows(Path(d) / "deliveries.jsonl"):
            if r["day"] in days:
                deliv[r["method"]][(r["story"], r["day"])] = r["ids"]
    common = set.intersection(*(set(v) for v in deliv.values()))
    pairs = sample_pairs([k for k in common if k in gold], [k for k in common if k not in gold])

    agent = MlxAgent()
    briefs = []
    with open(out / "briefs.jsonl", "w") as f:
        for m in sorted(deliv):
            for story, day in pairs:
                ctx = "\n\n".join(f"[{articles[i]['day']}] {articles[i]['text']}"
                                  for i in deliv[m][(story, day)] if i in articles)
                r = agent.answer(ctx, f"Story: {stories[story]['title']}\nDate: {day}", system=BRIEF_SYSTEM)
                row = {"method": m, "story": story, "day": day, "brief": r.text,
                       "gold": " ".join(gold[(story, day)]["summaries"]) if (story, day) in gold else None,
                       "context_tokens": count_tokens(ctx), "prompt_tokens": r.prompt_tokens,
                       "completion_tokens": r.completion_tokens}
                f.write(json.dumps(row) + "\n"); f.flush()
                briefs.append(row)

    with open(out / "briefing_scores.jsonl", "w") as f:
        for m in sorted(deliv):
            g = [b for b in briefs if b["method"] == m and b["gold"]]
            q = [b for b in briefs if b["method"] == m and not b["gold"]]
            rg = rouge([b["brief"] for b in g], [b["gold"] for b in g])
            bs = bertscore([b["brief"] for b in g], [b["gold"] for b in g])
            row = {"method": m, "n_gold": len(g), "n_quiet": len(q),
                   **{k: statistics.mean(r[k] for r in rg) for k in ("rouge1", "rouge2", "rougeL")},
                   "bertscore_f1": statistics.mean(bs),
                   "quiet_correct": statistics.mean(is_quiet(b["brief"]) for b in q) if q else None,
                   "false_quiet": statistics.mean(is_quiet(b["brief"]) for b in g),
                   "context_tokens": statistics.mean(b["context_tokens"] for b in g + q)}
            f.write(json.dumps(row) + "\n")
            print(row)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all offline tests**

Run: `pytest -v`
Expected: all pass.

- [ ] **Step 5: Smoke on the smoke runs (few pairs)**

```bash
python scripts/brief_newsdesk.py results/newsdesk/smoke-sub results/newsdesk/smoke-local \
  --out results/newsdesk/smoke-brief --start 2017-01-10 --days 3
```

Expected: `briefs.jsonl` holds `methods × sampled pairs` rows (the sample is small on a 3-day,
5-story slice), plus one `briefing_scores.jsonl` row per method.

- [ ] **Step 6: Commit**

```bash
git add contexeval/agent.py contexeval/scoring/briefing.py scripts/brief_newsdesk.py tests/test_briefing.py pyproject.toml
git commit -m "feat(newsdesk): daily briefings per method, scored with ROUGE/BERTScore and quiet-day accuracy"
```

---

### Task 7: Runs and write-up

This is an operational task: no new code. Every command writes under `results/newsdesk/`.

- [ ] **Step 1: T0 (1 week, 20 stories). Measure throughput and pick S.**

```bash
export CONTEX_MCP_URL=http://127.0.0.1:8011/mcp
python scripts/run_newsdesk.py --mode sub --start 2017-01-10 --days 7 --stories 20 --sec-per-day 60 --run t0-sub
```

Set `S` to the smallest of 30, 60 and 120 seconds for which `max(publish_s) < 0.5 × S` across
`t0-sub/days.jsonl`, so ingest finishes well before read time. Then set
`RECONCILE_SWEEP_SECONDS = S/4` in the override and rebuild.

- [ ] **Step 2: T1 (3 months, all stories with gold in the slice)**

Pick the 90-day slice with the most gold pairs:

```bash
python - <<'EOF'
import json
from collections import Counter
from datetime import date, timedelta
c = Counter(json.loads(l)["day"] for l in open("data/wcep/gold.jsonl"))
best = max((sum(c[(date.fromisoformat(d) + timedelta(days=i)).isoformat()] for i in range(90)), d) for d in c)
print(best)
EOF
```

Then, using that `START`:

```bash
python scripts/run_newsdesk.py --mode sub   --start $START --days 90 --sec-per-day $S --run t1-sub
python scripts/run_newsdesk.py --mode query --start $START --days 90 --sec-per-day $S --run t1-query
python scripts/run_newsdesk.py --local      --start $START --days 90 --run t1-local
CONTEXEVAL_AGENT=mlx python scripts/brief_newsdesk.py results/newsdesk/t1-{sub,query,local} --out results/newsdesk/t1-brief --start $START --days 90
python scripts/analyze_newsdesk.py results/newsdesk/t1-{sub,query,local,brief}
```

`server_cpu_s` (t1-sub) − `server_cpu_s` (t1-query) is the cost of subscriptions versus per-day
queries on the same engine, with identical ingest.

- [ ] **Step 3: Load sweep (7 days of the T1 slice)**

```bash
for n in 50 200 412; do for s in $S $((S/4)) $((S/16)); do
  python scripts/run_newsdesk.py --mode sub --start $START --days 7 --stories $n --sec-per-day $s --run sweep-n$n-s$s
done; done
python scripts/analyze_newsdesk.py results/newsdesk/sweep-*
```

Before each `s`, set `RECONCILE_SWEEP_SECONDS` to `s/4` and rebuild. `--stories` caps at however
many stories have gold in the slice; record the actual `n_stories` from `meta.json`. If `S/16`
drops below 2 s, use 2.

- [ ] **Step 4: T2 decision**

Run T2 (`--days 1099`) only if T1's `wall_s` × 12 is under 24 h. Otherwise, record in the write-up
that it didn't fit, with the measured T1 numbers.

- [ ] **Step 5: Write up and commit**

Write `docs/results/2026-10-<dd>-wcep-news-desk.md` in the style of the existing results docs.
Include:
- the setup, with the Contex commit and the `RECONCILE_SWEEP_SECONDS` disclosure
- the delivery table with CIs
- the briefing table
- cost (agent searches and tokens; server CPU for sub versus query)
- the lag and sweep table
- the limitations: entity-in-passing labels, machine-labelled event hits, undeliverable gold
  counts, and drop-out granularity from the sweep interval

Add a row and a link to the README's findings. Commit:

```bash
git add docs/results/2026-10-*-wcep-news-desk.md README.md
git commit -m "docs: WCEP news desk — live subscriptions vs per-day retrieval"
```
