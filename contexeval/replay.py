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
                        # When ingest runs behind, read time may already be past; still give
                        # delivery most of a day slot before calling it undelivered.
                        max(clock.read_time(day), published_t + READ_AT * clock.sec_per_day), out)))
        await sleep_until(clock.read_time(day))
        await asyncio.gather(*watchers)
        r0 = time.time()
        results = await _gather_limited([fetch(st) for st in stories])
        for st, ids in zip(stories, results):
            out.delivery({"method": f"contex-{mode}", "story": st["story"], "day": day, "ids": ids})
        out.day({"day": day, "n_articles": len(arts), "behind_s": behind,
                 "publish_s": published_t - t, "read_s": time.time() - r0})
