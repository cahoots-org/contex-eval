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


def check_sweep(sweep_s: float, sec_per_day: float) -> None:
    """Contex drops out-of-window items from a bundle only on a re-match: a publish that affects it,
    or the periodic sweep. A sweep slower than a tenth of a day lets stale items linger and widens
    contex-sub's effective window."""
    if sweep_s > sec_per_day / 10:
        raise ValueError(f"RECONCILE_SWEEP_SECONDS={sweep_s:g} > sec_per_day/10 ({sec_per_day / 10:g}); "
                         "lower it in contex/docker-compose.override.yml")


def parse_cputime(s: str) -> float:
    """`ps -o time=` value ([dd-][hh:]mm:ss.ss) -> seconds."""
    days, _, rest = s.strip().rpartition("-")
    secs = 0.0
    for part in rest.split(":"):
        secs = secs * 60 + float(part)
    return secs + int(days or 0) * 86400


async def sleep_until(t: float) -> float:
    delay = t - time.time()
    if delay > 0:
        await asyncio.sleep(delay)
        return 0.0
    return -delay


READ_CONCURRENCY = 16
LAG_CONCURRENCY = 4  # lag polling is extra server load; keep it small
LAG_POLL_S = 0.25


async def _gather_limited(coros, n=READ_CONCURRENCY):
    sem = asyncio.Semaphore(n)
    async def one(c):
        async with sem:
            return await c
    return await asyncio.gather(*(one(c) for c in coros))


async def _watch_lag(session, sem, uri, story, day, published, deadline, out):
    """Poll one subscription until each gold article shows up, or deadline.

    ``published`` maps article id -> wall time its publish batch returned."""
    pending = set(published)
    while pending and time.time() < deadline:
        async with sem:
            seen = set(await session.read_bundle(uri)) & pending
        for a in sorted(seen):
            out.lag({"story": story, "day": day, "article": a, "lag_s": time.time() - published[a]})
        pending -= seen
        await asyncio.sleep(LAG_POLL_S)
    for a in sorted(pending):
        out.lag({"story": story, "day": day, "article": a, "lag_s": None})


async def run_contex(session, mode, clock, days, by_day, stories, gold_ids, k, w, out, watch_lag=True, score_days=None):
    max_age = clock.max_age_seconds(w)
    lag_sem = asyncio.Semaphore(LAG_CONCURRENCY)
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

    watchers = []
    for day in days:
        behind = await sleep_until(clock.slot_start(day))
        arts = by_day.get(day, [])
        t = time.time()
        published = {}  # article id -> when its batch returned
        for batch in chunks(arts):
            await session.publish_batch([{"data_key": a["id"], "data": a["text"], "data_format": "text",
                                          "published_at": clock.published_at(day)} for a in batch])
            published |= dict.fromkeys((a["id"] for a in batch), time.time())
        publish_s = time.time() - t
        if mode == "sub" and watch_lag:
            for st in stories:
                g = gold_ids.get((st["story"], day), set()) & published.keys()
                if g:
                    # Runs alongside later days; the delivery read below never waits for it.
                    watchers.append(asyncio.create_task(_watch_lag(
                        session, lag_sem, uris[st["story"]], st["story"], day,
                        {a: published[a] for a in g}, time.time() + READ_AT * clock.sec_per_day, out)))
        await sleep_until(clock.read_time(day))
        read_offset = time.time() - clock.slot_start(day)
        if read_offset > max_age:
            raise RuntimeError(f"{day}: read {read_offset:.0f}s after slot start, behind past the "
                               f"{max_age}s window; today's articles are already out of it. Raise --sec-per-day.")
        r0 = time.time()
        # Pre-roll days are published to fill the window, not read or scored.
        scored = score_days is None or day in score_days
        results = await _gather_limited([fetch(st) for st in stories]) if scored else []
        for st, ids in zip(stories, results):
            out.delivery({"method": f"contex-{mode}", "story": st["story"], "day": day, "ids": ids})
        out.day({"day": day, "n_articles": len(arts), "behind_s": behind, "publish_s": publish_s,
                 "read_offset_s": read_offset, "read_s": time.time() - r0})
    await asyncio.gather(*watchers)


# --- Trickle replay: push vs pull at equal freshness (spec §11) ---------------------------

TRICKLE_SPAN = 0.8  # batches spread over the first 80% of a day slot; reads at READ_AT


def split(xs: list, n: int) -> list[list]:
    """n contiguous, near-equal parts (some empty when len(xs) < n)."""
    return [xs[i * len(xs) // n:(i + 1) * len(xs) // n] for i in range(n)]


def batch_times(clock: Clock, day: str, n: int) -> list[float]:
    step = TRICKLE_SPAN * clock.sec_per_day / n
    return [clock.slot_start(day) + i * step for i in range(n)]


async def run_trickle(session, mode, clock, days, by_day, stories, gold_ids, k, w, out,
                      n_batches=6, pull_every=1, score_days=None):
    """Publish each day in n_batches; agents hold whatever push (mode "push") or a query every
    pull_every batches (mode "pull") last gave them. Records the held bundle at READ_AT."""
    max_age = clock.max_age_seconds(w)
    method = "contex-push" if mode == "push" else f"contex-pull-k{pull_every}"
    held = {st["story"]: [] for st in stories}
    pending: dict[str, dict[str, tuple[str, float]]] = {st["story"]: {} for st in stories}
    counts = {"reads": 0, "queries": 0, "resubscribed": 0}
    by_uri, handlers = {}, set()
    read_sem = asyncio.Semaphore(READ_CONCURRENCY)

    def resolve(story, ids, t):
        for a in [a for a in pending[story] if a in ids]:
            day, published = pending[story].pop(a)
            out.lag({"story": story, "day": day, "article": a, "lag_s": t - published})

    async def on_push(uri):
        st = by_uri[uri]
        try:
            try:
                async with read_sem:
                    ids = await session.read_bundle(uri)
            except RuntimeError as e:
                if "expired" not in str(e).lower():
                    raise
                # Listening does not renew the lease; re-creating with the same arguments resumes
                # the same URI.
                await session.subscribe(st["need"], top_k=k, max_age_seconds=max_age)
                counts["resubscribed"] += 1
                async with read_sem:
                    ids = await session.read_bundle(uri)
        except Exception as e:
            raise RuntimeError(f"push read failed for story {st['story']}: {e}") from e
        counts["reads"] += 1
        held[st["story"]] = ids
        resolve(st["story"], set(ids), time.time())

    async def pull_round():
        async def one(st):
            try:
                return await session.query(st["need"], top_k=k, max_age_seconds=max_age)
            except Exception as e:
                raise RuntimeError(f"pull query failed for story {st['story']}: {e}") from e
        results = await _gather_limited([one(st) for st in stories])
        t = time.time()
        counts["queries"] += len(stories)
        for st, ids in zip(stories, results):
            held[st["story"]] = ids
            resolve(st["story"], set(ids), t)

    def check_handlers():
        for h in [h for h in handlers if h.done()]:
            handlers.discard(h)
            h.result()  # re-raises a failed push read

    async def replay():
        for day in days:
            gold_today = {}
            for st in stories:
                for a in gold_ids.get((st["story"], day), ()):
                    gold_today.setdefault(a, []).append(st["story"])
            t_day, behind_max = time.time(), 0.0
            for i, (t_batch, part) in enumerate(zip(batch_times(clock, day, n_batches),
                                                    split(by_day.get(day, []), n_batches))):
                behind = await sleep_until(t_batch)
                behind_max = max(behind_max, behind)
                if behind > clock.sec_per_day:
                    raise RuntimeError(f"{day} batch {i}: {behind:.0f}s behind the clock; raise --sec-per-day")
                for c in chunks(part):
                    await session.publish_batch([{"data_key": a["id"], "data": a["text"], "data_format": "text",
                                                  "published_at": clock.published_at(day)} for a in c])
                    tp = time.time()
                    for a in c:
                        for story in gold_today.get(a["id"], ()):
                            pending[story][a["id"]] = (day, tp)
                if mode == "pull" and (i + 1) % pull_every == 0:
                    await pull_round()
                check_handlers()
            await sleep_until(clock.read_time(day))
            read_offset = time.time() - clock.slot_start(day)
            if read_offset > max_age:
                raise RuntimeError(f"{day}: read {read_offset:.0f}s after slot start, behind past the "
                                   f"{max_age}s window. Raise --sec-per-day.")
            check_handlers()
            if score_days is None or day in score_days:
                for st in stories:
                    out.delivery({"method": method, "story": st["story"], "day": day, "ids": list(held[st["story"]])})
            out.day({"day": day, "n_articles": len(by_day.get(day, [])), "behind_s": behind_max,
                     "read_offset_s": read_offset, **counts})
            counts.update(reads=0, queries=0, resubscribed=0)

    if mode == "push":
        for st in stories:
            uri = await session.subscribe(st["need"], top_k=k, max_age_seconds=max_age)
            by_uri[uri] = st
        async with session.listen(list(by_uri)) as events:
            async def listener():
                async for ev in events:
                    if ev.uri in by_uri:
                        handlers.add(asyncio.create_task(on_push(ev.uri)))
            listening = asyncio.create_task(listener())
            try:
                await replay()
            finally:
                listening.cancel()
                await asyncio.gather(listening, *handlers, return_exceptions=True)
    else:
        await replay()
    for story, left in pending.items():
        for a, (day, _) in left.items():
            out.lag({"story": story, "day": day, "article": a, "lag_s": None})
