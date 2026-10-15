import asyncio
import time
from contextlib import asynccontextmanager

import pytest

from contexeval.replay import Clock, batch_times, run_trickle, split

STORIES = [{"story": "/wiki/A", "need": "news about A"}, {"story": "/wiki/B", "need": "news about B"}]
BY_DAY = {"2017-01-01": [{"id": f"a{i}", "text": "x"} for i in range(8)],
          "2017-01-02": [{"id": f"b{i}", "text": "y"} for i in range(8)]}


class FakePush:
    """Publishing pushes an event for every listened URI; bundles hold the last 10 published ids."""
    def __init__(self, expire_first_read=False, fail_read=False):
        self.published, self.subscribes, self.queries, self.reads = [], 0, 0, 0
        self.queue, self.listened = asyncio.Queue(), []
        self.expire_first_read, self.fail_read, self.expired_once = expire_first_read, fail_read, set()
    async def subscribe(self, need, top_k, max_age_seconds):
        self.subscribes += 1
        return f"sub://{need}"
    async def publish_batch(self, items):
        self.published += [i["data_key"] for i in items]
        for uri in self.listened:
            self.queue.put_nowait(type("Ev", (), {"uri": uri})())
    async def read_bundle(self, uri):
        if self.fail_read:
            raise RuntimeError(f"reading {uri} failed: boom")
        if self.expire_first_read and uri not in self.expired_once:
            self.expired_once.add(uri)
            raise RuntimeError(f"reading {uri} failed: Subscription expired. Call contex_create_subscription")
        self.reads += 1
        return self.published[-10:][::-1]
    async def query(self, need, top_k, max_age_seconds):
        self.queries += 1
        return self.published[-10:][::-1]
    @asynccontextmanager
    async def listen(self, uris):
        self.listened = list(uris)
        async def events():
            while True:
                yield await self.queue.get()
        yield events()


class Sink:
    def __init__(self): self.d, self.l, self.days = [], [], []
    def delivery(self, r): self.d.append(r)
    def lag(self, r): self.l.append(r)
    def day(self, r): self.days.append(r)


def _run(session, mode, pull_every=1, gold=None, n=4):
    clock = Clock(t0=time.time(), d0="2017-01-01", sec_per_day=0.4)
    sink = Sink()
    asyncio.run(run_trickle(session, mode, clock, sorted(BY_DAY), BY_DAY, STORIES, gold or {},
                            k=10, w=3, out=sink, n_batches=n, pull_every=pull_every))
    return sink


def test_split_and_batch_times_spread_a_day_over_80_percent_of_its_slot():
    assert split(list(range(10)), 4) == [[0, 1], [2, 3, 4], [5, 6], [7, 8, 9]]
    assert split([], 3) == [[], [], []]
    c = Clock(t0=0.0, d0="2017-01-01", sec_per_day=100.0)
    assert batch_times(c, "2017-01-02", 4) == [100.0, 120.0, 140.0, 160.0]


def test_push_delivers_without_queries_and_measures_lag():
    s = FakePush()
    sink = _run(s, "push", gold={("/wiki/A", "2017-01-01"): {"a0"}, ("/wiki/A", "2017-01-02"): {"b7"}})
    assert s.queries == 0 and s.reads > 0
    assert {(r["method"], r["story"], r["day"]) for r in sink.d} == {
        ("contex-push", st["story"], d) for st in STORIES for d in BY_DAY}
    day2 = next(r for r in sink.d if r["story"] == "/wiki/A" and r["day"] == "2017-01-02")
    assert day2["ids"][0] == "b7"                                   # held bundle is current
    lags = {r["article"]: r["lag_s"] for r in sink.l}
    assert lags["b7"] is not None and lags["b7"] < 0.2
    assert sum(r["reads"] for r in sink.days) == s.reads


def test_pull_every_k_batches_queries_each_story_that_often():
    s = FakePush()
    sink = _run(s, "pull", pull_every=2)
    assert s.queries == len(STORIES) * 2 * len(BY_DAY) and s.reads == 0 and s.subscribes == 0
    assert {r["method"] for r in sink.d} == {"contex-pull-k2"}


def test_push_resubscribes_when_a_read_finds_the_subscription_expired():
    s = FakePush(expire_first_read=True)
    sink = _run(s, "push")
    assert s.subscribes == 2 * len(STORIES)                        # initial + one re-create each
    assert sum(r["resubscribed"] for r in sink.days) == len(STORIES)


def test_push_read_errors_fail_the_run_with_the_story_named():
    with pytest.raises(RuntimeError, match="push read failed for story /wiki/"):
        _run(FakePush(fail_read=True), "push")
