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

def test_lag_is_measured_even_when_replay_is_behind_the_clock():
    # Publishing can finish after the day's read time when ingest is slow; the watcher must
    # still poll instead of recording "not delivered" without looking.
    class SlowPublish(FakeSession):
        async def publish_batch(self, items):
            await asyncio.sleep(0.3)  # longer than the 0.2 s day slot
            self.published.append(items)
    sink = _run(SlowPublish(), "sub", gold={("/wiki/A", "2017-01-01"): {"a1"}})
    assert sink.l[0]["lag_s"] is not None
