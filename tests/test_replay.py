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
