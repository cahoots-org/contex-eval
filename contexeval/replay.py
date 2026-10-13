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
