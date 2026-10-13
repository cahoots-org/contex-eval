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
