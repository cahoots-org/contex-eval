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
from contexeval.replay import Clock, check_sweep, parse_cputime, run_contex, run_trickle

DATA = Path("data/wcep")
CONTAINERS = ["contexeval-contex-1", "contexeval-postgres-1"]
K, W = 10, 3


def load(name):
    return [json.loads(l) for l in open(DATA / name)]


NATIVE_ENV = Path("scripts/contex-native.env")


def native_env() -> dict[str, str]:
    return dict(l.split("=", 1) for l in NATIVE_ENV.read_text().split("\n") if l and not l.startswith("#"))


def native_pid() -> str:
    """PID of the native Contex listening on the eval port (scripts/contex_native.sh)."""
    port = native_env()["CONTEX_PORT"]
    return subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                          capture_output=True, text=True, check=True).stdout.split()[0]


def cpu_seconds(native: bool) -> dict[str, float]:
    out = {}
    if native:  # host process CPU; MPS GPU time is not included
        t = subprocess.run(["ps", "-o", "time=", "-p", native_pid()], capture_output=True, text=True, check=True)
        out["contex-native"] = parse_cputime(t.stdout)
    for c in CONTAINERS[1:] if native else CONTAINERS:
        stat = subprocess.run(["docker", "exec", c, "cat", "/sys/fs/cgroup/cpu.stat"],
                              capture_output=True, text=True, check=True).stdout
        out[c] = int(next(l.split()[1] for l in stat.splitlines() if l.startswith("usage_usec"))) / 1e6
    return out


def sweep_seconds(native: bool) -> float:
    if native:
        return float(native_env().get("RECONCILE_SWEEP_SECONDS", 300))
    v = subprocess.run(["docker", "exec", CONTAINERS[0], "printenv", "RECONCILE_SWEEP_SECONDS"],
                       capture_output=True, text=True).stdout.strip()
    return float(v or 300)  # Contex's default when unset


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
    p.add_argument("--mode", choices=["sub", "query", "push", "pull"],
                   help="sub/query: one read per day (T1). push/pull: trickled batches (spec §11)")
    p.add_argument("--batches", type=int, default=6, help="push/pull: publish batches per day")
    p.add_argument("--pull-every", type=int, default=1, help="pull: query after every k-th batch")
    p.add_argument("--local", action="store_true")
    p.add_argument("--start", required=True)
    p.add_argument("--days", type=int, required=True)
    p.add_argument("--stories", type=int, default=1000)
    p.add_argument("--sec-per-day", type=float, default=60.0)
    p.add_argument("--run", required=True)
    p.add_argument("--native", action="store_true",
                   help="Contex runs natively via scripts/contex_native.sh (GPU embedding), not in Docker")
    p.add_argument("--no-lag", action="store_true",
                   help="skip lag polling (use for cost runs: polling adds server load)")
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
        sweep = sweep_seconds(args.native)
        check_sweep(sweep, args.sec_per_day)
        meta["reconcile_sweep_seconds"] = sweep
        gold_ids = {(g["story"], g["day"]): set(g["articles"]) for g in gold}
        project = f"newsdesk-{args.run}-{uuid.uuid4().hex[:6]}"
        cpu0, t_start = cpu_seconds(args.native), time.time()
        clock = Clock(t0=time.time() + 5, d0=replay_days[0], sec_per_day=args.sec_per_day)

        async def go():
            async with ContexSession(url=CONTEX_MCP_URL, project_id=project) as s:
                if args.mode in ("push", "pull"):
                    await run_trickle(s, args.mode, clock, replay_days, by_day, stories, gold_ids, K, W, Sink(out),
                                      n_batches=args.batches, pull_every=args.pull_every, score_days=set(days))
                else:
                    await run_contex(s, args.mode, clock, replay_days, by_day, stories, gold_ids, K, W, Sink(out),
                                     watch_lag=not args.no_lag, score_days=set(days))
        asyncio.run(go())
        cpu1 = cpu_seconds(args.native)
        meta |= {"project": project, "wall_s": time.time() - t_start,
                 "server_cpu_s": {c: cpu1[c] - cpu0[c] for c in cpu1}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
