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
        print(f"  {m:13s} recall@10={statistics.mean(rec) if rec else float('nan'):.3f} (n={len(rec)})  "
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
