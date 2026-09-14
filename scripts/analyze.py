"""Paired bootstrap confidence intervals on Contex-vs-baseline metric gaps.

Reads data/results.jsonl (written by run.py) and, for each metric, computes the
per-question paired difference (Contex - baseline) and a 95% bootstrap CI on the
mean difference, plus the win/tie/loss counts. A CI that excludes 0 means the gap
is unlikely to be sampling noise.

Usage: python scripts/analyze.py
"""
import json
import random
from collections import defaultdict

random.seed(13)  # reproducible bootstrap

RESULTS = "data/results.jsonl"
METHODS = ["contex", "bm25", "dense"]
METRICS = ["recall", "em", "answer_f1"]  # em/answer_f1 only where feasible


def load():
    by = defaultdict(dict)  # method -> qid -> record
    for line in open(RESULTS):
        r = json.loads(line)
        if "error" not in r:
            by[r["method"]][r["qid"]] = r
    return by


def boot_ci(diffs, n=10000):
    if not diffs:
        return (None, None)
    means = []
    for _ in range(n):
        s = [random.choice(diffs) for _ in diffs]
        means.append(sum(s) / len(s))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    by = load()
    qids = sorted(set.intersection(*[set(by[m]) for m in METHODS]))
    print(f"n = {len(qids)} questions (all three methods present)\n")

    # per-method means
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    print("Per-method means:")
    for m in METHODS + (["dump-all"] if "dump-all" in by else []):
        rs = list(by[m].values())
        feas = [r for r in rs if r.get("feasible")]
        rec = mean([r["recall"] for r in rs])
        em = mean([r["em"] for r in feas])
        af = mean([r["answer_f1"] for r in feas])
        ct = mean([r["context_tokens"] for r in rs])
        print(f"  {m:9s} recall={rec if rec is None else round(rec,3)}  "
              f"EM={em if em is None else round(em,3)}  F1={af if af is None else round(af,3)}  "
              f"ctx_tokens={ct if ct is None else round(ct,0)}  (feasible {len(feas)}/{len(rs)})")

    print("\nPaired bootstrap 95% CI on (Contex - baseline), 10k resamples:")
    for metric in METRICS:
        for base in ["bm25", "dense"]:
            diffs = []
            for q in qids:
                cv = by["contex"][q].get(metric)
                bv = by[base][q].get(metric)
                if cv is not None and bv is not None:
                    diffs.append(cv - bv)
            if not diffs:
                continue
            mdiff = sum(diffs) / len(diffs)
            lo, hi = boot_ci(diffs)
            wins = sum(1 for d in diffs if d > 0)
            losses = sum(1 for d in diffs if d < 0)
            ties = len(diffs) - wins - losses
            sig = "" if (lo <= 0 <= hi) else "  *CI excludes 0*"
            print(f"  {metric:9s} contex-{base:5s}: mean={mdiff:+.3f} "
                  f"95%CI=[{lo:+.3f},{hi:+.3f}] W/T/L={wins}/{ties}/{losses} n={len(diffs)}{sig}")


if __name__ == "__main__":
    main()
