"""Briefings on a fixed seeded sample of (story, day) pairs, one per method, scored vs editors.

Usage: python scripts/brief_newsdesk.py results/newsdesk/t1-sub results/newsdesk/t1-query results/newsdesk/t1-local --out results/newsdesk/t1-brief --start 2017-01-01 --days 90
"""
import argparse
import json
import statistics
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from contexeval.agent import MlxAgent
from contexeval.scoring.briefing import bertscore, is_quiet, rouge, sample_pairs
from contexeval.tokens import count_tokens

DATA = Path("data/wcep")
BRIEF_SYSTEM = (
    "You are a news analyst following one story. Using ONLY the articles provided, say what "
    "happened in this story on the given date, in at most 60 words. If the articles contain "
    "nothing new about the story on that date, reply exactly: nothing new"
)


def rows(p):
    return [json.loads(l) for l in open(p)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--days", type=int, required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    articles = {x["id"]: x for x in rows(DATA / "articles.jsonl")}
    stories = {s["story"]: s for s in rows(DATA / "stories.jsonl")}
    days = {(date.fromisoformat(a.start) + timedelta(days=i)).isoformat() for i in range(a.days)}
    gold = {(g["story"], g["day"]): g for g in rows(DATA / "gold.jsonl") if g["day"] in days}

    deliv = defaultdict(dict)  # method -> (story, day) -> ids
    for d in a.runs:
        for r in rows(Path(d) / "deliveries.jsonl"):
            if r["day"] in days:
                deliv[r["method"]][(r["story"], r["day"])] = r["ids"]
    common = set.intersection(*(set(v) for v in deliv.values()))
    pairs = sample_pairs([k for k in common if k in gold], [k for k in common if k not in gold])

    agent = MlxAgent()
    briefs = []
    with open(out / "briefs.jsonl", "w") as f:
        for m in sorted(deliv):
            for story, day in pairs:
                ctx = "\n\n".join(f"[{articles[i]['day']}] {articles[i]['text']}"
                                  for i in deliv[m][(story, day)] if i in articles)
                r = agent.answer(ctx, f"Story: {stories[story]['title']}\nDate: {day}", system=BRIEF_SYSTEM)
                row = {"method": m, "story": story, "day": day, "brief": r.text,
                       "gold": " ".join(gold[(story, day)]["summaries"]) if (story, day) in gold else None,
                       "context_tokens": count_tokens(ctx), "prompt_tokens": r.prompt_tokens,
                       "completion_tokens": r.completion_tokens}
                f.write(json.dumps(row) + "\n"); f.flush()
                briefs.append(row)

    with open(out / "briefing_scores.jsonl", "w") as f:
        for m in sorted(deliv):
            g = [b for b in briefs if b["method"] == m and b["gold"]]
            q = [b for b in briefs if b["method"] == m and not b["gold"]]
            rg = rouge([b["brief"] for b in g], [b["gold"] for b in g])
            bs = bertscore([b["brief"] for b in g], [b["gold"] for b in g])
            row = {"method": m, "n_gold": len(g), "n_quiet": len(q),
                   **{k: statistics.mean(r[k] for r in rg) for k in ("rouge1", "rouge2", "rougeL")},
                   "bertscore_f1": statistics.mean(bs),
                   "quiet_correct": statistics.mean(is_quiet(b["brief"]) for b in q) if q else None,
                   "false_quiet": statistics.mean(is_quiet(b["brief"]) for b in g),
                   "context_tokens": statistics.mean(b["context_tokens"] for b in g + q)}
            f.write(json.dumps(row) + "\n")
            print(row)


if __name__ == "__main__":
    main()
