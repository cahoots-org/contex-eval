"""WCEP (Ghalandari et al., ACL 2020) -> news-desk replay inputs.

Stories are entities linked by >= MIN_EVENTS events. Gold for (story, day) is the events on that
day linking the story: their editor-cited (origin WCEP) articles and editor summaries.
Common Crawl articles were attached by the dataset authors' similarity model, so they are replay
volume, never gold.

Usage: python -m contexeval.prep_wcep data/wcep/raw data/wcep
"""
import gzip
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

MIN_EVENTS = 10
TEXT_CHARS = 1000
SPLITS = ("train", "val", "test")


def parse_day(s):
    for fmt in ("%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except (TypeError, ValueError):
            pass
    return None


def story_title(link: str) -> str:
    return unquote(link.removeprefix("/wiki/")).replace("_", " ")


def prep(clusters, lo="2016-08-25", hi="2019-08-28"):
    counts = Counter(link for c in clusters for link in set(c["wiki_links"]))
    story_ids = {link for link, n in counts.items() if n >= MIN_EVENTS}

    articles: dict[str, dict] = {}
    gold: dict[tuple, dict] = defaultdict(lambda: {"articles": [], "summaries": [], "events": []})
    for c in clusters:
        for a in c["articles"]:
            if a["id"] in articles:  # an article can sit in several clusters; first wins
                continue
            day = parse_day(a.get("publish_data"))
            if day is None or not lo <= day <= hi:
                day = c["date"]
            # Body as one paragraph: Contex embeds one node per blank-line paragraph.
            body = " ".join((a.get("text") or "").split())
            text = ((a.get("title") or "") + "\n\n" + body).strip()[:TEXT_CHARS]
            articles[a["id"]] = {"id": a["id"], "day": day, "text": text,
                                 "origin": a["origin"], "event": c["id"]}
        cited = [a["id"] for a in c["articles"] if a["origin"] == "WCEP"]
        for link in sorted(set(c["wiki_links"]) & story_ids):
            g = gold[(link, c["date"])]
            g["articles"] += [i for i in cited if i not in g["articles"]]
            g["summaries"].append(c["summary"])
            g["events"].append(c["id"])

    stories = [{"story": l, "title": story_title(l), "need": f"news about {story_title(l)}",
                "n_events": counts[l]} for l in sorted(story_ids)]
    gold_rows = [{"story": s, "day": d, **g} for (s, d), g in sorted(gold.items())]
    return sorted(articles.values(), key=lambda a: (a["day"], a["id"])), stories, gold_rows


def _read(raw_dir: Path):
    for split in SPLITS:
        with gzip.open(raw_dir / f"{split}.jsonl.gz", "rt") as f:
            for line in f:
                c = json.loads(line)
                # Keep only what prep reads; full article bodies would not fit in memory.
                c["articles"] = [{k: a.get(k) for k in ("id", "origin", "publish_data", "title")}
                                 | {"text": (a.get("text") or "")[:TEXT_CHARS]} for a in c["articles"]]
                yield c


def _write(path: Path, rows):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    articles, stories, gold = prep(list(_read(raw)))
    _write(out / "articles.jsonl", articles)
    _write(out / "stories.jsonl", stories)
    _write(out / "gold.jsonl", gold)
    print(f"{len(articles)} articles, {len(stories)} stories, {len(gold)} gold (story, day) pairs")
