from contexeval.scoring.delivery import score_delivery
from contexeval.stats import boot_ci

EVENT = {"g1": 7, "g2": 7, "early": 7, "auto": 7, "x": 9}

def test_recall_counts_only_deliverable_gold():
    s = score_delivery(["g1", "x"], gold_articles=["g1", "g2", "early"],
                       window={"g1", "g2", "x", "auto"}, article_event=EVENT, gold_events=[7])
    assert s["recall"] == 0.5 and s["n_deliverable"] == 2 and s["n_undeliverable"] == 1
    assert s["event_hit"] == 1.0

def test_no_deliverable_gold_gives_none_recall_but_event_hit_still_scored():
    s = score_delivery(["auto"], gold_articles=["early"], window={"auto"},
                       article_event=EVENT, gold_events=[7])
    assert s["recall"] is None and s["event_hit"] == 1.0

def test_boot_ci_brackets_mean():
    lo, hi = boot_ci([0.1, 0.2, 0.3, 0.2, 0.1] * 10)
    assert lo < 0.18 < hi
