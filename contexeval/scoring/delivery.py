def score_delivery(ids, gold_articles, window, article_event, gold_events):
    """Recall over gold articles dated inside the window (deliverable); event hit over any origin."""
    deliverable = [a for a in gold_articles if a in window]
    got = set(ids)
    recall = (sum(a in got for a in deliverable) / len(deliverable)) if deliverable else None
    hit = float(any(article_event.get(i) in set(gold_events) for i in ids))
    return {"recall": recall, "n_deliverable": len(deliverable),
            "n_undeliverable": len(gold_articles) - len(deliverable), "event_hit": hit}
