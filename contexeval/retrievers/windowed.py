"""Offline per-day baselines over the same W-day window Contex sees. No wall clock needed."""
import os
import numpy as np
from rank_bm25 import BM25Okapi
from ..config import EMBED_MODEL
from ..replay import window_days
from .bm25 import _tok


def _gte():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device=os.getenv("CONTEXEVAL_DEVICE") or None)
    return lambda texts: model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                      batch_size=64, show_progress_bar=len(texts) > 1000)


def local_deliveries(by_day, days, stories, k, w, embed=None):
    embed = embed or _gte()
    arts = [a for d in sorted(by_day) for a in by_day[d]]
    idx = {a["id"]: i for i, a in enumerate(arts)}
    emb = embed([a["text"] for a in arts])
    q = embed([s["need"] for s in stories])
    snapshot = {}
    for day in days:
        ids = [a["id"] for d in window_days(day, w) for a in by_day.get(d, [])]
        if not ids:
            for s in stories:
                for m in ("dense", "bm25"):
                    yield {"method": m, "story": s["story"], "day": day, "ids": []}
                yield {"method": "snapshot", "story": s["story"], "day": day, "ids": snapshot.get(s["story"], [])}
            continue
        rows = np.array([idx[i] for i in ids])
        dense_scores = emb[rows] @ q.T                      # (n_window, n_stories)
        bm25 = BM25Okapi([_tok(arts[r]["text"]) for r in rows])
        for j, s in enumerate(stories):
            dense = [ids[i] for i in np.argsort(-dense_scores[:, j], kind="stable")[:k]]
            sc = bm25.get_scores(_tok(s["need"]))
            bm = [ids[i] for i in np.argsort(-sc, kind="stable")[:k]]
            snapshot.setdefault(s["story"], dense)
            yield {"method": "dense", "story": s["story"], "day": day, "ids": dense}
            yield {"method": "bm25", "story": s["story"], "day": day, "ids": bm}
            yield {"method": "snapshot", "story": s["story"], "day": day, "ids": snapshot[s["story"]]}
