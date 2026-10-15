import numpy as np
from contexeval.retrievers.windowed import local_deliveries

BY_DAY = {
    "2017-01-01": [{"id": "old", "text": "volcano erupts in iceland"}],
    "2017-01-04": [{"id": "v", "text": "volcano lava iceland"}, {"id": "f", "text": "football cup final"},
                   {"id": "e", "text": "election vote count"}],  # 3 docs so BM25 idf for "volcano" is > 0
}
STORIES = [{"story": "/wiki/Volcano", "need": "news about volcano"}]

def fake_embed(texts):
    vocab = ["volcano", "lava", "iceland", "football", "cup", "election", "vote", "news", "about"]
    m = np.array([[t.count(w) for w in vocab] for t in texts], dtype=float) + 1e-9
    return m / np.linalg.norm(m, axis=1, keepdims=True)

def test_window_excludes_old_articles_and_snapshot_freezes():
    rows = list(local_deliveries(BY_DAY, ["2017-01-01", "2017-01-04"], STORIES, k=1, w=3, embed=fake_embed))
    got = {(r["method"], r["day"]): r["ids"] for r in rows}
    assert got[("dense", "2017-01-01")] == ["old"]
    assert got[("dense", "2017-01-04")] == ["v"]          # "old" is outside 01-02..01-04
    assert got[("bm25", "2017-01-04")] == ["v"]
    assert got[("snapshot", "2017-01-04")] == ["old"]     # frozen at the first day
