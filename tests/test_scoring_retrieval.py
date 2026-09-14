from contexeval.scoring.retrieval import retrieval_prf

def test_perfect_retrieval():
    assert retrieval_prf(["a", "b"], ["a", "b"]) == (1.0, 1.0, 1.0)

def test_one_gold_among_extras():
    # retrieved 4, one is gold; gold set size 2 -> P=1/4, R=1/2
    p, r, f = retrieval_prf(["a", "x", "y", "z"], ["a", "b"])
    assert abs(p - 0.25) < 1e-9 and abs(r - 0.5) < 1e-9
    assert abs(f - (2 * 0.25 * 0.5 / 0.75)) < 1e-9

def test_empty_retrieval():
    assert retrieval_prf([], ["a", "b"]) == (0.0, 0.0, 0.0)

def test_dedupes_retrieved():
    assert retrieval_prf(["a", "a", "b"], ["a", "b"]) == (1.0, 1.0, 1.0)
