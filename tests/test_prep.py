from contexeval.prep import slug, pool_examples

def _ex(qid, q, ans, titles, sentences, sup_titles):
    return {
        "id": qid, "question": q, "answer": ans,
        "supporting_facts": {"title": sup_titles, "sent_id": [0] * len(sup_titles)},
        "context": {"title": titles, "sentences": sentences},
    }

def test_slug_is_dot_free_and_normalized():
    assert slug("The Beatles!") == "the-beatles"
    assert "." not in slug("Dr. No (film)")

def test_pool_dedups_shared_titles_and_maps_gold():
    ex1 = _ex("q1", "Q1?", "A1",
              titles=["Alpha", "Beta"], sentences=[["a1.", "a2."], ["b1."]],
              sup_titles=["Alpha"])
    ex2 = _ex("q2", "Q2?", "A2",
              titles=["Beta", "Gamma"], sentences=[["b1."], ["g1."]],
              sup_titles=["Gamma"])
    corpus, questions = pool_examples([ex1, ex2])
    ids = {c["para_id"] for c in corpus}
    assert ids == {slug("Alpha"), slug("Beta"), slug("Gamma")}  # Beta deduped
    assert questions[0]["gold_para_ids"] == [slug("Alpha")]
    assert questions[1]["gold_para_ids"] == [slug("Gamma")]
    alpha = next(c for c in corpus if c["para_id"] == slug("Alpha"))
    assert alpha["text"] == "a1. a2."  # sentences joined
