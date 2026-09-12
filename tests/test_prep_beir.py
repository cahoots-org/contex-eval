from contexeval.prep_beir import pid, build


def test_pid_is_dot_free_and_stable():
    assert pid("MED-10") == "med-10"
    assert pid(31715818) == "31715818"
    assert "." not in pid("doc.v2")


CORPUS = [
    {"_id": "MED-1", "title": "A", "text": "statin cholesterol study"},
    {"_id": "MED-2", "title": "B", "text": "unrelated topic"},
    {"_id": "MED-3", "title": "C", "text": "another distractor"},
]
QUERIES = [
    {"_id": "Q1", "title": "", "text": "do statins help?"},
    {"_id": "Q2", "title": "", "text": "no labels here"},  # not in qrels -> dropped
]
QRELS = [
    {"query-id": "Q1", "corpus-id": "MED-1", "score": 1},
    {"query-id": "Q1", "corpus-id": "MED-2", "score": 0},  # score 0 -> not gold
]


def test_build_full_corpus_and_gold_mapping():
    corpus, questions = build(CORPUS, QUERIES, QRELS)
    assert {c["para_id"] for c in corpus} == {"med-1", "med-2", "med-3"}  # full corpus
    assert len(questions) == 1  # Q2 has no gold -> dropped
    assert questions[0]["qid"] == "Q1"
    assert questions[0]["gold_para_ids"] == ["med-1"]  # only score>0
    assert questions[0]["answer"] == ""


def test_build_pooled_corpus_keeps_gold_plus_distractors():
    corpus, questions = build(CORPUS, QUERIES, QRELS, max_corpus=2)
    ids = {c["para_id"] for c in corpus}
    assert "med-1" in ids  # gold always kept
    assert len(corpus) == 2  # gold + 1 distractor
    assert questions[0]["gold_para_ids"] == ["med-1"]
