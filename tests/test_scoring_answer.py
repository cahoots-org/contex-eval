from contexeval.scoring.answer import normalize_answer, answer_em, answer_f1

def test_normalize_strips_articles_punct_case():
    assert normalize_answer("The  Beatles!") == "beatles"

def test_em_exact_after_normalization():
    assert answer_em("the beatles", "Beatles") == 1.0
    assert answer_em("Rolling Stones", "Beatles") == 0.0

def test_f1_partial_overlap():
    # pred "Arthur Conan Doyle" vs gold "Conan Doyle" -> P=2/3, R=1, F1=0.8
    assert abs(answer_f1("Arthur Conan Doyle", "Conan Doyle") - 0.8) < 1e-6

def test_f1_yesno_mismatch_is_zero():
    assert answer_f1("yes", "no") == 0.0
    assert answer_f1("the president", "yes") == 0.0
