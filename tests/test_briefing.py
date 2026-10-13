from contexeval.agent import build_prompt
from contexeval.scoring.briefing import is_quiet, rouge, sample_pairs

def test_is_quiet_tolerates_case_and_punctuation():
    assert is_quiet("Nothing new.") and is_quiet("  nothing new")
    assert not is_quiet("Nothing new was expected, but the parliament voted.")

def test_rouge_identical_is_one_and_disjoint_is_zero():
    r = rouge(["the cat sat", "dogs bark"], ["the cat sat", "volcano erupts"])
    assert r[0]["rougeL"] == 1.0 and r[1]["rouge1"] == 0.0

def test_sample_pairs_is_seeded_and_sized():
    g = [("s", f"d{i}") for i in range(500)]
    q = [("s", f"q{i}") for i in range(500)]
    a, b = sample_pairs(g, q), sample_pairs(g, q)
    assert a == b and len(a) == 400 and len(set(a) & set(q)) == 100

def test_build_prompt_takes_system():
    assert build_prompt("ctx", "q", system="SYS")[0] == {"role": "system", "content": "SYS"}
