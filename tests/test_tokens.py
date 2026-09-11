from contexeval.tokens import count_tokens

def test_counts_are_positive_and_monotonic():
    a = count_tokens("hello world")
    b = count_tokens("hello world hello world hello world")
    assert a > 0 and b > a

def test_empty_string_is_zero():
    assert count_tokens("") == 0
