import re
import string
from collections import Counter

def normalize_answer(s: str) -> str:
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t): return " ".join(t.split())
    def remove_punc(t): return "".join(ch for ch in t if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def answer_em(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))

def answer_f1(pred: str, gold: str) -> float:
    npred, ngold = normalize_answer(pred), normalize_answer(gold)
    special = {"yes", "no", "noanswer"}
    if ngold in special and npred != ngold: return 0.0
    if npred in special and npred != ngold: return 0.0
    pt, gt = npred.split(), ngold.split()
    common = Counter(pt) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(gt)
    return 2 * precision * recall / (precision + recall)
