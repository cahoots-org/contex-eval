import random
import re
from rouge_score import rouge_scorer

_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def is_quiet(text: str) -> bool:
    return re.sub(r"[^a-z ]", "", text.lower()).strip() == "nothing new"


def rouge(cands, refs):
    return [{k: v.fmeasure for k, v in _SCORER.score(r, c).items()} for c, r in zip(cands, refs)]


def bertscore(cands, refs) -> list[float]:
    from bert_score import score
    return score(cands, refs, lang="en", verbose=False)[2].tolist()


def sample_pairs(gold_keys, quiet_keys, n_gold=300, n_quiet=100, seed=13):
    rng = random.Random(seed)
    return (rng.sample(sorted(gold_keys), min(n_gold, len(gold_keys)))
            + rng.sample(sorted(quiet_keys), min(n_quiet, len(quiet_keys))))
