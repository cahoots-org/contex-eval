from functools import lru_cache
from transformers import AutoTokenizer
from .config import TOKENIZER_MODEL

@lru_cache(maxsize=1)
def _tok():
    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_tok().encode(text))
