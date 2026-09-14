from dataclasses import dataclass
from ..tokens import count_tokens

@dataclass
class RetrievalResult:
    para_ids: list[str]
    context_text: str
    context_tokens: int
    @property
    def bundle_size(self) -> int:
        return len(self.para_ids)

def corpus_index(corpus: list[dict]) -> dict[str, dict]:
    return {c["para_id"]: c for c in corpus}

def assemble_context(pids: list[str], by_id: dict[str, dict]) -> str:
    return "\n\n".join(f"Title: {by_id[p]['title']}\n{by_id[p]['text']}" for p in pids)

def make_result(pids: list[str], by_id: dict[str, dict]) -> RetrievalResult:
    ctx = assemble_context(pids, by_id)
    return RetrievalResult(para_ids=pids, context_text=ctx, context_tokens=count_tokens(ctx))
