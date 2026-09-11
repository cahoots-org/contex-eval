from dataclasses import dataclass
from openai import OpenAI
from .config import AGENT_BASE_URL, AGENT_MODEL


def build_prompt(context_text: str, question: str) -> list[dict]:
    """Build a chat-format prompt with context and question."""
    return [
        {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context."},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
    ]


@dataclass
class AnswerResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class AnswerAgent:
    def __init__(self):
        self.client = OpenAI(base_url=AGENT_BASE_URL, api_key="fake-key")
        self.model = AGENT_MODEL

    def warmup(self):
        """Pre-warm the model by making a dummy call."""
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi."}],
                max_tokens=1,
            )
        except Exception:
            pass  # Server may not be up; warmup is optional

    def answer(self, context_text: str, question: str) -> AnswerResult:
        """Get an answer from the agent given context and question."""
        prompt = build_prompt(context_text, question)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            max_tokens=100,
        )
        return AnswerResult(
            text=response.choices[0].message.content.strip(),
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
