from dataclasses import dataclass
from openai import OpenAI
from .config import AGENT_BASE_URL, AGENT_MODEL

SYSTEM = (
    "You are a precise question-answering assistant. "
    "Answer the question using ONLY the provided context. "
    "Reply with the shortest exact span or phrase that answers the question. "
    "For yes/no questions, reply with exactly 'yes' or 'no'. "
    "Do not explain, do not add extra words."
)


def build_prompt(context_text: str, question: str) -> list[dict]:
    """Build a chat-format prompt with context and question."""
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:",
        },
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
                temperature=0,
            )
        except Exception:
            pass  # Server may not be up; warmup is optional

    def answer(self, context_text: str, question: str) -> AnswerResult:
        """Get an answer from the agent given context and question."""
        prompt = build_prompt(context_text, question)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=0,
            max_tokens=256,
        )
        return AnswerResult(
            text=response.choices[0].message.content.strip(),
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )


class MlxAgent:
    """Same contract as AnswerAgent, but runs mlx-lm in-process (no HTTP server).

    mlx_lm.server 0.31.x accepted connections but never answered on this machine; the model
    itself generates fine in-process. Greedy decoding (mlx default) == temperature 0.
    """
    def __init__(self):
        from mlx_lm import load
        self.model, self.tokenizer = load(AGENT_MODEL)

    def warmup(self):
        self.answer("", "Hi.")

    def answer(self, context_text: str, question: str) -> AnswerResult:
        from mlx_lm import generate
        prompt = self.tokenizer.apply_chat_template(
            build_prompt(context_text, question), add_generation_prompt=True, tokenize=False)
        text = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=256)
        return AnswerResult(
            text=text.strip(),
            prompt_tokens=len(self.tokenizer.encode(prompt)),
            completion_tokens=len(self.tokenizer.encode(text)),
        )
