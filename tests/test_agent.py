import pytest
from contexeval.agent import build_prompt, AnswerAgent, AnswerResult

def test_prompt_contains_context_and_question():
    """Offline: verify prompt builder format."""
    prompt = build_prompt("Einstein was a physicist.", "Who was Einstein?")
    assert isinstance(prompt, list)
    assert len(prompt) > 0
    assert all(isinstance(msg, dict) for msg in prompt)
    assert any("Einstein was a physicist." in str(msg) for msg in prompt)
    assert any("Who was Einstein?" in str(msg) for msg in prompt)

@pytest.mark.integration
def test_agent_answers_from_context():
    """Integration: live mlx-lm server required."""
    agent = AnswerAgent()
    agent.warmup()
    result = agent.answer("Scott Joplin was an American composer of ragtime music.", "Who composed ragtime?")
    assert isinstance(result, AnswerResult)
    assert isinstance(result.text, str) and len(result.text) > 0
    assert isinstance(result.prompt_tokens, int) and result.prompt_tokens > 0
    assert isinstance(result.completion_tokens, int) and result.completion_tokens >= 0
