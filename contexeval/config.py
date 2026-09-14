from pathlib import Path

SEED = 13
HIGH_TOPK = 100
CONTEXT_BUDGET = 28000
DEFAULT_THRESHOLD = 0.5

AGENT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
AGENT_BASE_URL = "http://127.0.0.1:8080/v1"
TOKENIZER_MODEL = "Qwen/Qwen2.5-7B-Instruct"

CONTEX_MCP_URL = "http://127.0.0.1:8001/mcp"
CONTEX_PROJECT_ID = "hotpotqa"
EMBED_MODEL = "all-MiniLM-L6-v2"

DATA_DIR = Path("data")
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUESTIONS_PATH = DATA_DIR / "questions.jsonl"
RESULTS_PATH = DATA_DIR / "results.jsonl"
