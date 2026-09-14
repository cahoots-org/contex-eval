from contexeval import config

def test_core_constants_present():
    assert config.SEED == 13
    assert config.HIGH_TOPK == 100
    assert config.CONTEXT_BUDGET == 28000
    assert config.DEFAULT_THRESHOLD == 0.5
    assert config.AGENT_BASE_URL.endswith("/v1")
    assert config.CONTEX_MCP_URL.endswith("/mcp")
    assert config.EMBED_MODEL == "all-MiniLM-L6-v2"
