import uuid
import pytest
from contexeval.contex_client import ContexClient
from contexeval.retrievers.contex import ContexRetriever
from contexeval.config import CONTEX_MCP_URL

pytestmark = pytest.mark.integration

CORPUS = [
    {"para_id": "scott-joplin", "title": "Scott Joplin", "text": "Scott Joplin was an American composer of ragtime music."},
    {"para_id": "mount-fuji", "title": "Mount Fuji", "text": "Mount Fuji is the tallest mountain in Japan."},
]

def test_contex_retriever_returns_result_with_context():
    project = f"cr-{uuid.uuid4().hex[:8]}"
    client = ContexClient(url=CONTEX_MCP_URL, project_id=project)
    client.publish_corpus(CORPUS)
    r = ContexRetriever(CORPUS, client, threshold=0.3).retrieve("Who composed ragtime music?")
    assert "scott-joplin" in r.para_ids
    assert "Scott Joplin" in r.context_text
    assert r.context_tokens > 0
