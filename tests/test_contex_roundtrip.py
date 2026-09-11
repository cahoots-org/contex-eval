import uuid
import pytest
from contexeval.contex_client import ContexClient
from contexeval.config import CONTEX_MCP_URL

pytestmark = pytest.mark.integration

PARAS = [
    {"para_id": "scott-joplin", "title": "Scott Joplin", "text": "Scott Joplin was an American composer of ragtime music."},
    {"para_id": "the-entertainer", "title": "The Entertainer", "text": "The Entertainer is a 1902 piano rag by Scott Joplin."},
    {"para_id": "mount-fuji", "title": "Mount Fuji", "text": "Mount Fuji is the tallest mountain in Japan."},
    {"para_id": "python-language", "title": "Python (language)", "text": "Python is a high-level programming language."},
    {"para_id": "great-barrier-reef", "title": "Great Barrier Reef", "text": "The Great Barrier Reef is the world's largest coral reef system."},
]

def test_publish_and_query_maps_back_to_para_ids():
    project = f"rt-{uuid.uuid4().hex[:8]}"
    client = ContexClient(url=CONTEX_MCP_URL, project_id=project)
    client.publish_corpus(PARAS)
    hits = client.query("Who composed ragtime piano music?", top_k=100, threshold=0.3)
    ids = {pid for pid, _ in hits}
    assert ids <= {p["para_id"] for p in PARAS}          # only known ids, cleanly mapped
    assert "scott-joplin" in ids                          # relevant gold surfaced
    assert all(0.0 <= sim <= 1.0 for _, sim in hits)
