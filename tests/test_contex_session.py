import asyncio
import uuid
import pytest
from contexeval.contex_client import ContexSession, doc_ids

def test_doc_ids_prefers_document_and_dedups_in_order():
    ms = [{"data_key": "a/p1", "document": "a"}, {"data_key": "b"}, {"data_key": "a/p0", "document": "a"}]
    assert doc_ids(ms) == ["a", "b"]

@pytest.mark.integration
def test_publish_subscribe_read_and_query_roundtrip():
    async def go():
        async with ContexSession(project_id=f"nd-{uuid.uuid4().hex[:8]}") as c:
            uri = await c.subscribe("news about volcanoes", top_k=10, max_age_seconds=3600)
            assert await c.read_bundle(uri) == []
            await c.publish_batch([
                {"data_key": "v1", "data": "Volcano erupts\n\nLava flows in Iceland.", "data_format": "text", "published_at": None},
                {"data_key": "f1", "data": "Football final\n\nThe cup was won.", "data_format": "text", "published_at": None},
            ])
            for _ in range(40):  # reconcile runs in the background since Contex #457
                if "v1" in await c.read_bundle(uri):
                    break
                await asyncio.sleep(0.25)
            assert (await c.read_bundle(uri))[0] == "v1"
            assert (await c.query("news about volcanoes", top_k=10, max_age_seconds=3600))[0] == "v1"
            with pytest.raises(RuntimeError, match="contex://subscriptions/nope"):
                await c.read_bundle("contex://subscriptions/nope")
    asyncio.run(go())
