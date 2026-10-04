import asyncio
import json
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from .config import CONTEX_MCP_URL, CONTEX_PROJECT_ID


class ContexClient:
    def __init__(self, url: str = CONTEX_MCP_URL, project_id: str = CONTEX_PROJECT_ID):
        self.url = url
        self.project_id = project_id

    async def _run(self, fn):
        # mcp==2.0.0: streamable_http_client yields a 2-tuple (read, write).
        # CONTEX_MCP_URL is /mcp (Contex v0.2.x fixed the old /mcp/mcp double-mount).
        async with streamable_http_client(self.url) as (read, write):
            async with ClientSession(read, write) as s:
                await s.initialize()
                return await fn(s)

    @staticmethod
    def _text(result) -> str:
        # CallToolResult.content is a list of content blocks; the tool returns one text block.
        return result.content[0].text

    def publish_corpus(self, paragraphs: list[dict], batch_size: int = 100) -> None:
        # contex_publish_batch: single contex_publish is rate-limited (60/min) since Contex v1.
        async def _pub(s):
            for i in range(0, len(paragraphs), batch_size):
                r = await s.call_tool("contex_publish_batch", {
                    "project_id": self.project_id,
                    "items": [{
                        "data_key": p["para_id"],
                        "data": {"para_id": p["para_id"], "title": p["title"], "text": p["text"]},
                        "data_format": "json",
                    } for p in paragraphs[i:i + batch_size]],
                })
                if r.is_error:  # tool errors come back as results, not exceptions
                    raise RuntimeError(f"contex_publish_batch failed: {self._text(r)}")
        asyncio.run(self._run(_pub))

    def query(self, question: str, top_k: int, threshold: float) -> list[tuple[str, float]]:
        async def _q(s):
            r = await s.call_tool("contex_query", {
                "project_id": self.project_id, "query": question,
                "top_k": top_k, "threshold": threshold,
            })
            return json.loads(self._text(r))
        payload = asyncio.run(self._run(_q))
        out: list[tuple[str, float]] = []
        seen: set[str] = set()
        for m in payload.get("matches", []):
            pid = m["data_key"].split(".", 1)[0]  # "<para_id>.root" -> "<para_id>"
            if pid not in seen:
                seen.add(pid)
                out.append((pid, float(m["similarity"])))
        return out
