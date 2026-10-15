import asyncio
import json
from contextlib import AsyncExitStack
from mcp import Client, ClientSession
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


def doc_ids(matches: list[dict]) -> list[str]:
    out: list[str] = []
    for m in matches:
        d = m.get("document") or m["data_key"]
        if d not in out:
            out.append(d)
    return out


class ContexSession:
    """One MCP session held for a whole replay. ContexClient opens a session per call, which is
    fine for one-shot evals but too slow for hundreds of reads per simulated day."""

    def __init__(self, url: str = CONTEX_MCP_URL, project_id: str = CONTEX_PROJECT_ID):
        self.url = url
        self.project_id = project_id

    async def __aenter__(self):
        # mcp.Client negotiates the 2026-07-28 protocol, which subscriptions/listen requires.
        # cache=None: bundle reads must never be served from a client-side cache.
        self._stack = AsyncExitStack()
        self.s = await self._stack.enter_async_context(Client(self.url, cache=None))
        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()

    async def _call(self, tool: str, args: dict) -> str:
        r = await self.s.call_tool(tool, {"project_id": self.project_id, **args})
        if r.is_error:
            raise RuntimeError(f"{tool} failed: {r.content[0].text}")
        return r.content[0].text

    async def publish_batch(self, items: list[dict]) -> None:
        await self._call("contex_publish_batch", {"items": items})

    async def subscribe(self, need: str, top_k: int, max_age_seconds: int) -> str:
        return json.loads(await self._call("contex_create_subscription", {
            "needs": [need], "top_k": top_k, "threshold": 0.0, "max_age_seconds": max_age_seconds,
        }))["resource_uri"]

    async def read_bundle(self, uri: str) -> list[str]:
        try:
            r = await self.s.read_resource(uri)
            bundle = json.loads(r.contents[0].text)
        except Exception as e:
            raise RuntimeError(f"reading {uri} failed: {e}") from e
        return doc_ids([m for ms in bundle.values() for m in ms])

    async def query(self, need: str, top_k: int, max_age_seconds: int) -> list[str]:
        return doc_ids(json.loads(await self._call("contex_query", {
            "query": need, "top_k": top_k, "threshold": 0.0, "max_age_seconds": max_age_seconds,
        }))["matches"])

    def listen(self, uris: list[str]):
        """Async context manager over a subscriptions/listen stream; iterate it for events
        carrying ``.uri``. Contex coalesces repeat events for one URI."""
        return self.s.listen(resource_subscriptions=list(uris))
