import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:  # Prefer renamed package but fall back if older dependency is present
    from ddgs import DDGS
except ImportError:  # pragma: no cover - compatibility path
    from duckduckgo_search import DDGS  # type: ignore
from mcp.server.fastmcp import FastMCP

from AgenticAI.Chunker.Chunk import Chunk
from AgenticAI.Vectorization.VectorEmbedder import VectorEmbedder
from AgenticAI.Vectorization.vectorStore.FAISS import FAISSStore

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

mcp = FastMCP("agenticai-regulations")

VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", "artifacts/vector_store"))
CONFIG_PATH = VECTOR_STORE_DIR / "store_config.json"
MAX_WEB_CONTENT_CHARS = 4000
REQUEST_HEADERS = {
    "User-Agent": "AgenticAI-Regulations/0.1 (+https://github.com/cdabrt/ING-AgenticAI)",
}

_VECTOR_CONTEXT: Dict[str, Any] | None = None
_VECTOR_LOCK = asyncio.Lock()


async def _ensure_vector_context():
    global _VECTOR_CONTEXT
    if _VECTOR_CONTEXT is not None:
        return _VECTOR_CONTEXT

    async with _VECTOR_LOCK:
        if _VECTOR_CONTEXT is not None:
            return _VECTOR_CONTEXT

        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"store_config.json not found at {CONFIG_PATH}. Run ingestion before starting the server."
            )

        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        embedder = VectorEmbedder(model_name=config.get("embedding_model_name"))
        store = FAISSStore.load(str(VECTOR_STORE_DIR), use_cosine_similarity=config.get("use_cosine_similarity", True))

        _VECTOR_CONTEXT = {
            "config": config,
            "embedder": embedder,
            "store": store,
        }
        logger.info("Loaded FAISS store with %s chunks", config.get("chunk_count"))
        return _VECTOR_CONTEXT


def _format_chunk(chunk_json: str, score: float) -> Dict[str, Any]:
    chunk = Chunk.model_validate_json(chunk_json)
    return {
        "chunk_id": chunk.chunk_id,
        "score": round(float(score), 4),
        "page": chunk.document.meta_data.page,
        "source": chunk.document.meta_data.source,
        "parent_heading": chunk.parent_heading,
        "text": chunk.document.page_content,
    }


def _clean_text(raw: str, max_chars: int = MAX_WEB_CONTENT_CHARS) -> str:
    collapsed = " ".join(raw.split())
    return collapsed[:max_chars]


async def _fetch_page_text(
    client: httpx.AsyncClient,
    url: str | None,
    max_chars: int = MAX_WEB_CONTENT_CHARS,
) -> str:
    if not url:
        return ""

    try:
        response = await client.get(url, headers=REQUEST_HEADERS, follow_redirects=True)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - network variability
        logger.warning("Failed to fetch %s: %s", url, exc)
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "noscript", "header", "footer"]):
        element.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return _clean_text(text, max_chars)


async def _run_ddg_search(
    query: str,
    max_results: int,
    include_content: bool = False,
) -> List[Dict[str, Optional[str]]]:
    def _search():
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    results = await asyncio.to_thread(_search)
    if not results:
        return []

    contents: List[str | None]
    if include_content:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            contents = await asyncio.gather(
                *[_fetch_page_text(client, item.get("href")) for item in results],
                return_exceptions=False,
            )
    else:
        contents = [None for _ in results]

    formatted: List[Dict[str, Optional[str]]] = []
    for item, content in zip(results, contents):
        formatted.append(
            {
                "title": item.get("title"),
                "snippet": item.get("body"),
                "href": item.get("href"),
                "content": content,
            }
        )
    return formatted


@mcp.tool()
async def retrieve_chunks(query: str, top_k: int = 8) -> str:
    """Return the most similar regulation chunks for the provided query."""

    context = await _ensure_vector_context()
    embedder: VectorEmbedder = context["embedder"]
    store: FAISSStore = context["store"]

    query_embedding = embedder.embed_queries([query])[0]
    raw_results = store.top_k_search(query_embedding, top_k=top_k)

    payload = [_format_chunk(res["chunk"], res["score"]) for res in raw_results]
    return json.dumps({"query": query, "results": payload})


@mcp.tool()
async def web_search(query: str, num_results: int = 5, include_content: bool = False) -> str:
    """Search the open web for additional compliance or regulatory references.

    By default, only metadata is returned so the agent can pre-screen results. Set include_content
    to true to fetch and clean the body text for each candidate in a single call.
    """

    if num_results < 1:
        num_results = 1

    results = await _run_ddg_search(query=query, max_results=num_results, include_content=include_content)
    return json.dumps({"query": query, "results": results})


@mcp.tool()
async def fetch_web_page(url: str, max_chars: int = MAX_WEB_CONTENT_CHARS) -> str:
    """Fetch and sanitize a single web page for inclusion in context."""

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        content = await _fetch_page_text(client, url, max_chars)

    payload = {
        "href": url,
        "content": content,
        "content_length": len(content),
    }
    return json.dumps(payload)


def main():
    logger.info("Starting MCP regulation server with store dir %s", VECTOR_STORE_DIR)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
