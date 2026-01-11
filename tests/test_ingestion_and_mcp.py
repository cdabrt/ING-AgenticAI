import asyncio
import importlib
import json
from pathlib import Path

from fpdf import FPDF

from AgenticAI.pipeline import ingestion
from AgenticAI.Vectorization.vectorStore.FAISS import FAISSStore


class FakeEmbedder:
    def __init__(self, model_name=None):
        self.model_name = model_name or "fake-embedder"

    def embed_vectors_in_chunks(self, chunks):
        dimension = 3
        payload = []
        for idx, chunk in enumerate(chunks, start=1):
            payload.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk": chunk.model_dump_json(),
                    "embedding": [float(idx), 0.0, 0.0],
                }
            )
        return dimension, payload

    def embed_queries(self, queries):
        return [[1.0, 0.0, 0.0] for _ in queries]


def _write_pdf(path: Path, lines: list[str]) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in lines:
        pdf.cell(0, 10, text=line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def test_ingestion_persists_vector_store(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_pdf(data_dir / "sample.pdf", ["Heading", "Body text for ingestion."])

    vector_dir = tmp_path / "vector_store"
    monkeypatch.setattr(ingestion, "VectorEmbedder", FakeEmbedder)

    result = ingestion.ingest_documents(
        data_dir=str(data_dir),
        persist_dir=str(vector_dir),
        chunk_size=120,
        chunk_overlap=0,
        table_row_overlap=0,
        embedding_batch_size=10,
    )

    assert result["chunks"] > 0
    assert ingestion.vector_store_exists(str(vector_dir))

    store = FAISSStore.load(str(vector_dir), use_cosine_similarity=True)
    results = store.top_k_search([1.0, 0.0, 0.0], top_k=1)
    assert results


def test_mcp_tools_use_persisted_store(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_pdf(data_dir / "sample.pdf", ["Heading", "Body text for MCP test."])

    vector_dir = tmp_path / "vector_store"
    monkeypatch.setattr(ingestion, "VectorEmbedder", FakeEmbedder)
    ingestion.ingest_documents(
        data_dir=str(data_dir),
        persist_dir=str(vector_dir),
        chunk_size=120,
        chunk_overlap=0,
        table_row_overlap=0,
        embedding_batch_size=10,
    )

    monkeypatch.setenv("VECTOR_STORE_DIR", str(vector_dir))

    import AgenticAI.mcp_servers.regulation_server as server

    importlib.reload(server)
    monkeypatch.setattr(server, "VectorEmbedder", FakeEmbedder)
    server._VECTOR_CONTEXT = None

    payload = asyncio.run(server.retrieve_chunks("test query", top_k=1))
    data = json.loads(payload)
    assert data["results"], "Expected retrieval results from FAISS store"

    async def fake_search(*args, **kwargs):
        return [
            {
                "title": "Example",
                "snippet": "Snippet",
                "href": "http://example.com",
                "content": None,
            }
        ]

    async def fake_fetch(*args, **kwargs):
        return "stubbed content"

    monkeypatch.setattr(server, "_run_ddg_search", fake_search)
    monkeypatch.setattr(server, "_fetch_page_text", fake_fetch)

    web_payload = asyncio.run(server.web_search("query", num_results=1, include_content=False))
    web_data = json.loads(web_payload)
    assert web_data["results"][0]["title"] == "Example"

    fetch_payload = asyncio.run(server.fetch_web_page("http://example.com"))
    fetch_data = json.loads(fetch_payload)
    assert fetch_data["content"] == "stubbed content"
