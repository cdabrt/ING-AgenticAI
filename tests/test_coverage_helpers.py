import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pytest

from AgenticAI.agentic.decision_logger import DecisionLogger
from AgenticAI.agentic.documents import group_documents_by_source, MAX_DOC_CHARS
from AgenticAI.PDF.Document import Document, Metadata, ElementType
from AgenticAI.agentic import pdf_report
from AgenticAI.mcp.client import MCPToolClient
from AgenticAI.Vectorization.VectorEmbedder import VectorEmbedder
from AgenticAI.Vectorization.vectorStore import store_factory
from AgenticAI.pipeline import ingestion
from AgenticAI.agentic import pipeline_runner
from AgenticAI import main as app_main


class FakeTool:
    def __init__(self, name: str):
        self.name = name


class FakeToolList:
    def __init__(self, tools):
        self.tools = tools


class FakeResult:
    def __init__(self, content):
        self.content = content


class FakeSession:
    def __init__(self):
        self.called = []

    async def call_tool(self, tool_name, arguments):
        self.called.append((tool_name, arguments))
        return FakeResult([FakeTextContent("ok")])

    async def list_tools(self):
        return FakeToolList([FakeTool("retrieve_chunks"), FakeTool("web_search")])


class FakeTextContent:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class FakeJsonContent:
    def __init__(self, payload: dict):
        self.type = "json"
        self.json = payload


class FakeSentenceTransformer:
    def __init__(self, name: str):
        self.name = name
        self.calls = []

    def encode(self, texts, normalize_embeddings=True):
        self.calls.append((list(texts), normalize_embeddings))
        return [np.array([1.0, 0.0, 0.0]) for _ in texts]


def test_decision_logger_writes_jsonl(tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    logger = DecisionLogger(str(log_path))
    asyncio.run(logger.log({"event_type": "unit_test"}))

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event_type"] == "unit_test"
    assert "event_id" in payload
    assert "timestamp" in payload


def test_group_documents_by_source_truncates():
    text = "x" * (MAX_DOC_CHARS + 25)
    docs = [
        Document(page_content="Heading", meta_data=Metadata(source="doc.pdf", page=1, type=ElementType.HEADING)),
        Document(page_content=text, meta_data=Metadata(source="doc.pdf", page=2, type=ElementType.PARAGRAPH)),
    ]
    grouped = group_documents_by_source(docs)
    assert len(grouped) == 1
    summary = grouped[0]
    assert summary.truncated is True
    assert len(summary.text) == MAX_DOC_CHARS
    assert summary.headings == ["Heading"]


def test_render_requirements_pdf_and_sanitize(tmp_path):
    output = tmp_path / "requirements.pdf"
    records = [
        {
            "document": "Doc – 1",
            "document_type": "Directive",
            "business_requirements": [
                {
                    "id": "BR-1",
                    "description": "Use – dash",
                    "rationale": "Because",
                    "document_sources": ["doc.pdf, page 1, chunk c1"],
                    "online_sources": [],
                }
            ],
            "data_requirements": [],
            "assumptions": ["None"],
        }
    ]
    path = pdf_report.render_requirements_pdf(records, output)
    assert path.exists()

    assert pdf_report._sanitize_text("A–B\u00a0C") == "A-B C"

    with pytest.raises(ValueError):
        pdf_report.render_requirements_pdf([], output)


def test_mcp_client_extract_text_and_session():
    content = [
        FakeTextContent("hello"),
        FakeJsonContent({"key": "value"}),
        "raw",
    ]
    payload = MCPToolClient._extract_text(content)
    assert "hello" in payload
    assert "raw" in payload
    assert "\"key\"" in payload

    client = MCPToolClient("server.py")
    with pytest.raises(RuntimeError):
        asyncio.run(client.list_tools())

    client.session = FakeSession()
    tools = asyncio.run(client.list_tools())
    assert tools == ["retrieve_chunks", "web_search"]

    result = asyncio.run(client.call_tool("retrieve_chunks", {"query": "test"}))
    assert result == "ok"


def test_vector_embedder_with_fake_model(monkeypatch):
    monkeypatch.setattr("AgenticAI.Vectorization.VectorEmbedder.SentenceTransformer", FakeSentenceTransformer)
    embedder = VectorEmbedder(model_name="fake")
    chunk = Document(
        page_content="hello",
        meta_data=Metadata(source="doc.pdf", page=1, type=ElementType.PARAGRAPH),
    )
    wrapped = [
        {
            "chunk_id": "c1",
            "document": chunk,
            "char_start": 0,
            "char_end": 5,
            "parent_heading": None,
        }
    ]
    from AgenticAI.Chunker.Chunk import Chunk

    chunks = [Chunk.model_validate(data) for data in wrapped]
    dim, payload = embedder.embed_vectors_in_chunks(chunks)
    assert dim == 3
    assert payload[0]["chunk_id"] == "c1"


def test_store_factory_helpers(monkeypatch, tmp_path):
    assert store_factory.resolve_backend(None) == "faiss"
    assert store_factory.resolve_backend("Milvus") == "milvus"

    monkeypatch.setenv("MILVUS_ENABLE_HYBRID", "true")
    assert store_factory.resolve_enable_hybrid({"enable_hybrid": False}) is True

    config_path = tmp_path / "store_config.json"
    config_path.write_text(json.dumps({"vector_store": "faiss"}))
    config = store_factory.load_store_config(str(tmp_path))
    assert config["vector_store"] == "faiss"


def test_ingestion_helpers(tmp_path):
    with pytest.raises(ValueError):
        list(ingestion._batched_chunks([], 0))

    payload = b"sample"
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(payload)
    digest = ingestion._hash_file(file_path)
    assert isinstance(digest, str) and len(digest) > 0

    persist_dir = tmp_path / "store"
    persist_dir.mkdir()
    (persist_dir / "store_config.json").write_text(json.dumps({"vector_store": "faiss"}))
    assert ingestion.vector_store_exists(str(persist_dir)) is False

    (persist_dir / "index.faiss").write_text("index")
    (persist_dir / "chunks.jsonl").write_text("data")
    assert ingestion.vector_store_exists(str(persist_dir)) is True


def test_pipeline_runner_helpers(monkeypatch, tmp_path):
    index_path = tmp_path / "document_index.json"
    index_path.write_text(json.dumps({"doc.pdf": {"chunk_count": 2}}))
    loaded = pipeline_runner._load_document_index(str(tmp_path))
    assert loaded["doc.pdf"]["chunk_count"] == 2

    index_path.write_text(json.dumps([
        {"source": "a.pdf", "chunk_count": 1},
        {"source": "b.pdf", "chunk_count": 2},
    ]))
    loaded = pipeline_runner._load_document_index(str(tmp_path))
    assert loaded["b.pdf"]["chunk_count"] == 2

    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    assert pipeline_runner._get_llm_provider() == "openrouter"

    assert pipeline_runner._parse_optional_bool("true") is True
    assert pipeline_runner._parse_optional_bool("off") is False
    assert pipeline_runner._parse_optional_bool("maybe") is None

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    pipeline_runner._ensure_llm_key("openrouter")
    assert os.getenv("OPENAI_API_KEY") == "key"

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(EnvironmentError):
        pipeline_runner._ensure_llm_key("gemini")


class FakeChatModelCtor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_build_chat_model(monkeypatch):
    monkeypatch.setattr(pipeline_runner, "ChatOpenAI", FakeChatModelCtor)
    monkeypatch.setattr(pipeline_runner, "ChatGoogleGenerativeAI", FakeChatModelCtor)

    model = pipeline_runner._build_chat_model("openrouter", "gpt", 0.1, reasoning_enabled=True)
    assert isinstance(model, FakeChatModelCtor)
    assert model.kwargs["model"] == "gpt"

    model = pipeline_runner._build_chat_model("gemini", "gemini-pro", 0.2)
    assert model.kwargs["model"] == "gemini-pro"


def test_main_helpers():
    assert app_main._extract_retry_delay("retryDelay: 3s") == 3
    assert app_main._extract_retry_delay("retry in 2.4s") == 2

    assert "Quota" in app_main._summarize_pipeline_error("RESOURCE_EXHAUSTED perday")
    assert "Retry after" in app_main._summarize_pipeline_error("RESOURCE_EXHAUSTED retryDelay: 4s")

    app_main._update_status(state="running", stage="starting", progress=0.0, started_at=app_main._now_iso())
    assert app_main.PIPELINE_STATUS["state"] == "running"

    app_main._update_embedding_status(state="running", stage="embedding", progress=0.3)
    assert app_main.EMBEDDING_STATUS["state"] == "running"
