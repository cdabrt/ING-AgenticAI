import asyncio
from types import SimpleNamespace
from pathlib import Path

import pytest

from AgenticAI.agentic import pipeline_runner
from AgenticAI.agentic.models import RequirementBundle, RequirementItem
from AgenticAI.PDF.Document import Document, Metadata, ElementType


class FakeRunner:
    def __init__(self, *args, **kwargs):
        self.bundle = RequirementBundle(
            document="doc.pdf",
            document_type="Directive",
            business_requirements=[
                RequirementItem(
                    id="BR-1",
                    description="Do thing",
                    rationale="Why",
                    document_sources=["doc.pdf, page 1, chunk c1"],
                    online_sources=[],
                )
            ],
            data_requirements=[],
            assumptions=[],
        )

    async def process_document(self, state, status_callback=None):
        return {**state, "retrieval_results": [], "document_summary": "summary", "document_type": "Directive"}

    async def distill_evidence(self, state, status_callback=None):
        return {"evidence_cards": []}

    async def generate_requirements(self, state, status_callback=None):
        return {"requirements": self.bundle.model_dump()}

    async def merge_requirements(self, bundles, doc_sources, status_callback=None):
        return self.bundle


class FakeMCP:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def list_tools(self):
        return ["retrieve_chunks", "web_search"]


async def _fake_load_structured_pdf(path):
    return [
        Document(
            page_content="Heading",
            meta_data=Metadata(source=path.name, page=1, type=ElementType.HEADING),
        ),
        Document(
            page_content="Body",
            meta_data=Metadata(source=path.name, page=1, type=ElementType.PARAGRAPH),
        ),
    ]


def test_run_pipeline_success(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "doc.pdf").write_bytes(b"%PDF-1.4")

    output_path = tmp_path / "out.json"
    pdf_output = tmp_path / "out.pdf"

    monkeypatch.setattr(pipeline_runner, "load_dotenv", lambda: None)
    monkeypatch.setattr(pipeline_runner, "_ensure_llm_key", lambda *_: None)
    monkeypatch.setattr(pipeline_runner, "_log_runtime_context", lambda *_: None)
    monkeypatch.setattr(pipeline_runner, "resolve_backend", lambda *_: "faiss")
    monkeypatch.setattr(pipeline_runner, "vector_store_exists", lambda *_: True)
    monkeypatch.setattr(pipeline_runner, "PDFParser", type("P", (), {"load_structured_pdf": _fake_load_structured_pdf}))
    monkeypatch.setattr(pipeline_runner, "AgenticGraphRunner", FakeRunner)
    monkeypatch.setattr(pipeline_runner, "MCPToolClient", FakeMCP)
    monkeypatch.setattr(pipeline_runner, "render_requirements_pdf", lambda *_: pdf_output)
    monkeypatch.setattr(pipeline_runner, "_build_chat_model", lambda *_args, **_kwargs: object())

    args = SimpleNamespace(
        data_dir=str(data_dir),
        vector_dir=str(tmp_path / "vector_store"),
        server_script="server.py",
        rebuild_store=False,
        skip_ingestion=True,
        top_k=3,
        output=str(output_path),
        pdf_output=str(pdf_output),
        decision_log=str(tmp_path / "decisions.jsonl"),
        throttle_enabled=False,
        web_search_enabled=False,
        max_web_queries_per_doc=0,
    )

    asyncio.run(pipeline_runner.run_pipeline(args))
    assert output_path.exists()


def test_run_pipeline_skip_without_store(monkeypatch, tmp_path):
    args = SimpleNamespace(
        data_dir=str(tmp_path / "data"),
        vector_dir=str(tmp_path / "vector_store"),
        server_script="server.py",
        rebuild_store=False,
        skip_ingestion=True,
        top_k=3,
        output=str(tmp_path / "out.json"),
        pdf_output=str(tmp_path / "out.pdf"),
        decision_log=str(tmp_path / "decisions.jsonl"),
        throttle_enabled=False,
        web_search_enabled=False,
        max_web_queries_per_doc=0,
    )

    monkeypatch.setattr(pipeline_runner, "load_dotenv", lambda: None)
    monkeypatch.setattr(pipeline_runner, "_ensure_llm_key", lambda *_: None)
    monkeypatch.setattr(pipeline_runner, "_log_runtime_context", lambda *_: None)
    monkeypatch.setattr(pipeline_runner, "vector_store_exists", lambda *_: False)

    with pytest.raises(ValueError):
        asyncio.run(pipeline_runner.run_pipeline(args))
