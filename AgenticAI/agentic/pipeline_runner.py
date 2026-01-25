from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
from importlib import metadata
from pathlib import Path
from typing import Callable, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from AgenticAI.PDF.Document import Document
from AgenticAI.PDF.PDFParser import PDFParser
from AgenticAI.agentic.decision_logger import DecisionLogger
from AgenticAI.agentic.documents import group_documents_by_source
from AgenticAI.agentic.langgraph_runner import AgenticGraphRunner, AgenticState
from AgenticAI.agentic.models import RequirementBundle
from AgenticAI.agentic.pdf_report import render_requirements_pdf
from AgenticAI.mcp.client import MCPToolClient
from AgenticAI.pipeline.ingestion import ingest_documents, vector_store_exists
from AgenticAI.Vectorization.vectorStore.store_factory import load_store_config, resolve_backend

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PACKAGE_PROBES = (
    "langchain",
    "langgraph",
    "langchain-google-genai",
    "langchain-openai",
    "openai",
    "pymilvus",
    "google-generativeai",
    "google-ai-generativelanguage",
    "modelcontextprotocol",
    "mcp",
)


def _load_document_index(vector_dir: str) -> Dict[str, Dict[str, str]]:
    config = load_store_config(vector_dir)
    index_path = config.get("document_index") if config else None
    if index_path:
        path = Path(index_path)
    else:
        path = Path(vector_dir) / "document_index.json"

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {item["source"]: item for item in payload if isinstance(item, dict) and "source" in item}
    return {}


def _get_llm_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    return provider or "gemini"


def _normalize_model_name(provider: str, model_name: str) -> str:
    normalized = (model_name or "").strip()
    if not normalized:
        return normalized
    if provider == "openrouter":
        if "/" not in normalized and normalized.startswith("gemini-"):
            return f"google/{normalized}"
        return normalized
    if provider == "gemini" and normalized.startswith("google/"):
        return normalized.split("/", 1)[1]
    return normalized


def _parse_optional_bool(value: str | None) -> Optional[bool]:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    return None


def _ensure_llm_key(provider: str) -> None:
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("Set OPENROUTER_API_KEY before running the pipeline")
        os.environ.setdefault("OPENAI_API_KEY", api_key)
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        os.environ.setdefault("OPENAI_API_BASE", base_url)
        os.environ.setdefault("OPENAI_BASE_URL", base_url)
        logger.info(
            "LLM key loaded (OPENROUTER_API_KEY present=%s, OPENROUTER_BASE_URL=%s)",
            bool(os.getenv("OPENROUTER_API_KEY")),
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
        return

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) before running the pipeline")
    os.environ.setdefault("GOOGLE_API_KEY", api_key)
    logger.info(
        "LLM key loaded (GEMINI_API_KEY present=%s, GOOGLE_API_KEY present=%s)",
        bool(os.getenv("GEMINI_API_KEY")),
        bool(os.getenv("GOOGLE_API_KEY")),
    )


def _build_chat_model(
    provider: str,
    model_name: str,
    temperature: float,
    reasoning_enabled: Optional[bool] = None,
) -> BaseChatModel:
    if provider == "openrouter":
        extra_args = {}
        if reasoning_enabled is not None:
            extra_args["reasoning"] = {"enabled": reasoning_enabled}
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            **extra_args,
        )
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


def _log_runtime_context(data_dir: str, vector_dir: str):
    logger.info(
        "Runtime context: python=%s, platform=%s, working_dir=%s",
        platform.python_version(),
        platform.platform(),
        os.getcwd(),
    )
    versions: List[str] = []
    for package in PACKAGE_PROBES:
        try:
            versions.append(f"{package}={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            versions.append(f"{package}=missing")
    logger.info("Key packages: %s", ", ".join(versions))
    logger.info("Data directory: %s (exists=%s)", data_dir, Path(data_dir).exists())
    logger.info("Vector directory: %s (exists=%s)", vector_dir, Path(vector_dir).exists())
    logger.info("Vector store config path: %s", Path(vector_dir) / "store_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Agentic AI regulatory pipeline")
    parser.add_argument("--data-dir", default="data", help="Directory containing source PDFs")
    parser.add_argument(
        "--vector-dir",
        default="artifacts/vector_store",
        help="Directory where vector store metadata is stored",
    )
    parser.add_argument(
        "--server-script",
        default=str(Path(__file__).resolve().parents[1] / "mcp_servers" / "regulation_server.py"),
        help="Path to the MCP regulation server script",
    )
    parser.add_argument("--rebuild-store", action="store_true", help="Force rebuilding the vector store")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip embedding; require existing vector store")
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of chunks to fetch per query during retrieval",
    )
    parser.add_argument(
        "--web-search-enabled",
        default=None,
        help="Enable open-web search (true/false).",
    )
    parser.add_argument(
        "--max-web-queries-per-doc",
        type=int,
        default=None,
        help="Maximum open-web queries per document.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/requirements.json",
        help="Path where the consolidated requirements JSON will be saved",
    )
    parser.add_argument(
        "--pdf-output",
        default="artifacts/requirements.pdf",
        help="Path where the readable PDF export will be saved",
    )
    parser.add_argument(
        "--decision-log",
        default="artifacts/agent_decisions.jsonl",
        help="Path to the JSONL file where agent decisions will be appended",
    )
    return parser.parse_args()


ProgressCallback = Callable[[str, float, Optional[str], Optional[Dict[str, Optional[str] | int]]], None]


async def run_pipeline(args: argparse.Namespace, progress_callback: Optional[ProgressCallback] = None):
    load_dotenv()
    provider = _get_llm_provider()
    _ensure_llm_key(provider)
    _log_runtime_context(args.data_dir, args.vector_dir)

    progress_stages = (
        ("init", 0.02),
        ("ingestion", 0.25),
        ("parsing", 0.08),
        ("mcp_start", 0.05),
        ("documents", 0.4),
        ("requirements", 0.12),
        ("output", 0.08),
    )

    def _overall_progress(stage: str, fraction: float) -> float:
        running = 0.0
        for name, weight in progress_stages:
            if name == stage:
                return min(1.0, running + weight * max(0.0, min(1.0, fraction)))
            running += weight
        return min(1.0, running)

    current_stage = "init"
    current_progress = 0.0
    current_message: Optional[str] = None
    llm_calls_done = 0

    def _notify(
        stage: str,
        fraction: float,
        message: Optional[str] = None,
        meta: Optional[Dict[str, Optional[str] | int]] = None,
    ) -> None:
        nonlocal current_stage, current_progress, current_message
        current_stage = stage
        current_progress = _overall_progress(stage, fraction)
        current_message = message
        if progress_callback:
            progress_callback(stage, current_progress, message, meta)

    def _notify_llm(meta: Dict[str, Optional[str] | int]) -> None:
        nonlocal llm_calls_done
        increment = meta.get("llm_increment")
        if isinstance(increment, int) and increment > 0:
            llm_calls_done += increment
        payload = {
            "llm_step": meta.get("llm_step"),
            "llm_detail": meta.get("llm_detail"),
            "llm_calls_done": llm_calls_done,
            "llm_calls_total": meta.get("llm_calls_total"),
            "current_doc": meta.get("current_doc"),
        }
        if progress_callback:
            progress_callback(current_stage, current_progress, current_message, payload)

    _notify("init", 1.0, "Pipeline initialized")

    vector_dir = args.vector_dir
    vector_backend = resolve_backend(os.getenv("VECTOR_STORE_BACKEND"))
    vector_exists = vector_store_exists(vector_dir)
    skip_ingestion = bool(getattr(args, "skip_ingestion", False))
    logger.info(
        "Vector store present=%s (backend=%s, rebuild=%s, skip_ingestion=%s)",
        vector_exists,
        vector_backend,
        args.rebuild_store,
        skip_ingestion,
    )
    if skip_ingestion and not vector_exists:
        raise ValueError("Vector store not found; run embeddings before generating requirements.")
    if not skip_ingestion and (args.rebuild_store or not vector_exists or vector_backend == "milvus"):
        _notify("ingestion", 0.0, "Ingesting PDFs into vector store")
        logger.info("Starting ingestion into %s", vector_dir)
        await asyncio.to_thread(
            ingest_documents,
            data_dir=args.data_dir,
            persist_dir=vector_dir,
            vector_store_backend=vector_backend,
            include_sources=None,
            progress_callback=lambda progress, message=None: _notify("ingestion", progress, message),
        )
        logger.info("Finished ingestion into %s", vector_dir)
        _notify("ingestion", 1.0, "Vector store ready")
    else:
        _notify("ingestion", 1.0, "Vector store already present")

    _notify("parsing", 0.0, "Parsing PDFs")
    pdf_files = sorted(Path(args.data_dir).glob("*.pdf"), key=lambda path: path.name.lower())
    if skip_ingestion:
        document_index = _load_document_index(vector_dir)
        embedded_sources = sorted(document_index.keys())
        if not embedded_sources:
            raise ValueError("No embedded PDFs found. Run embeddings before generating requirements.")
        resolved_files: List[Path] = []
        missing_sources: List[str] = []
        for source in embedded_sources:
            path = Path(args.data_dir) / source
            if path.exists():
                resolved_files.append(path)
            else:
                missing_sources.append(source)
        if missing_sources:
            logger.warning("Embedded sources missing from data dir: %s", ", ".join(missing_sources))
        pdf_files = resolved_files
    if not pdf_files:
        raise ValueError("No PDF files found for parsing")

    documents: List[Document] = []
    total_files = len(pdf_files)
    for idx, pdf_file in enumerate(pdf_files, start=1):
        _notify("parsing", (idx - 1) / total_files, f"Parsing {pdf_file.name} ({idx}/{total_files})")
        parsed_docs = await asyncio.to_thread(PDFParser.load_structured_pdf, pdf_file)
        documents.extend(parsed_docs)
        _notify("parsing", idx / total_files, f"Parsed {pdf_file.name} ({idx}/{total_files})")
    grouped_docs = group_documents_by_source(documents)
    logger.info("Parsed %s documents -> %s grouped sources", len(documents), len(grouped_docs))
    for doc in grouped_docs:
        logger.info("Document ready: source=%s, heading_count=%s, text_chars=%s", doc.source, len(doc.headings), len(doc.text))
    if not grouped_docs:
        raise ValueError("No parsed documents were found for agent processing")
    _notify("parsing", 1.0, "PDF parsing complete")

    query_model_name = os.getenv("QUERY_MODEL_NAME", "gemini-3-flash-preview")
    requirements_model_name = os.getenv("REQUIREMENTS_MODEL_NAME", "gemini-3-flash-preview")

    query_model_name = _normalize_model_name(provider, query_model_name)
    requirements_model_name = _normalize_model_name(provider, requirements_model_name)

    query_reasoning_enabled = _parse_optional_bool(os.getenv("QUERY_MODEL_REASONING_ENABLED"))
    requirements_reasoning_enabled = _parse_optional_bool(os.getenv("REQUIREMENTS_MODEL_REASONING_ENABLED"))

    query_model = _build_chat_model(
        provider,
        query_model_name,
        temperature=0.2,
        reasoning_enabled=query_reasoning_enabled,
    )
    requirements_model = _build_chat_model(
        provider,
        requirements_model_name,
        temperature=0.1,
        reasoning_enabled=requirements_reasoning_enabled,
    )

    raw_web_search_enabled = getattr(args, "web_search_enabled", None)
    if isinstance(raw_web_search_enabled, str):
        web_search_enabled = _parse_optional_bool(raw_web_search_enabled)
    elif isinstance(raw_web_search_enabled, bool):
        web_search_enabled = raw_web_search_enabled
    else:
        web_search_enabled = None
    if web_search_enabled is None:
        env_web_search = _parse_optional_bool(os.getenv("WEB_SEARCH_ENABLED"))
        web_search_enabled = True if env_web_search is None else env_web_search

    max_web_queries_per_doc = getattr(args, "max_web_queries_per_doc", None)
    if max_web_queries_per_doc is None:
        env_max_queries = os.getenv("WEB_SEARCH_MAX_QUERIES_PER_DOC", "3")
        try:
            max_web_queries_per_doc = int(env_max_queries)
        except ValueError:
            max_web_queries_per_doc = 3
    max_web_queries_per_doc = max(0, int(max_web_queries_per_doc))

    doc_bundles: List[RequirementBundle] = []
    doc_names: List[str] = []

    decision_logger = DecisionLogger(args.decision_log)

    server_script = args.server_script
    server_env = {"VECTOR_STORE_DIR": str(Path(vector_dir).resolve())}
    _notify("mcp_start", 0.0, "Starting MCP server")
    logger.info("Starting MCP server script=%s with env=%s", server_script, server_env)

    async with MCPToolClient(server_script=server_script, env=server_env) as mcp_client:
        _notify("mcp_start", 1.0, "MCP server ready")
        try:
            tools = await mcp_client.list_tools()
            logger.info("MCP tools available: %s", ", ".join(tools))
        except Exception as exc:  # noqa: BLE001 - diagnostics path
            logger.error("Failed to list MCP tools: %s", exc)
        throttle_enabled = getattr(args, "throttle_enabled", True)
        runner = AgenticGraphRunner(
            query_model=query_model,
            requirements_model=requirements_model,
            mcp_client=mcp_client,
            retrieval_top_k=args.top_k,
            web_search_enabled=web_search_enabled,
            max_web_queries_per_doc=max_web_queries_per_doc,
            decision_logger=decision_logger,
            throttle_enabled=throttle_enabled,
            llm_status_callback=_notify_llm,
        )

        total_docs = len(grouped_docs)
        for idx, doc in enumerate(grouped_docs, start=1):
            _notify("documents", (idx - 1) / total_docs, f"Processing {doc.source} ({idx}/{total_docs})")
            logger.info("Processing %s", doc.source)
            state: AgenticState = {
                "doc_source": doc.source,
                "doc_text": doc.text,
                "doc_headings": doc.headings,
            }

            def doc_status(message: str, fraction: float) -> None:
                progress = (idx - 1 + max(0.0, min(1.0, fraction))) / total_docs
                _notify("documents", progress, f"{message} - {doc.source} ({idx}/{total_docs})")

            def doc_status_scaled(message: str, fraction: float, start: float, end: float) -> None:
                span = max(0.0, end - start)
                doc_status(message, start + span * max(0.0, min(1.0, fraction)))

            processed_state = await runner.process_document(state, status_callback=doc_status)
            evidence_state = await runner.distill_evidence(
                processed_state,
                status_callback=lambda message, fraction: doc_status_scaled(message, fraction, 0.72, 0.9),
            )
            processed_state.update(evidence_state)
            requirements_state = await runner.generate_requirements(
                processed_state,
                status_callback=lambda message, fraction: doc_status_scaled(message, fraction, 0.9, 1.0),
            )
            bundle_payload = requirements_state.get("requirements")
            if not bundle_payload:
                raise RuntimeError(f"Agent 2 did not return requirements for {doc.source}.")
            doc_bundles.append(RequirementBundle.model_validate(bundle_payload))
            _notify("documents", idx / total_docs, f"Completed {doc.source} ({idx}/{total_docs})")

            doc_names.append(doc.source)

        if not doc_names:
            raise RuntimeError("No documents were processed; cannot generate requirements.")

        _notify("requirements", 0.0, "Generating consolidated requirements")
        def requirements_status(message: str, fraction: float) -> None:
            _notify("requirements", fraction, message)

        merged_bundle = await runner.merge_requirements(
            doc_bundles,
            doc_names,
            status_callback=requirements_status,
        )
        _notify("requirements", 1.0, "Requirements generated")
        output_records: List[dict] = [merged_bundle.model_dump()]

    _notify("output", 0.0, "Writing output files")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_records, handle, indent=2)

    logger.info("Wrote %s requirement bundles to %s", len(output_records), output_path)
    _notify("output", 0.5, "Requirements JSON saved")

    try:
        pdf_path = await asyncio.to_thread(render_requirements_pdf, output_records, args.pdf_output)
        logger.info("Rendered PDF summary to %s", pdf_path)
        _notify("output", 1.0, "PDF summary rendered")
    except Exception:  # noqa: BLE001 - diagnostics for PDF failures
        logger.exception("Failed to render PDF summary")
        _notify("output", 1.0, "Pipeline completed without PDF")


def main():
    args = parse_args()
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
