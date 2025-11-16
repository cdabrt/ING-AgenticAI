from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from AgenticAI.PDF.PDFParser import PDFParser
from AgenticAI.agentic.decision_logger import DecisionLogger
from AgenticAI.agentic.documents import group_documents_by_source
from AgenticAI.agentic.langgraph_runner import AgenticGraphRunner, AgenticState
from AgenticAI.mcp.client import MCPToolClient
from AgenticAI.pipeline.ingestion import ingest_documents, vector_store_exists

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _ensure_gemini_key():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) before running the pipeline")
    os.environ.setdefault("GOOGLE_API_KEY", api_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Agentic AI regulatory pipeline")
    parser.add_argument("--data-dir", default="data", help="Directory containing source PDFs")
    parser.add_argument(
        "--vector-dir",
        default="artifacts/vector_store",
        help="Directory where the FAISS store and metadata are stored",
    )
    parser.add_argument(
        "--server-script",
        default=str(Path(__file__).resolve().parents[1] / "mcp_servers" / "regulation_server.py"),
        help="Path to the MCP regulation server script",
    )
    parser.add_argument("--rebuild-store", action="store_true", help="Force rebuilding the vector store")
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of chunks to fetch per query during retrieval",
    )
    parser.add_argument(
        "--output",
        default="artifacts/requirements.json",
        help="Path where the consolidated requirements JSON will be saved",
    )
    parser.add_argument(
        "--decision-log",
        default="artifacts/agent_decisions.jsonl",
        help="Path to the JSONL file where agent decisions will be appended",
    )
    return parser.parse_args()


async def run_pipeline(args: argparse.Namespace):
    load_dotenv()
    _ensure_gemini_key()

    vector_dir = args.vector_dir
    if args.rebuild_store or not vector_store_exists(vector_dir):
        ingest_documents(data_dir=args.data_dir, persist_dir=vector_dir)

    documents = PDFParser.load_structured_pdfs(args.data_dir)
    grouped_docs = group_documents_by_source(documents)
    if not grouped_docs:
        raise ValueError("No parsed documents were found for agent processing")

    query_model_name = os.getenv("QUERY_MODEL_NAME", "gemini-2.5-flash")
    requirements_model_name = os.getenv("REQUIREMENTS_MODEL_NAME", "gemini-2.5-pro")

    query_model = ChatGoogleGenerativeAI(model=query_model_name, temperature=0.2)
    requirements_model = ChatGoogleGenerativeAI(model=requirements_model_name, temperature=0.1)

    aggregated_chunks: List[dict] = []
    aggregated_web: List[dict] = []
    doc_summaries: List[str] = []
    doc_types: List[str] = []
    doc_names: List[str] = []

    decision_logger = DecisionLogger(args.decision_log)

    server_script = args.server_script
    server_env = {"VECTOR_STORE_DIR": str(Path(vector_dir).resolve())}

    async with MCPToolClient(server_script=server_script, env=server_env) as mcp_client:
        runner = AgenticGraphRunner(
            query_model=query_model,
            requirements_model=requirements_model,
            mcp_client=mcp_client,
            retrieval_top_k=args.top_k,
            decision_logger=decision_logger,
        )

        for doc in grouped_docs:
            logger.info("Processing %s", doc.source)
            state: AgenticState = {
                "doc_source": doc.source,
                "doc_text": doc.text,
                "doc_headings": doc.headings,
            }
            processed_state = await runner.process_document(state)

            aggregated_chunks.extend(processed_state.get("retrieval_results", []))
            aggregated_web.extend(processed_state.get("web_context", []))
            doc_names.append(doc.source)
            doc_types.append(processed_state.get("document_type", "unknown"))
            doc_summaries.append(
                processed_state.get("document_summary", "No summary generated for this document.")
            )

        if not doc_names:
            raise RuntimeError("No documents were processed; cannot generate requirements.")

        combined_summary_lines = [
            f"- {name} ({doc_type}): {summary}"
            for name, doc_type, summary in zip(doc_names, doc_types, doc_summaries)
        ]
        combined_summary = "\n".join(combined_summary_lines)
        combined_type = ", ".join(sorted({dt for dt in doc_types if dt})) or "multiple"

        aggregated_state: AgenticState = {
            "doc_source": f"Combined documents: {', '.join(doc_names)}",
            "document_type": combined_type,
            "document_summary": combined_summary,
            "retrieval_results": aggregated_chunks,
            "web_context": aggregated_web,
        }

        result_state = await runner.generate_requirements(aggregated_state)
        bundle = result_state.get("requirements")
        if not bundle:
            raise RuntimeError("Agent 2 did not return consolidated requirements.")
        output_records: List[dict] = [bundle]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_records, handle, indent=2)

    logger.info("Wrote %s requirement bundles to %s", len(output_records), output_path)


def main():
    args = parse_args()
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
