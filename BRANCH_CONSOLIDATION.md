# Branch Consolidation Plan (ING-AgenticAI)

This document summarizes what to combine into `main` to deliver a complete, end-to-end RAG pipeline for ING ESG compliance analysis. It maps each branch to its functional contributions, recommends a base branch, and outlines the specific file-level picks and rationale.

## Executive Recommendation

Use `origin/feat/backend-frontend` as the integration base (it contains the frontend, backend scaffolding, Docker Compose, and documentation). Then apply agentic pipeline upgrades from `origin/feat/LLM`, plus the PDF parsing improvements from `origin/feat/agnostic-pdf-parsing`.

This yields:
- A full-stack app (Next.js UI + FastAPI backend).
- An end-to-end agentic RAG pipeline (MCP + LangGraph + Gemini + FAISS).
- Improved PDF parsing (more robust heading detection across heterogeneous PDFs).
- Visualizer for requirement bundles.
- Optional PDF export of requirements.

## Branch Inventory

### 1) `origin/feat/backend-frontend` (Recommended Base)
Contains the full-stack scaffolding and most operational assets.

Key assets:
- API + Docker: `AgenticAI/main.py`, `AgenticAI/Dockerfile`, `Frontend/Dockerfile`.
- Frontend UI: `Frontend/ing-frontend/*`.
- Backend test server: `TestBackend/*` (mock endpoints and TODO placeholders; removed in consolidation).
- Docker Compose: `docker-compose.local.yml`, `docker-compose.local.test.yml`.
- Documentation and diagrams: `documentation/*`.
- Orchestration scripts: `run.ps1`, `run-test.sh`, etc.

Why this base:
- It already includes the UI and local stack orchestration.
- It is the only branch with the frontend and dockerized structure.
- It includes the editable `.d2` diagram sources alongside rendered images.

### 2) `origin/feat/LLM` (Agentic RAG Core)
Most complete agentic pipeline with MCP tooling and LangGraph orchestration.

Key assets to merge from this branch:
- `AgenticAI/agentic/langgraph_runner.py`
  - Adds web candidate screening, content vetting, and a requirements refinement step.
  - Adds safe failure handling for malformed web screening responses.
- `AgenticAI/agentic/pipeline_runner.py`
  - Adds runtime logging, validation, and PDF output via `pdf_report.py`.
  - Uses `DecisionLogger` and MCP client lifecycle.
- `AgenticAI/agentic/pdf_report.py`
  - Renders PDF summaries of requirement bundles.
- `AgenticAI/mcp_servers/regulation_server.py`
  - Adds `fetch_web_page` tool and sanitizes web content.
- `AgenticAI/pipeline/ingestion.py`
  - Robust ingestion flow with FAISS persistence and config write-out.
- `AgenticAI/Vectorization/vectorStore/FAISS.py`
  - Persistence and load support + safety checks.
- `AgenticAI/Vectorization/VectorEmbedder.py`
  - Query embedding support used by MCP retrieval.
- `visualizer/index.html`
  - Static UI for `artifacts/requirements.json`.
- `.env.example` and updated `README.md`
  - Clear environment variables and run instructions.

Why these files:
- They represent the most complete pipeline version.
- MCP + LangGraph + Gemini workflow is only fully described and implemented here.

### 3) `origin/feat/agnostic-pdf-parsing` (PDF Parsing Improvement)
Mostly identical to `origin/feat/backend-frontend`, except for the PDF parser enhancements.

Key asset to pick:
- `AgenticAI/PDF/PDFParser.py`
  - Adds font-size and spacing heuristics (not just bold detection).
  - Works better across heterogeneous PDF formats.

Why:
- Parsing quality is a critical upstream dependency for chunking and RAG quality.

### 4) `origin/feat/retrieval` and `origin/feat/retrieval-cud` (Legacy/Alternate RAG)
Older RAG pipeline without the agentic workflow.

Potentially useful assets (optional):
- `AgenticAI/Vectorization/vectorStore/milvus/*` (Milvus backend).
- `AgenticAI/Vectorization/StoredChunk.py` (metadata structure).
- `pyproject.toml`, `requirements.txt` (packaging alternative).

Why optional:
- These are not wired into the current MCP-based pipeline.
- Integrating multi-store support requires additional wiring in `ingestion.py` and MCP server.

### 5) `origin/dev`, `origin/feat/pdf_parsing`, `origin/feat/vectorization`
Early-stage components and architectural notes.

Status:
- Superseded by `origin/feat/LLM` and `origin/feat/backend-frontend`.
- Not recommended for direct merge.

## File-Level Merge Plan

### Base branch
Start from: `origin/feat/backend-frontend`

### Pull in agentic pipeline upgrades from `origin/feat/LLM`
1) Agentic orchestration:
   - `AgenticAI/agentic/langgraph_runner.py`
   - `AgenticAI/agentic/pipeline_runner.py`
   - `AgenticAI/agentic/pdf_report.py`
   - `AgenticAI/agentic/models.py`
   - `AgenticAI/agentic/decision_logger.py`
   - `AgenticAI/agentic/documents.py`
2) MCP tools:
   - `AgenticAI/mcp_servers/regulation_server.py`
   - `AgenticAI/mcp/client.py`
3) Ingestion + vector store:
   - `AgenticAI/pipeline/ingestion.py`
   - `AgenticAI/Vectorization/VectorEmbedder.py`
   - `AgenticAI/Vectorization/vectorStore/FAISS.py`
4) UI and assets:
   - `visualizer/index.html`
5) Config + docs:
   - `.env.example`
   - `README.md` (merge content, keep diagrams section from backend/frontend as needed)

### Apply PDF parsing improvement from `origin/feat/agnostic-pdf-parsing`
Replace:
- `AgenticAI/PDF/PDFParser.py`

### Keep from base (`origin/feat/backend-frontend`)
Preserve:
- `Frontend/ing-frontend/*`
- `docker-compose.local.yml`, `docker-compose.local.test.yml`
- `documentation/*` (keep `.d2` sources and rendered `.png` images together)
- `run.ps1`, `run-test.sh`, `run-test.ps1`
- `AgenticAI/Dockerfile`, `Frontend/Dockerfile`

## Integration Notes / Conflicts

1) `AgenticAI/agentic/langgraph_runner.py`
   - `origin/feat/LLM` has an additional requirements refinement node and web content triage error handling.
   - Prefer the `origin/feat/LLM` version to avoid brittle parsing failures.

2) `AgenticAI/agentic/pipeline_runner.py`
   - `origin/feat/LLM` includes PDF export and decision logging.
   - Keep `origin/feat/LLM` and integrate with any base CLI hooks as needed.

3) `AgenticAI/PDF/PDFParser.py`
   - Use `origin/feat/agnostic-pdf-parsing` for more robust heading detection.

4) `TestBackend/*`
   - Removed. The API surface is now provided by `AgenticAI/main.py`, which exposes pipeline and PDF endpoints backed by the local `data/` and `artifacts/` folders.

5) `README.md`
   - `origin/feat/LLM` has detailed architecture and run instructions.
   - `origin/feat/backend-frontend` adds diagrams. Merge both.

## What the Consolidated Main Will Contain

### Backend Pipeline
- PDF ingestion with chunking and FAISS persistence.
- MCP regulation server with tools:
  - `retrieve_chunks` (FAISS semantic search)
  - `web_search` (DuckDuckGo metadata)
  - `fetch_web_page` (sanitized fetch)
- LangGraph agent flow:
  - Query agent -> retrieval -> context assessment -> requirements -> refinement
- Decision logging to `artifacts/agent_decisions.jsonl`
- JSON output to `artifacts/requirements.json`
- Optional PDF export to `artifacts/requirements.pdf`

### Frontend + Visualization
- Next.js frontend in `Frontend/ing-frontend/*`.
- Static viewer in `visualizer/index.html`.

### DevOps / Docs
- Docker Compose for local run.
- Architecture and flow diagrams in `documentation/*` (keep `.d2` sources for edits).
- `.env.example` with GEMINI + model config.

## Suggested Next Actions (Optional)

1) Decide if you want Milvus support from `origin/feat/retrieval-cud`; it requires integration work.
2) Decide whether to keep large PDFs in `data/` in the repo or move to a data bucket.
