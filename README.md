# ING-AgenticAI

## Updated architecture

The project now delivers an end-to-end agentic RAG workflow tailored for regulatory intelligence:

- **Ingestion** – `AgenticAI.pipeline.ingestion` parses PDFs with `pdfplumber`, chunks them, embeds the text with SentenceTransformers, and persists a FAISS index plus metadata under `artifacts/vector_store`.
- **MCP tooling** – `AgenticAI/mcp_servers/regulation_server.py` exposes two MCP tools via FastMCP: `retrieve_chunks` (FAISS semantic search) and `web_search` (DuckDuckGo). They are consumed through `AgenticAI.mcp.client.MCPToolClient` so every retrieval / open-web lookup happens through MCP.
- **LangGraph agents** – `AgenticAI.agentic.langgraph_runner.AgenticGraphRunner` wires two Gemini-powered agents:
  - *Agent 1 (query agent)* reads each document (grouped by source) and emits focused retrieval queries.
  - *Agent 2 (requirements agent)* reviews retrieved chunks, decides if more context is required (triggering the MCP `web_search` tool), and emits structured JSON with business/data requirements and citations.

All orchestration, including MCP server lifecycle, happens in `AgenticAI.agentic.pipeline_runner` which loops through uploaded documents one-by-one and records consolidated requirements to `artifacts/requirements.json`.

## Getting started

1. **Install dependencies**

	```bash
	pip install -e .
	```

2. **Configure environment**

	```bash
	cp .env.example .env  # if you keep one
	export GEMINI_API_KEY="<your_gemini_key>"
	```

	The same key is forwarded to `GOOGLE_API_KEY` for the LangChain Gemini client. Optionally set `VECTOR_STORE_DIR` if you want a custom persistence folder.

3. **Run the pipeline**

	```bash
	python -m AgenticAI.agentic.pipeline_runner --data-dir data --vector-dir artifacts/vector_store
	```

	The runner will:
	- rebuild the FAISS store (unless it already exists and `--rebuild-store` is omitted),
	- start the MCP regulation server automatically,
	- execute the LangGraph pipeline per document,
	- persist the JSON requirements bundle to `artifacts/requirements.json`.

4. **Inspect results** – open `artifacts/requirements.json` to review generated business & data requirements alongside citations to both document chunks and optional online sources.

## Visualizing the JSON output

For a friendlier view of the generated requirements, a static viewer lives under `visualizer/index.html`.

1. Serve the repository root (so the page can fetch `artifacts/requirements.json` and the source PDFs):

	```bash
	python -m http.server 8000
	```

2. Open `http://localhost:8000/visualizer/` in a browser. The UI lists business and data requirements, shows rationales, and renders document/online sources as clickable pills. Document citations link back to the PDF page (e.g., `data/CELEX_...pdf#page=4`), while online sources open the captured URLs.

## Useful CLI flags

- `--rebuild-store` – force ingestion even if a vector store already exists.
- `--top-k` – number of FAISS chunks fetched per query.
- `--server-script` – run a custom MCP server implementation if needed.
- `--output` – change the output JSON path.

## Project layout

- `AgenticAI/pipeline/ingestion.py` – ingestion & FAISS persistence helpers.
- `AgenticAI/mcp_servers/regulation_server.py` – FastMCP server exposing retrieval + web search tools.
- `AgenticAI/mcp/client.py` – lightweight stdio MCP client for the LangGraph runner.
- `AgenticAI/agentic/*` – document grouping utilities, Pydantic schemas, and LangGraph orchestration.
- `AgenticAI/agentic/pipeline_runner.py` – main entrypoint that ties everything together.
- `visualizer/index.html` – lightweight UI for browsing the requirement bundles and jumping to cited sources.

This setup ensures the regulatory requirements agent uses LangGraph, Gemini, and MCP tools without shortcuts.

