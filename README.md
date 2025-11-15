# ING-AgenticAI

## Updated architecture

The project now delivers an end-to-end agentic RAG workflow tailored for regulatory intelligence. At a high level:

- **Ingestion** – `AgenticAI.pipeline.ingestion` parses PDFs with `pdfplumber`, chunks them, embeds text with SentenceTransformers, and persists a FAISS index plus metadata under `artifacts/vector_store`.
- **MCP tooling** – `AgenticAI/mcp_servers/regulation_server.py` exposes two MCP tools via FastMCP: `retrieve_chunks` (FAISS semantic search) and `web_search` (DuckDuckGo + html2text). They are consumed through `AgenticAI.mcp.client.MCPToolClient`, so every retrieval or open-web lookup stays behind the MCP boundary.
- **LangGraph agents** – `AgenticAI.agentic.langgraph_runner.AgenticGraphRunner` wires a small LangGraph (`query_agent → retrieval → context → requirements → END`) across two Gemini models (Flash + Pro) and the MCP tools.

All orchestration, including MCP server lifecycle, happens in `AgenticAI.agentic.pipeline_runner`, which loops through uploaded documents one-by-one and records consolidated requirements to `artifacts/requirements.json`.

### Agent orchestration in detail

`AgenticGraphRunner` defines an `AgenticState` dictionary that flows through four LangGraph nodes:

1. **`_query_agent_node` (Agent 1 – regulatory discovery strategist)**
	- Model: Gemini Flash (`ChatGoogleGenerativeAI` with temperature 0.2).
	- Prompt: Reads the grouped document text + headings and returns a `QuerySpecification` (document type, short summary, and a small set of retrieval queries).
	- Tools: None. This node is pure model inference.

2. **`_retrieval_node` (deterministic vector search)**
	- For each query from Agent 1, calls `MCPToolClient.call_tool("retrieve_chunks", {"query": query, "top_k": retrieval_top_k})`.
	- The MCP tool embeds the query with the persisted SentenceTransformer model, performs FAISS search, and returns chunk metadata. Each row is validated via the `RetrievalChunk` model before being appended to `state["retrieval_results"]`.

3. **`_context_node` (context assessor)**
	- Model: Same Gemini Flash instance but with a different prompt instructing it to decide whether additional context is needed.
	- Output: `ContextAssessment` containing the boolean `needs_additional_context`, up to three `missing_information_queries`, and an explanation.
	- Tools: If `needs_additional_context` is `True`, the node invokes `web_search` per missing query (max three). The MCP server runs DuckDuckGo, fetches each result with `httpx`, strips markup, and returns a `WebResult` list. These are stored under `state["web_context"]`.

4. **`_requirements_node` (Agent 2 – regulatory implementation architect)**
	- Model: Gemini Pro (`ChatGoogleGenerativeAI` with temperature 0.1).
	- Prompt: Receives document metadata, the JSON-serialized retrieval chunks, and any web context. It emits a `RequirementBundle` with structured business/data requirements, rationales, and citations (chunk IDs/URLs) plus assumptions.
	- Tools: Consumes previously gathered context only; no additional tool calls are made here.

### Document vs. portfolio workflow

- `pipeline_runner` loads parsed PDFs via `PDFParser`, groups them per source (`documents.group_documents_by_source`), and for each group constructs the initial state `{doc_source, doc_text, doc_headings}`.
- `runner.process_document` executes the first three nodes (query, retrieval, context). This ensures each document produces its own summary, document type, chunk list, and optional web context.
- After all documents finish, `pipeline_runner` combines their summaries/types/chunks into a single aggregate state and calls `runner.generate_requirements`, which executes only `_requirements_node`. The result is one consolidated `RequirementBundle` persisted to `artifacts/requirements.json`.

### Tool visibility and permissions

- Only the retrieval node ever calls `retrieve_chunks`.
- Only the context node conditionally calls `web_search` (maximum three queries per document).
- Agents never bypass MCP: they interact solely via `MCPToolClient`, which keeps a single stdio session open for the duration of the pipeline run.

### LLM context lifecycle

- Every LangGraph node builds an independent prompt → model → parser chain and calls `ainvoke` with just the current `AgenticState` fields. There is no shared chat history or conversation memory between nodes.
- Within a document, Agent 1 and the context assessor both use the Gemini Flash client, but each invocation starts from a fresh prompt; the only “memory” is explicit state values (e.g., `document_summary`) passed as template variables.
- Across documents, `pipeline_runner` creates a brand-new base state for each source, so Flash is never implicitly primed by previous documents.
- The Gemini Pro requirements agent runs once at the end with the aggregated summaries/chunks/web context. Its context window contains only that final prompt—no previous exchanges are retained.

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

