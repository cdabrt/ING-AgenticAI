# High-level one-sentence

RAG = **retrieve relevant chunks (via embeddings + vector search), optionally expand those with graph traversal (Neo4j), then synthesize a grounded answer with an LLM** — all while tracking provenance so every claim can be cited.

---

# Step-by-step (simple)

1. **PDF parsing**

   * What: Convert each PDF page into raw text and simple structure (headings, paragraphs, tables).
   * Why: Raw binary PDF is unreadable to models — we need text and metadata (source, page).
   * Typical tool: `UnstructuredPDFLoader` or any PDF → text extractor.
   * Output: a list of `Document` objects with `page_content` and `metadata` (filename, page).

2. **Chunking (structure-aware)**

   * What: Break long documents into smaller, meaningful pieces (“chunks”) — not random bytes.
   * Why: LLM context windows and embedding quality improve if chunks are paragraph-like and preserve sentence boundaries.
   * Typical approach: `RecursiveCharacterTextSplitter` or rule-based split that respects headings/tables.
   * Important metadata: chunk_id, source document, page number, character offsets.
   * Typical defaults: chunk ~500–1200 chars with 50–200 char overlap.

3. **Vectorization / Embeddings**

   * What: Turn each chunk into a numeric vector that captures semantic meaning.
   * Why: You can then ask “which chunks are semantically similar to this query?” using nearest-neighbour search.
   * Your choice: `SentenceTransformer("all-MiniLM-L6-v2")` → fast, local, free. Normalize vectors for cosine similarity.
   * Store: save the vector alongside the chunk metadata.

4. **Indexing (vector DB / FAISS)**

   * What: Put all vectors into a searchable index.
   * Options:

     * **FAISS** (local prototype): very fast, good for dev.
     * **Neo4j vector index** or **Pinecone/Weaviate** (production).
   * Query: index.search(query_embedding, k) → top-k chunk IDs + similarity scores.

5. **(Optional) Build a Knowledge Graph (Neo4j)**

   * What: Extract entities and relationships from chunks (LLM-assisted), create nodes like `:Regulation`, `:Requirement`, `:Actor`, and link them to `:Chunk` nodes.
   * Why: Graph traversal lets you surface connected context that vector search misses — e.g., “show all clauses that mention Actor X and Data Element Y.”
   * How: For each chunk, run a prompt that extracts entities & relations and write them as nodes/edges to Neo4j. Also store the chunk’s embedding as a property on the `:Chunk` node.
   * Benefit: hybrid retrieval — start from vector hits, then expand along graph edges for richer context.

6. **Query time — the RAG flow**

   * User asks a question.
   * a) **Embed the question** with Sentence-Transformer.
   * b) **Vector retrieve** top-k similar chunks from FAISS or Neo4j vector index.
   * c) **Graph expand** (optional): take those chunks as seeds, run a Cypher traversal to pull related entities/chunks/regulations.
   * d) **Assemble context**: concatenate the top chunks plus any graph-expanded nodes into a structured context (include chunk metadata for citations).
   * e) **Call the LLM (Groq Responses API)**: pass a strict prompt that instructs the model to *only* use the given context and to include citations for every factual claim (e.g. `[source:doc.pdf,page:5,chunk:abc123]`).
   * f) **Return answer** (and store it as a `pending` item if human validation is required).

7. **Human-in-the-loop (validate & commit)**

   * If analysts must approve: show the AI’s answer and citations in the UI (Streamlit), let them Approve/Reject/Correct.
   * On Approve: mark the result as validated in the DB (Neo4j/Postgres) and optionally attach the validated requirement node to the graph.
   * On Reject: allow editing and re-run the generation or adjust graph extraction.

---

# How the pieces map (concrete tech)

* PDF parsing & chunking → **LangChain loaders + RecursiveCharacterTextSplitter** (or custom)
* Embeddings → **sentence-transformers (all-MiniLM-L6-v2)** (local)
* Vector store → **FAISS** (local) or **Neo4j vector index** / **Pinecone** (prod)
* Graph / KG → **Neo4j** (nodes: Document, Chunk, Entity, Requirement; edges: CONTAINS, MENTIONS, RELATES_TO)
* Orchestration → **LangGraph** or a simple FastAPI controller that sequences the steps
* LLM / Chat → **Groq Responses API** (OpenAI-compatible)
* UI & validation → **Streamlit** + **FastAPI** endpoints

---

# Example user query flow (short)

1. User: “What data elements does Regulation X require?”
2. System: embed query → FAISS top-5 chunks.
3. System: expand those chunks in Neo4j to include related requirement nodes and citations.
4. Build context: list chunks + entity summaries with metadata.
5. Prompt Groq: "Use ONLY this context. Answer and cite each claim."
6. Groq returns answer with citation tags.
7. UI shows answer + source links; analyst approves → store as validated node.

---

# Practical tips & gotchas

* **Chunking matters**: too long → noisy context; too short → loss of meaning. Start with ~800 chars and 150 overlap.
* **Normalize embeddings** for cosine similarity (L2 norm) when using inner-product index (IndexFlatIP).
* **Keep provenance**: always attach chunk_id + document + page to every chunk — these are your citations.
* **Hybrid retrieval** is powerful: vector search finds relevance, graph traversal finds precise relations. Use both.
* **Prompt discipline**: force the model to cite and to answer `INSUFFICIENT CONTEXT` when it can’t find evidence. This reduces hallucinations.
* **History for Groq**: Groq’s Responses API is stateless — you must keep & pass conversation history yourself. Trim or summarize old turns to stay inside token limits.
* **Rate limits**: Groq free tiers have limits — implement retries and a local fallback (small llama.cpp) if needed.
* **Validation is crucial** for compliance outputs — never auto-publish without human review.

---

# Quick defaults to start with

* Chunk size: **600–1000 chars**; overlap: **100–200**.
* k (vector retrieve): **5–8**.
* Embedding model: `all-MiniLM-L6-v2`.
* Vector index: start with **FAISS** locally; migrate to Neo4j vector index or managed vector DB for production.
* LLM: **Groq Responses API** with model of your choice; include `system` instruction to force citations.