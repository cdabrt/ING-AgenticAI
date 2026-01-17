# RAG Architecture Justification (Late 2025 / Early 2026)

This note captures the current RAG architecture decision and its justification based on the assessment provided in this project discussion. It is intended to be included in the project report.

## Current System (Baseline)

- Dense retrieval only using Milvus (SentenceTransformer embeddings).
- Optional web search if the agent decides more context is required.
- No sparse retrieval, no reranker, no explicit hybrid fusion step.

## Options Considered (2026 Perspective)

### Option 1: Hybrid BM25 + Dense (Milvus) + RRF + Reranker

- Status: Industry standard for production RAG.
- Why it fits regulatory domains:
  - Lexical precision is critical for clause IDs, section numbers, and named regulations.
  - Dense retrieval handles semantic paraphrases and broader intent.
  - RRF provides robust fusion without heavy weighting tuning.
  - Lightweight rerankers (e.g., BGE-style or Cohere-class) provide strong precision gains at modest latency cost.

### Option 2: Hierarchical Retrieval (Parent/Child)

- Status: Useful chunking strategy, but not sufficient alone.
- Best used as a supporting technique within hybrid retrieval (Option 1).

### Option 3: Multi-Query / Query Expansion

- Status: Improves recall but adds latency linearly with query count.
- Often replaced by agentic, conditional re-querying in 2026 workflows.

### Option 4: GraphRAG (Knowledge Graph RAG)

- Status: State of the art for cross-reference reasoning and global context.
- Strengths:
  - Explicit linkage of definitions, amendments, and cross-references.
  - Better for global, multi-document reasoning questions.
- Trade-off: Higher engineering cost (graph extraction, storage, and traversal).

## Decision: Option 1 is the Best Justified Choice Today

Based on the provided assessment, Option 1 (Hybrid BM25 + Dense + RRF + Reranker) is the most practical and accurate architecture for the project today. It balances:

- Accuracy for regulatory phrasing and section identifiers.
- Semantic coverage for paraphrased questions.
- Low operational complexity compared to GraphRAG.
- Latency and cost that remain suitable for production use.

## Why GraphRAG is Not Chosen Now

GraphRAG is technically stronger for cross-reference reasoning, but the overhead of building and maintaining a graph is high. The current scope prioritizes reliability and speed for clause-level retrieval, which hybrid search delivers with far less engineering effort.

## Proposed Upgrade Path

Phase 1 (Now):
- Implement hybrid retrieval (BM25 + Milvus) with RRF and a reranker.
- Keep hierarchical chunk metadata so the reranker sees enough context.

Phase 2 (Later, if needed):
- If users need cross-document reasoning (amendments, definition propagation), introduce a focused knowledge graph.
- Use the graph to augment hybrid retrieval rather than replace it.

## Implementation Plan (High Level)

### Plan A: Hybrid BM25 + Milvus + RRF + Reranker (Recommended)

- Add a BM25 index over the same chunk corpus and persist it with the vector store config.
- Implement hybrid retrieval by querying BM25 + Milvus and fusing results with RRF.
- Add a reranker to refine the fused top-k list before generation.
- Expose configuration toggles such as `RETRIEVAL_MODE`, `RRF_K`, and `RERANKER_ENABLED`.
- Update logs and tests to cover hybrid retrieval output consistency.

### Plan B: GraphRAG (Advanced)

- Build an entity and relationship extraction step from chunks.
- Store entities and edges in a graph database (e.g., Neo4j) alongside Milvus.
- Add graph traversal to augment retrieval context for cross-document reasoning.
- Keep vector retrieval for clause-level lookup, and fuse graph context into prompts.

## Hardware Notes

- Plan A runs on CPU and works on Apple Silicon; no CUDA required.
- Plan B also runs on CPU but is slower for extraction and graph construction.
- GPU (CUDA) is optional and only needed to improve throughput for embedding/reranking at scale.

## Notes and Limitations

- This justification summarizes the user-provided 2026 assessment and has not been independently validated here.
- Any external references should be added separately if required for formal reporting.
