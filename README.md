# ING-AgenticAI

Quick architecture overview:

Data ingestion

Chunking

Embedding

Add/search embeddings in vector database: Chroma (adapter for other database implementations)

store relationships in grpah database: Neo4j (adapter for other database implementations)

GraphRAG: vector search -> graph expand -> LLM response generation

LLM adapter: Should have adapter for different APIs

Validation UI: Analyst reviews answers

