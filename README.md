# ING-AgenticAI

Quick architecture overview:

Data ingestion

Chunking

Embedding

Add/search embeddings in vector database: Chroma, FAISS, Neo4j vector index or Pinecone/Weaviate

store relationships in grpah database: Neo4j (adapter for other database implementations)

GraphRAG: vector search -> graph expand -> LLM response generation

LLM adapter: Should have adapter for different APIs

Validation UI: Analyst reviews answers

```shell
# !!! dependencies should have, because pc is hard to run so i didnt try
pip install --upgrade pymilvus "pymilvus[model]"
```