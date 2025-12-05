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

# HOTO

1. create `.env` in root folder with correct variables
2.

```shell
pip install -r requirements.txt
```

3.

```shell
# !!! dependencies should have, because pc is hard to run so i didnt try
pip install --upgrade pymilvus "pymilvus[model]"
```

4. install `pytorch` with a compatible version according to your cuda