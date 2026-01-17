import os
from typing import List, Dict
from AgenticAI.Chunker.Chunk import Chunk
from AgenticAI.Chunker.Chunker import Chunker
from AgenticAI.PDF.PDFParser import PDFParser
from AgenticAI.Vectorization.VectorEmbedder import VectorEmbedder
from AgenticAI.Vectorization.vectorStore.FAISS import FAISSStore
from AgenticAI.Vectorization.vectorStore.VectorStoreAdapter import IVectorStore
from AgenticAI.Vectorization.vectorStore.store_factory import resolve_backend

def store_chunks_and_embeds(chunk_list : List[Chunk]):
    vector_embedder : VectorEmbedder = VectorEmbedder()
    embed_tuple = vector_embedder.embed_vectors_in_chunks(chunk_list)
    dimension : int = embed_tuple[0]
    chunk_vector_embed_dict : List[Dict] = embed_tuple[1]

    backend = resolve_backend(os.getenv("VECTOR_STORE_BACKEND"))
    if backend == "milvus":
        from AgenticAI.Vectorization.vectorStore.Milvus import MilvusStore
        vector_store = MilvusStore.from_env(dimensions=dimension, use_cosine_similarity=True, auto_create=True)
    else:
        vector_store = FAISSStore(dimensions=dimension, use_cosine_similarity=True)

    vector_store.store_embeds_and_metadata(chunk_vector_embed_dict)


    test_query = "Sustainability reporting obligations for companies in the EU"
    query_embedding = vector_embedder.model.encode([test_query], normalize_embeddings=True)[0]

    results = vector_store.top_k_search(query_embedding, top_k=10)

    for r in results:
        chunk = Chunk.model_validate_json(r["chunk"])

        print("Similarity score:", r["score"])
        print("ChunkId:", chunk.chunk_id)
        print("Chunk content:", chunk.document.page_content, "\n")
        print("-" * 80)


if __name__ == "__main__":
    elements = PDFParser.load_structured_pdfs("./data")

    CHUNK_SIZE : int = 1000
    CHUNK_OVERLAP : int = 150
    CHUNK_TABLE_OVERLAP : int = 1
    chunks = Chunker.chunk_headings_with_paragraphs(
        documents=elements,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        table_row_overlap=CHUNK_TABLE_OVERLAP
    )

    store_chunks_and_embeds(chunks)
