import json
import logging
from typing import List, Dict
from dotenv import load_dotenv
import os
from AgenticAI.Chunker.Chunk import Chunk
from AgenticAI.Chunker.Chunker import Chunker
from AgenticAI.PDF.PDFParser import PDFParser
from AgenticAI.Vectorization.StoredChunk import StoredChunk
from AgenticAI.Vectorization.VectorEmbedder import VectorEmbedder
from AgenticAI.Vectorization.vectorStore.faiss.FAISS import FAISSStore
from AgenticAI.Vectorization.vectorStore.VectorStoreAdapter import IVectorStore
from AgenticAI.Vectorization.vectorStore.milvus.MilvusStore import MilvusStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def store_chunks_and_embeds(chunk_list: List[Chunk]):
    vector_embedder: VectorEmbedder = VectorEmbedder()
    embed_tuple = vector_embedder.embed_vectors_in_chunks(chunk_list)
    dimension: int = embed_tuple[0]
    chunk_vector_embed_dict: List[Dict] = embed_tuple[1]

    # TODO: still dependency to implementation. Might need something like a simple factory if we wish to expand upon this
    vector_store: IVectorStore = FAISSStore(dimensions=dimension, use_cosine_similarity=True)

    vector_store.store_embeds_and_metadata(chunk_vector_embed_dict)

    test_query = "Sustainability reporting obligations for companies in the EU"
    query_embedding = vector_embedder.model.encode([test_query], normalize_embeddings=True)[0]

    results = vector_store.top_k_search(query_embedding, top_k=10)

    for r in results:
        chunk = Chunker.model_validate_json(r["chunk"])

        print("Similarity score:", r["score"])
        print("ChunkId:", chunk.chunk_id)
        print("Chunk content:", chunk.document.page_content, "\n")
        print("-" * 80)


if __name__ == "__main__":
    load_dotenv()

    # elements = PDFParser.load_structured_pdfs("../data")

    # CHUNK_SIZE: int = 1000
    # CHUNK_OVERLAP: int = 150
    # CHUNK_TABLE_OVERLAP: int = 1
    # chunks = Chunker.chunk_headings_with_paragraphs(
    #     documents=elements,
    #     chunk_size=CHUNK_SIZE,
    #     chunk_overlap=CHUNK_OVERLAP,
    #     table_row_overlap=CHUNK_TABLE_OVERLAP
    # )
    # stored_chunks = [StoredChunk.map_chunk(chunk).model_dump() for chunk in chunks]
    # with open('../data/data.json', 'w', encoding="utf-8") as file:
    #     file.write(json.dumps(stored_chunks, ensure_ascii=False))

    print("Loading stored chunks...")
    mil = MilvusStore(os.getenv("MILVUS_URI"), os.getenv("MILVUS_TOKEN"), os.getenv("MILVUS_COLLECTION"))
    ret = mil.hybrid_search("Sustainability reporting obligations for companies in the EU")
    print(ret)