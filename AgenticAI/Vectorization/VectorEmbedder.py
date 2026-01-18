import os
from typing import List, Dict

from sentence_transformers import SentenceTransformer

from AgenticAI.Chunker.Chunk import Chunk


class VectorEmbedder:
    def __init__(self, model_name: str | None = None):
        resolved_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        self.model_name = resolved_name
        self.model = SentenceTransformer(resolved_name)

    def embed_vectors_in_chunks(self, chunks: List[Chunk]) -> tuple[int, List[Dict]]:
        texts = [chunk.document.page_content for chunk in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        list_output: List[Dict] = []
        for chunk, emb in zip(chunks, embeddings):
            doc = {
                "chunk_id": chunk.chunk_id,
                # Dump JSON from the chunk model.
                "chunk": chunk.model_dump_json(),
                "chunk_text": chunk.document.page_content,
                "doc_source": chunk.document.meta_data.source,
                # Turn embeddings into a list, so it's JSON serializable.
                "embedding": emb.tolist()
            }
            list_output.append(doc)

        return embeddings[0].shape[0], list_output
    def embed_queries(self, queries: List[str]):
        return self.model.encode(queries, normalize_embeddings=True)


