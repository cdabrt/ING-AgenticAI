from typing import List, Dict
from sentence_transformers import SentenceTransformer
from AgenticAI.Chunker.Chunk import Chunk


class VectorEmbedder:
    def __init__(self, model_name:str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_vectors_in_chunks(self, chunks:List[Chunk]) -> List[Dict]:
        texts = [chunk.document.page_content for chunk in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        list_output = []
        for chunk, emb in zip(chunks, embeddings):
            doc = {
                "chunk_id": chunk.chunk_id,
                # Dump JSON from the chunk model.
                "chunk": chunk.model_dump_json(),
                # Turn embeddings into a list, so it's JSON serializable.
                "embedding": emb.tolist()
            }
            list_output.append(doc)

        return list_output



