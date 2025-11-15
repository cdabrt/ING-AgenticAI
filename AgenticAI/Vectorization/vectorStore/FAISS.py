from pathlib import Path
from typing import override, List, Dict

import faiss
import numpy

from AgenticAI.Chunker.Chunk import Chunk
from AgenticAI.Vectorization.vectorStore.VectorStoreAdapter import IVectorStore


class FAISSStore(IVectorStore):
    def __init__(self, dimensions:int, use_cosine_similarity=True):
        super().__init__(dimensions, use_cosine_similarity)

        # FAISS index
        self.index = faiss.IndexFlatIP(self.dimensions) if use_cosine_similarity else faiss.IndexFlatL2(dimensions)

        # In-memory metadata store. FAISS only stores vector embeddings. We keep chunk metadata in an in-memory list.
        # This is why FAISS is meant to be used in development, and not in production
        self.chunk_store: List[Chunk] = []

    @override
    def store_embeds_and_metadata(self, chunks_with_embeds : List[Dict]):
        embeddings = numpy.array([chunk['embedding'] for chunk in chunks_with_embeds], dtype='float32')

        if self.use_cosine_similarity:
            faiss.normalize_L2(embeddings)

        # Indexes in FAISS are kept in the same order as the order in the chunks_with_embeds list.
        # Thus later, when doing a search query, the FAISS index 0 corresponds exactly to self.chunk_store[0]
        self.index.add(embeddings)
        self.chunk_store.extend([chunk['chunk'] for chunk in chunks_with_embeds])

    @override
    def top_k_search(self, query_embedding, top_k=5):
        query_vec = numpy.array([query_embedding], dtype='float32')
        if self.use_cosine_similarity:
            faiss.normalize_L2(query_vec)

        if self.index.ntotal == 0:
            return []

        D, I = self.index.search(query_vec, top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.chunk_store):
                continue
            results.append({"chunk": self.chunk_store[idx], "score": score})
        return results

    def persist(self, directory: str):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        index_path = path / "index.faiss"
        chunks_path = path / "chunks.jsonl"

        faiss.write_index(self.index, str(index_path))
        with chunks_path.open("w", encoding="utf-8") as handle:
            for chunk_json in self.chunk_store:
                handle.write(chunk_json)
                handle.write("\n")

    @classmethod
    def load(cls, directory: str, use_cosine_similarity: bool = True) -> "FAISSStore":
        path = Path(directory)
        index_path = path / "index.faiss"
        chunks_path = path / "chunks.jsonl"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"Missing FAISS artifacts in {directory}")

        index = faiss.read_index(str(index_path))
        dimensions = index.d

        store = cls(dimensions=dimensions, use_cosine_similarity=use_cosine_similarity)
        store.index = index

        with chunks_path.open("r", encoding="utf-8") as handle:
            store.chunk_store = [line.strip() for line in handle if line.strip()]

        return store