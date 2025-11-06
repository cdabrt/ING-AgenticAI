import faiss
import numpy
from typing import override, List, Dict

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
    def search(self, query_embedding, top_k=5):
        query_vec = numpy.array([query_embedding], dtype='float32')
        if self.use_cosine_similarity:
            faiss.normalize_L2(query_vec)

        D, I = self.index.search(query_vec, top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            results.append({"chunk": self.chunks[idx], "score": score})
        return results