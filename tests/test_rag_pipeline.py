from AgenticAI.RAGPipeline import store_chunks_and_embeds
from AgenticAI.Chunker.Chunk import Chunk
from AgenticAI.PDF.Document import Document, Metadata, ElementType


class FakeModel:
    def encode(self, texts, normalize_embeddings=True):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeEmbedder:
    def __init__(self, *args, **kwargs):
        self.model = FakeModel()

    def embed_vectors_in_chunks(self, chunks):
        payload = []
        for chunk in chunks:
            payload.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk": chunk.model_dump_json(),
                    "embedding": [0.1, 0.2, 0.3],
                }
            )
        return 3, payload


class FakeStore:
    def __init__(self, *args, **kwargs):
        self.stored = []
        self.searches = 0

    def store_embeds_and_metadata(self, records):
        self.stored.extend(records)

    def top_k_search(self, query_embedding, top_k=10, query_text=None):
        self.searches += 1
        return [
            {
                "chunk": self.stored[0]["chunk"],
                "score": 0.9,
            }
        ]


def test_store_chunks_and_embeds(monkeypatch):
    monkeypatch.setattr("AgenticAI.RAGPipeline.VectorEmbedder", FakeEmbedder)
    monkeypatch.setattr("AgenticAI.RAGPipeline.resolve_backend", lambda *_: "faiss")
    monkeypatch.setattr("AgenticAI.RAGPipeline.FAISSStore", FakeStore)

    doc = Document(
        page_content="Body",
        meta_data=Metadata(source="doc.pdf", page=1, type=ElementType.PARAGRAPH),
    )
    chunk = Chunk(chunk_id="c1", document=doc, char_start=0, char_end=4)

    store_chunks_and_embeds([chunk])
