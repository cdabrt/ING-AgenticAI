import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from AgenticAI.Chunker.Chunk import Chunk
from AgenticAI.Chunker.Chunker import Chunker
from AgenticAI.PDF.PDFParser import PDFParser
from AgenticAI.Vectorization.VectorEmbedder import VectorEmbedder
from AgenticAI.Vectorization.vectorStore.FAISS import FAISSStore

logger = logging.getLogger(__name__)


def _write_store_config(target_dir: Path, config: Dict):
    config_path = target_dir / "store_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def vector_store_exists(persist_dir: str) -> bool:
    path = Path(persist_dir)
    return all(
        [
            (path / "index.faiss").exists(),
            (path / "chunks.jsonl").exists(),
            (path / "store_config.json").exists(),
        ]
    )


def ingest_documents(
    data_dir: str,
    persist_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    table_row_overlap: int = 1,
) -> Dict:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    logger.info("Loading PDFs from %s", data_dir)
    documents = PDFParser.load_structured_pdfs(data_dir)
    if not documents:
        raise ValueError(f"No PDF content found in {data_dir}")

    logger.info("Chunking %s document elements", len(documents))
    chunks: List[Chunk] = Chunker.chunk_headings_with_paragraphs(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        table_row_overlap=table_row_overlap,
    )

    if not chunks:
        raise ValueError("Chunker did not return any chunks — check input documents")

    logger.info("Embedding %s chunks", len(chunks))
    vector_embedder = VectorEmbedder()
    dimension, chunk_vector_embed_dict = vector_embedder.embed_vectors_in_chunks(chunks)

    vector_store = FAISSStore(dimensions=dimension, use_cosine_similarity=True)
    vector_store.store_embeds_and_metadata(chunk_vector_embed_dict)

    target_dir = Path(persist_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    vector_store.persist(persist_dir)

    store_config = {
        "vector_store": "faiss",
        "use_cosine_similarity": True,
        "embedding_model_name": vector_embedder.model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "table_row_overlap": table_row_overlap,
        "dimension": dimension,
        "chunk_count": len(chunks),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_dir": str(data_path.resolve()),
    }
    _write_store_config(target_dir, store_config)

    logger.info("Persisted vector store with %s chunks to %s", len(chunks), persist_dir)

    return {
        "chunks": len(chunks),
        "documents": len(documents),
        "persist_dir": persist_dir,
        "store_config": store_config,
    }