import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

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


def _batched_chunks(chunks: List[Chunk], batch_size: int) -> Iterator[List[Chunk]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    for start in range(0, len(chunks), batch_size):
        yield chunks[start:start + batch_size]


def vector_store_exists(persist_dir: str) -> bool:
    path = Path(persist_dir)
    return all(
        [
            (path / "index.faiss").exists(),
            (path / "chunks.jsonl").exists(),
            (path / "store_config.json").exists(),
        ]
    )


ProgressCallback = Callable[[float, Optional[str]], None]


def ingest_documents(
    data_dir: str,
    persist_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    table_row_overlap: int = 1,
    embedding_batch_size: int = 64,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    if embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be greater than zero")

    pdf_files = sorted(data_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {data_dir}")

    vector_embedder = VectorEmbedder()
    vector_store: FAISSStore | None = None
    embedding_dimension: int | None = None
    total_chunk_count = 0
    total_element_count = 0

    logger.info("Processing %s PDF files from %s", len(pdf_files), data_dir)

    total_files = len(pdf_files)

    def notify(progress: float, message: Optional[str] = None) -> None:
        if progress_callback:
            progress_callback(progress, message)

    for index, pdf_file in enumerate(pdf_files, start=1):
        base_progress = (index - 1) / total_files
        file_weight = 1 / total_files
        notify(base_progress, f"Parsing {pdf_file.name} ({index}/{total_files})")
        logger.info("Parsing %s", pdf_file.name)
        documents = PDFParser.load_structured_pdf(pdf_file)
        total_element_count += len(documents)

        if not documents:
            logger.warning("Skipping %s — parser returned no elements", pdf_file.name)
            notify(base_progress + file_weight, f"Skipped {pdf_file.name} (no elements)")
            continue

        notify(base_progress + file_weight * 0.2, f"Chunking {pdf_file.name}")
        chunks: List[Chunk] = Chunker.chunk_headings_with_paragraphs(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            table_row_overlap=table_row_overlap,
        )

        if not chunks:
            logger.warning("Skipping %s — chunker returned no chunks", pdf_file.name)
            notify(base_progress + file_weight, f"Skipped {pdf_file.name} (no chunks)")
            continue

        total_chunk_count += len(chunks)
        logger.info("Embedding %s chunks from %s", len(chunks), pdf_file.name)
        total_batches = max(1, math.ceil(len(chunks) / embedding_batch_size))

        for batch_index, batch in enumerate(_batched_chunks(chunks, embedding_batch_size), start=1):
            embed_fraction = batch_index / total_batches
            notify(
                base_progress + file_weight * (0.3 + 0.7 * embed_fraction),
                f"Embedding {pdf_file.name} ({batch_index}/{total_batches})",
            )
            dimension, chunk_vector_embed_dict = vector_embedder.embed_vectors_in_chunks(batch)
            if vector_store is None:
                vector_store = FAISSStore(dimensions=dimension, use_cosine_similarity=True)
                embedding_dimension = dimension
            elif vector_store.dimensions != dimension:
                raise ValueError("Embedding dimension changed between batches; aborting ingestion")

            vector_store.store_embeds_and_metadata(chunk_vector_embed_dict)

    if vector_store is None or embedding_dimension is None or total_chunk_count == 0:
        raise ValueError("No chunks were ingested; ensure the input PDFs contain parsable content")

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
        "embedding_batch_size": embedding_batch_size,
        "dimension": embedding_dimension,
        "chunk_count": total_chunk_count,
        "parsed_elements": total_element_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_path.resolve()),
    }
    _write_store_config(target_dir, store_config)

    logger.info(
        "Persisted vector store with %s chunks from %s parsed elements to %s",
        total_chunk_count,
        total_element_count,
        persist_dir,
    )

    return {
        "chunks": total_chunk_count,
        "documents": total_element_count,
        "pdf_files": len(pdf_files),
        "persist_dir": persist_dir,
        "store_config": store_config,
    }
