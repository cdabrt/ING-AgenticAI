import hashlib
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

from AgenticAI.Chunker.Chunk import Chunk
from AgenticAI.Chunker.Chunker import Chunker
from AgenticAI.PDF.PDFParser import PDFParser
from AgenticAI.Vectorization.VectorEmbedder import VectorEmbedder
from AgenticAI.Vectorization.vectorStore.FAISS import FAISSStore
from AgenticAI.Vectorization.vectorStore.store_factory import load_store_config, resolve_backend

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from AgenticAI.Vectorization.vectorStore.Milvus import MilvusStore


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
    config_path = path / "store_config.json"
    if not config_path.exists():
        return False

    config = load_store_config(persist_dir)
    backend = resolve_backend(config.get("vector_store"))
    if backend == "milvus":
        collection_name = config.get("collection_name") or os.getenv("MILVUS_COLLECTION", "agenticai_chunks")
        try:
            from AgenticAI.Vectorization.vectorStore.Milvus import MilvusStore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pymilvus is required for VECTOR_STORE_BACKEND=milvus. "
                "Install it with `pip install pymilvus`."
            ) from exc
        return MilvusStore.collection_exists(collection_name=collection_name)

    return all(
        [
            (path / "index.faiss").exists(),
            (path / "chunks.jsonl").exists(),
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
    vector_store_backend: Optional[str] = None,
    include_sources: Optional[List[str]] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict:
    backend = resolve_backend(vector_store_backend or os.getenv("VECTOR_STORE_BACKEND"))
    if backend == "milvus":
        return _ingest_documents_milvus(
            data_dir=data_dir,
            persist_dir=persist_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            table_row_overlap=table_row_overlap,
            embedding_batch_size=embedding_batch_size,
            include_sources=include_sources,
            progress_callback=progress_callback,
        )
    return _ingest_documents_faiss(
        data_dir=data_dir,
        persist_dir=persist_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        table_row_overlap=table_row_overlap,
        embedding_batch_size=embedding_batch_size,
        include_sources=include_sources,
        progress_callback=progress_callback,
    )


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_document_index(target_dir: Path) -> Dict[str, Dict[str, str]]:
    index_path = target_dir / "document_index.json"
    if not index_path.exists():
        return {}
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    return {item["source"]: item for item in payload if isinstance(item, dict) and "source" in item}


def _save_document_index(target_dir: Path, index: Dict[str, Dict[str, str]]) -> None:
    index_path = target_dir / "document_index.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)


def _ingest_documents_faiss(
    data_dir: str,
    persist_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    table_row_overlap: int = 1,
    embedding_batch_size: int = 64,
    include_sources: Optional[List[str]] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    if embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be greater than zero")

    pdf_files = sorted(data_path.glob("*.pdf"))
    if include_sources:
        include_set = {name.strip() for name in include_sources if name.strip()}
        pdf_files = [path for path in pdf_files if path.name in include_set]
        missing = include_set.difference({path.name for path in pdf_files})
        if missing:
            raise ValueError(f"Missing PDFs: {', '.join(sorted(missing))}")
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


def _ingest_documents_milvus(
    data_dir: str,
    persist_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    table_row_overlap: int = 1,
    embedding_batch_size: int = 64,
    include_sources: Optional[List[str]] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict:
    try:
        from AgenticAI.Vectorization.vectorStore.Milvus import MilvusStore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pymilvus is required for VECTOR_STORE_BACKEND=milvus. "
            "Install it with `pip install pymilvus`."
        ) from exc
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    if embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be greater than zero")

    pdf_files = sorted(data_path.glob("*.pdf"))
    include_set = None
    if include_sources:
        include_set = {name.strip() for name in include_sources if name.strip()}
        pdf_files = [path for path in pdf_files if path.name in include_set]
        missing = include_set.difference({path.name for path in pdf_files})
        if missing:
            raise ValueError(f"Missing PDFs: {', '.join(sorted(missing))}")
    if not pdf_files:
        raise ValueError(f"No PDF files found in {data_dir}")

    target_dir = Path(persist_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    document_index = _load_document_index(target_dir)
    existing_config = load_store_config(persist_dir)
    collection_name = existing_config.get("collection_name") or os.getenv("MILVUS_COLLECTION", "agenticai_chunks")

    current_sources: Dict[str, Dict[str, str]] = {}
    sources_to_ingest: List[Path] = []
    sources_to_delete: List[str] = []

    collection_exists = MilvusStore.collection_exists(collection_name=collection_name)
    if not collection_exists:
        document_index = {}

    for pdf_file in pdf_files:
        file_hash = _hash_file(pdf_file)
        current_sources[pdf_file.name] = {"file_hash": file_hash}
        existing = document_index.get(pdf_file.name)
        if not existing or existing.get("file_hash") != file_hash or not collection_exists:
            sources_to_ingest.append(pdf_file)
            if existing:
                sources_to_delete.append(pdf_file.name)

    removed_sources = []
    if include_set is None:
        removed_sources = [source for source in document_index.keys() if source not in current_sources]
    sources_to_delete.extend(removed_sources)

    existing_dimension = existing_config.get("dimension") if existing_config else None
    previous_parsed_elements = existing_config.get("parsed_elements", 0) if existing_config else 0

    vector_embedder = VectorEmbedder()
    vector_store: Optional["MilvusStore"] = None
    total_chunk_count = 0
    total_element_count = 0

    if collection_exists:
        vector_store = MilvusStore.from_env(
            dimensions=existing_dimension,
            use_cosine_similarity=True,
            collection_name=collection_name,
            auto_create=False,
        )
        if sources_to_delete:
            vector_store.delete_by_sources(sources_to_delete)
            sources_to_delete = []

    def notify(progress: float, message: Optional[str] = None) -> None:
        if progress_callback:
            progress_callback(progress, message)

    for index, pdf_file in enumerate(sources_to_ingest, start=1):
        base_progress = (index - 1) / max(1, len(sources_to_ingest))
        file_weight = 1 / max(1, len(sources_to_ingest))
        notify(base_progress, f"Parsing {pdf_file.name} ({index}/{len(sources_to_ingest)})")
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
                vector_store = MilvusStore.from_env(
                    dimensions=dimension,
                    use_cosine_similarity=True,
                    collection_name=collection_name,
                    auto_create=True,
                )
                if sources_to_delete:
                    vector_store.delete_by_sources(sources_to_delete)
                    sources_to_delete = []
            elif vector_store.dimensions != dimension:
                raise ValueError("Embedding dimension changed between batches; aborting ingestion")

            vector_store.store_embeds_and_metadata(chunk_vector_embed_dict)

        document_index[pdf_file.name] = {
            "source": pdf_file.name,
            "file_hash": current_sources[pdf_file.name]["file_hash"],
            "chunk_count": len(chunks),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    for source in removed_sources:
        document_index.pop(source, None)

    _save_document_index(target_dir, document_index)

    if vector_store is None:
        raise ValueError("No chunks were ingested; ensure the input PDFs contain parsable content")

    if include_set is None:
        parsed_elements = total_element_count or previous_parsed_elements
    else:
        parsed_elements = previous_parsed_elements or total_element_count
    store_config = {
        "vector_store": "milvus",
        "collection_name": collection_name,
        "use_cosine_similarity": True,
        "embedding_model_name": vector_embedder.model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "table_row_overlap": table_row_overlap,
        "embedding_batch_size": embedding_batch_size,
        "dimension": vector_store.dimensions,
        "chunk_count": vector_store.count(),
        "parsed_elements": parsed_elements,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_path.resolve()),
        "document_index": str((target_dir / "document_index.json").resolve()),
    }
    _write_store_config(target_dir, store_config)

    logger.info(
        "Persisted Milvus vector store with %s chunks across %s documents",
        store_config["chunk_count"],
        len(document_index),
    )

    return {
        "chunks": store_config["chunk_count"],
        "documents": total_element_count,
        "pdf_files": len(pdf_files),
        "persist_dir": persist_dir,
        "store_config": store_config,
    }
