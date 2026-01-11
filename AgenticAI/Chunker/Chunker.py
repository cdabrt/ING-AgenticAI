import logging
import re
import uuid
from typing import List, Optional, Tuple
from AgenticAI.Chunker.Chunk import Chunk
from AgenticAI.PDF.Document import ElementType, Document, Metadata

logger = logging.getLogger(__name__)

# Made static as the chunker is built from pure functions. No state or options have to be saved.
class Chunker:
    # Splits text into rough sentence blocks using punctuation
    @staticmethod
    def _split_sentence(text: str) -> list[str]:
        # ., ?, ! followed by whitespace or line break
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s]

    # Chunking a table is handled separately.
    # A row_overlap parameter is introduced to repeat some rows in the next chunk.
    @staticmethod
    def _chunk_table(table_text: str, chunk_size: int, row_overlap: int) -> list[str]:
        rows = table_text.strip().split("\n")
        table_chunks = []

        i = 0
        while i < len(rows):
            current = []
            current_length = 0
            start_index = i
            while i < len(rows) and current_length + len(rows[i]) <= chunk_size:
                current.append(rows[i].strip())
                # +1 for newline character
                current_length += len(rows[i]) + 1
                i += 1

            if not current:
                # Handle single rows that are larger than chunk_size by slicing
                oversized_row = rows[i].strip()
                for start in range(0, len(oversized_row), chunk_size):
                    table_chunks.append(oversized_row[start:start + chunk_size])
                i += 1
                continue

            table_chunks.append("\n".join(current))

            # Only overlap rows that were actually part of the chunk to avoid reprocessing
            consumed = i - start_index
            if consumed <= 0:
                break
            overlap = min(row_overlap, max(consumed - 1, 0))
            i -= overlap

        return table_chunks

    @staticmethod
    def _chunk_paragraph(
        paragraph_buffer: str,
        paragraph_meta: Optional[Metadata],
        chunk_size: int,
        chunk_overlap: int,
        chunks: List[Chunk],
        parent_heading: Optional[str]
    ) -> Tuple[str, Optional[Metadata]]:
        """Flush buffered paragraph text into fixed-size chunks.

        The previous implementation assumed that each paragraph was already smaller than
        the configured chunk size which is not true for many PDFs. Long paragraphs were
        therefore emitted as single giant chunks, causing embedding calls to blow up and
        stall the pipeline. The new implementation enforces the `chunk_size` limit by
        breaking paragraphs down into sentences (falling back to hard splits when a single
        sentence exceeds the limit) and emitting overlap windows in characters.
        """

        if not paragraph_buffer or paragraph_meta is None:
            return "", None

        paragraphs = [p.strip() for p in paragraph_buffer.split("\n||PARA||\n") if p.strip()]
        current_chunk_text = ""
        chunk_dirty = False

        max_chunk_size = max(200, chunk_size)  # guard against nonsensical config
        overlap_chars = max(0, min(chunk_overlap, max_chunk_size - 1))

        def _emit_chunk(force_clear: bool = False):
            nonlocal current_chunk_text, chunk_dirty
            chunk_text = current_chunk_text.strip()
            if not chunk_text or not chunk_dirty:
                current_chunk_text = "" if force_clear else chunk_text
                chunk_dirty = False if force_clear else chunk_dirty
                return

            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                document=Document(
                    page_content=chunk_text,
                    meta_data=paragraph_meta
                ),
                char_start=0,
                char_end=len(chunk_text),
                parent_heading=parent_heading
            ))

            chunk_dirty = False
            if force_clear or overlap_chars == 0:
                current_chunk_text = ""
                return

            current_chunk_text = chunk_text[-overlap_chars:]

        def _append_text(text: str):
            nonlocal current_chunk_text, chunk_dirty
            normalized = re.sub(r"\s+", " ", text).strip()
            if not normalized:
                return

            while normalized:
                available = max_chunk_size - len(current_chunk_text) - (1 if current_chunk_text else 0)
                if available <= 0:
                    _emit_chunk()
                    continue

                take = min(available, len(normalized))
                slice_text = normalized[:take]
                current_chunk_text = (current_chunk_text + (" " if current_chunk_text else "") + slice_text).strip()
                chunk_dirty = True
                normalized = normalized[take:]

                if len(current_chunk_text) >= max_chunk_size:
                    _emit_chunk()

        for paragraph in paragraphs:
            sentences = Chunker._split_sentence(paragraph) or [paragraph]
            for sentence in sentences:
                if len(sentence) <= max_chunk_size:
                    _append_text(sentence)
                else:
                    for start in range(0, len(sentence), max_chunk_size):
                        _append_text(sentence[start:start + max_chunk_size])

        if chunk_dirty and current_chunk_text.strip():
            _emit_chunk(force_clear=True)

        return "", None

    # Chunk all content under a single header
    @staticmethod
    def _chunk_header_group(
            current_group: List[Tuple[str, Metadata]],
            chunk_size: int,
            chunk_overlap: int,
            chunks: List[Chunk],
            table_row_overlap: int,
            parent_heading: Optional[str] = None,
    ):
        if not current_group:
            return

        paragraph_buffer = ""
        paragraph_meta: Optional[Metadata] = None

        logger.debug(
            "Chunking header group parent=%s elements=%s",
            parent_heading,
            len(current_group),
        )

        for element_text, element_meta in current_group:
            if element_meta.type == ElementType.TABLE:
                paragraph_buffer, paragraph_meta = Chunker._chunk_paragraph(
                    paragraph_buffer, paragraph_meta, chunk_size, chunk_overlap, chunks, parent_heading
                )
                table_chunks = Chunker._chunk_table(element_text, chunk_size=chunk_size, row_overlap=table_row_overlap)
                for sub in table_chunks:
                    chunks.append(Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document=Document(
                            page_content=sub,
                            meta_data=element_meta
                        ),
                        char_start=0,
                        char_end=len(sub),
                        parent_heading=parent_heading
                    ))
                logger.debug("Added %s table chunks under %s", len(table_chunks), parent_heading)
            else:
                if paragraph_meta is None:
                    paragraph_meta = element_meta
                paragraph_buffer += ("\n||PARA||\n" if paragraph_buffer else "") + element_text

        Chunker._chunk_paragraph(
            paragraph_buffer, paragraph_meta, chunk_size, chunk_overlap, chunks, parent_heading
        )
        logger.debug("Header group %s produced %s total chunks so far", parent_heading, len(chunks))

    # Processes the entire document, header by header
    @staticmethod
    def chunk_headings_with_paragraphs(
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        table_row_overlap: int = 1
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        current_heading: Optional[str] = None
        current_group: List[Tuple[str, Metadata]] = []

        logger.info("Starting chunking for %s parsed elements", len(documents))

        for doc in documents:
            meta = doc.meta_data
            text = doc.page_content.strip()

            if meta.type == ElementType.HEADING:
                if current_group:
                    Chunker._chunk_header_group(
                        current_group=current_group,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        chunks=chunks,
                        table_row_overlap=table_row_overlap,
                        parent_heading=current_heading
                    )

                current_heading = text
                current_group = []
                logger.debug("Encountered heading: %s", current_heading)
            else:
                current_group.append((text, meta))

        if current_group:
            Chunker._chunk_header_group(
                current_group=current_group,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunks=chunks,
                table_row_overlap=table_row_overlap,
                parent_heading=current_heading
            )

        logger.info("Chunking finished with %s chunks", len(chunks))

        return chunks
