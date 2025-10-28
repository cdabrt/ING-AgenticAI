import re
import uuid
from typing import List, Optional, Tuple
from AgenticAI.Chunker.Chunk import Chunk
from AgenticAI.PDF.Document import ElementType, Document, Metadata


class Chunker:

    # Splits text into rough sentence blocks using punctuation
    @staticmethod
    def split_sentence(text: str) -> list[str]:
        # ., ?, ! followed by whitespace or line break
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s]

    # Chunking a table is handled separately.
    # A row_overlap parameter is introduced to repeat some rows in the next chunk.
    @staticmethod
    def chunk_table(table_text: str, chunk_size: int, row_overlap: int) -> list[str]:
        rows = table_text.strip().split("\n")
        table_chunks = []

        i = 0
        while i < len(rows):
            current = []
            current_length = 0
            while i < len(rows) and current_length + len(rows[i]) <= chunk_size:
                current.append(rows[i].strip())
                # +1 for newline character
                current_length += len(rows[i]) + 1
                i += 1
            table_chunks.append("\n".join(current))
            i = max(i - row_overlap, 0)

        return table_chunks

    # Chunk all content under a single header
    @staticmethod
    def chunk_header_group(
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

        def flush_paragraph_buffer():
            nonlocal paragraph_buffer, paragraph_meta
            if not paragraph_buffer or paragraph_meta is None:
                paragraph_buffer = ""
                paragraph_meta = None
                return
            current_chunk_text = ""
            for para_text in paragraph_buffer.split("\n||PARA||\n"):
                para = para_text.strip()
                if len(current_chunk_text) + len(para) + 1 <= chunk_size:
                    current_chunk_text += (" " if current_chunk_text else "") + para
                else:
                    chunks.append(Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document=Document(
                            page_content=current_chunk_text,
                            meta_data=paragraph_meta
                        ),
                        char_start=0,
                        char_end=len(current_chunk_text),
                        parent_heading=parent_heading
                    ))
                    if chunk_overlap > 0:
                        words = current_chunk_text.split()
                        overlap_word_count = max(1, chunk_overlap // 6)
                        overlap_words = words[-overlap_word_count:]
                        overlap_text = " ".join(overlap_words)
                    else:
                        overlap_text = ""
                    current_chunk_text = (overlap_text + " " + para).strip()

            if current_chunk_text:
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document=Document(
                        page_content=current_chunk_text,
                        meta_data=paragraph_meta
                    ),
                    char_start=0,
                    char_end=len(current_chunk_text),
                    parent_heading=parent_heading
                ))

            paragraph_buffer = ""
            paragraph_meta = None

        for element_text, element_meta in current_group:
            if element_meta.type == ElementType.TABLE:
                flush_paragraph_buffer()
                table_chunks = Chunker.chunk_table(element_text, chunk_size=chunk_size, row_overlap=table_row_overlap)
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
            else:
                if paragraph_meta is None:
                    paragraph_meta = element_meta
                paragraph_buffer += ("\n||PARA||\n" if paragraph_buffer else "") + element_text

        flush_paragraph_buffer()

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

        for doc in documents:
            meta = doc.meta_data
            text = doc.page_content.strip()

            if meta.type == ElementType.HEADING:
                if current_group:
                    Chunker.chunk_header_group(
                        current_group=current_group,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        chunks=chunks,
                        table_row_overlap=table_row_overlap,
                        parent_heading=current_heading
                    )

                current_heading = text
                current_group = []
            else:
                current_group.append((text, meta))

        if current_group:
            Chunker.chunk_header_group(
                current_group=current_group,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunks=chunks,
                table_row_overlap=table_row_overlap,
                parent_heading=current_heading
            )

        return chunks
