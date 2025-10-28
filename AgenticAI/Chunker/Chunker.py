import re
import uuid
from typing import List, Optional
from AgenticAI.Chunker.Chunk import Chunk
from AgenticAI.PDF.Document import ElementType, Document, Metadata


# Chunks multiple paragraphs belonging to a header together.
class Chunker:

    # Splits text into rough sentence blocks using punctuation
    @staticmethod
    def split_sentence(text: str) -> list[str]:
        # ., ?, ! followed by whitespace or line break
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s]

    # Build the chunks
    @staticmethod
    def chunk_paragraph(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        sentences = Chunker.split_sentence(text)
        chunks, current = [], ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= chunk_size:
                current += (" " if current else "") + sentence
            else:
                chunks.append(current.strip())
                # Ensure we don't take more than current length for overlap
                safe_overlap = min(len(current), chunk_overlap)
                tail = current[-safe_overlap:] if safe_overlap > 0 else ""
                current = (tail + " " + sentence).strip()
        if current.strip():
            chunks.append(current.strip())

        return chunks

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
            current_group: List[str],
            chunk_size: int,
            chunk_overlap: int,
            chunks: List[Chunk],
            table_row_overlap: int,
            current_meta: Optional[Metadata],
            parent_heading: Optional[str] = None,
    ):
        if not current_group or not current_meta:
            return

        if current_meta.type == ElementType.TABLE:
            for element in current_group:
                table_chunks = Chunker.chunk_table(element, chunk_size=chunk_size, row_overlap=table_row_overlap)
                for sub in table_chunks:
                    chunks.append(Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document=Document(
                            page_content=sub,
                            meta_data=current_meta
                        ),
                        char_start=0,
                        char_end=len(sub),
                        parent_heading=parent_heading
                    ))
        else:
            # Merge paragraphs under the same header until chunk_size is reached.
            # Overlap is applied only within the same header.
            current_chunk_text = ""
            for para in current_group:
                para = para.strip()
                if len(current_chunk_text) + len(para) + 1 <= chunk_size:
                    current_chunk_text += (" " if current_chunk_text else "") + para
                else:
                    # finalize current chunk
                    chunks.append(Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document=Document(
                            page_content=current_chunk_text,
                            meta_data=current_meta
                        ),
                        char_start=0,
                        char_end=len(current_chunk_text),
                        parent_heading=parent_heading
                    ))
                    # start new chunk with overlap
                    overlap_text = current_chunk_text[-chunk_overlap:] if chunk_overlap > 0 else ""
                    current_chunk_text = (overlap_text + " " + para).strip()

            if current_chunk_text:
                # final chunk for this header, no extra overlap
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document=Document(
                        page_content=current_chunk_text,
                        meta_data=current_meta
                    ),
                    char_start=0,
                    char_end=len(current_chunk_text),
                    parent_heading=parent_heading
                ))

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
        current_group: List[str] = []
        current_meta: Optional[Metadata] = None

        for doc in documents:
            meta = doc.meta_data
            text = doc.page_content.strip()

            if meta.type == ElementType.HEADING:
                # finalize the previous header group if any
                if current_group:
                    Chunker.chunk_header_group(
                        current_group=current_group,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        chunks=chunks,
                        table_row_overlap=table_row_overlap,
                        current_meta=current_meta,
                        parent_heading=current_heading
                    )

                # start new header group
                current_heading = text
                current_group = []
                current_meta = None
            else:
                # assign metadata for first element in the header group
                if current_meta is None:
                    current_meta = meta
                current_group.append(text)

        # finalize the last header group
        if current_group:
            Chunker.chunk_header_group(
                current_group=current_group,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunks=chunks,
                table_row_overlap=table_row_overlap,
                current_meta=current_meta,
                parent_heading=current_heading
            )

        return chunks
