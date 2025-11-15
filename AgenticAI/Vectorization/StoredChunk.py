from typing import Optional

from pydantic import BaseModel
from Chunker.Chunk import Chunk


class StoredChunk(BaseModel):
    chunk_id: str
    page_content: str
    source: str
    page: int
    type: str
    char_start: int
    char_end: int
    parent_heading: Optional[str] = None
    sparse_vector: Optional[dict[str, str]] = None
    dense_vector: Optional[list[float]] = None
    score: Optional[float] = None

    @classmethod
    def map_chunk(cls, chunk: Chunk) -> 'StoredChunk':
        return cls(
            chunk_id=chunk.chunk_id,
            page_content=chunk.document.page_content,
            source=chunk.document.meta_data.source,
            page=chunk.document.meta_data.page,
            type=chunk.document.meta_data.type,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            parent_heading=chunk.parent_heading
        )
