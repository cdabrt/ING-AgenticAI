from typing import Optional
from PDF.Document import Document
from pydantic import BaseModel

class Chunk(BaseModel):
    chunk_id: str
    document: Document
    char_start: int
    char_end: int
    parent_heading: Optional[str] = None