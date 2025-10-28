from dataclasses import dataclass
from typing import Optional
from AgenticAI.PDF.Document import Document


@dataclass
class Chunk:
    chunk_id: str
    document: Document
    char_start: int
    char_end: int
    parent_heading: Optional[str] = None