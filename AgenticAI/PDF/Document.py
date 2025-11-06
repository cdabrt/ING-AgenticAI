from enum import Enum
from pydantic import BaseModel

class ElementType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"

class Metadata(BaseModel):
    source: str
    page: int
    type: ElementType

class Document(BaseModel):
    page_content: str
    meta_data: Metadata