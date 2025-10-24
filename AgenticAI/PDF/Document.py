from dataclasses import dataclass
from enum import Enum, auto

class ElementType(Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"

@dataclass
class Metadata:
    def __init__(self, source : str, page : int, type : ElementType):
        self.source = source
        self.page = page
        self.type = type

@dataclass
class Document:
    def __init__(self, page_content : str, meta_data : Metadata):
        self.page_content = page_content
        self.meta_data = meta_data