from typing import List

from pydantic import BaseModel


class RequirementItem(BaseModel):
    id: str
    description: str
    rationale: str
    document_sources: List[str]
    online_sources: List[str]
    type: str
    # TODO Add this to the old requirement item


class RequirementBundle(BaseModel):
    document: str
    document_type: str
    business_requirements: List[RequirementItem]
    data_requirements: List[RequirementItem]
    assumptions: List[str]

class Source(BaseModel):
    id: int
    source_type: str
    reference: str

class PDFDocument(BaseModel):
    id: int
    filename: str
    pdf_data: bytes
    sources: List[Source]