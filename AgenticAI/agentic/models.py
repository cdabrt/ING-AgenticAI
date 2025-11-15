from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class QuerySpecification(BaseModel):
    document_type: str = Field(..., description="Human readable classification of the document")
    summary: str = Field(..., description="Short summary of regulatory focus")
    queries: List[str] = Field(..., description="Targeted retrieval queries")


class RetrievalChunk(BaseModel):
    chunk_id: str
    score: float
    page: int
    source: str
    parent_heading: Optional[str]
    text: str


class WebResult(BaseModel):
    title: Optional[str]
    snippet: Optional[str]
    href: Optional[str]


class ContextAssessment(BaseModel):
    needs_additional_context: bool
    missing_information_queries: List[str]
    explanation: str


class RequirementItem(BaseModel):
    id: str
    description: str
    rationale: str
    document_sources: List[str]
    online_sources: List[str]


class RequirementBundle(BaseModel):
    document: str
    document_type: str
    business_requirements: List[RequirementItem]
    data_requirements: List[RequirementItem]
    assumptions: List[str]