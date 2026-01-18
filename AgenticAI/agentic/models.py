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
    content: Optional[str]
    selection_reason: Optional[str]
    inclusion_reason: Optional[str]
    summary: Optional[str]


class WebSourceCandidate(BaseModel):
    identifier: str = Field(..., description="Internal reference for the candidate result")
    title: Optional[str]
    snippet: Optional[str]
    href: Optional[str]


class WebSourceSelection(BaseModel):
    identifier: str = Field(..., description="Candidate identifier being evaluated")
    fetch: bool = Field(..., description="True if the agent wants to load the page content")
    rationale: str = Field(..., description="Why this decision was taken")


class WebSelectionResponse(BaseModel):
    selections: List[WebSourceSelection]


class WebContentDecision(BaseModel):
    include: bool = Field(..., description="Whether the cleaned page should augment context")
    rationale: str = Field(..., description="Justification for including or dropping the page")
    summary: str = Field(..., description="Two-sentence synopsis highlighting the relevant insight")


class ContextAssessment(BaseModel):
    needs_additional_context: bool
    missing_information_queries: List[str]
    explanation: str


class EvidenceCard(BaseModel):
    origin: str = Field(..., description="document or web")
    claim: str = Field(..., description="Atomic obligation or fact extracted from evidence")
    source: str = Field(..., description="Document name or URL")
    page: Optional[int] = Field(None, description="Page number for document evidence")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier for document evidence")
    certainty: Optional[str] = Field(None, description="High/Medium/Low confidence tag")
    support: Optional[str] = Field(None, description="Short excerpt supporting the claim")


class EvidenceDigest(BaseModel):
    cards: List[EvidenceCard]


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
