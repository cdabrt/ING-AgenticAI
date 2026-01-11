from typing import List

from pydantic import BaseModel


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