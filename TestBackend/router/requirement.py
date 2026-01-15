from fastapi import APIRouter, Depends
from AgenticAI.agentic.models import RequirementItem
from TestBackend.repository.requirement_repository import RequirementRepository

router = APIRouter()

# GET
# -------------------------------------------

# POST
# -------------------------------------------

@router.post("/api/requirement", tags=["Requirement"], response_model=RequirementItem)
def create_requirement_item(
    requirement: RequirementItem,
    client: RequirementRepository = Depends(RequirementRepository)
) -> RequirementItem:
    saved_item = client.save_requirement_item(requirement)
    return saved_item

# DELETE
# -------------------------------------------