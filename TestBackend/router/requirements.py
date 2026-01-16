from fastapi import APIRouter, Depends, HTTPException
from AgenticAI.agentic.models import RequirementItem
from repository.requirement_repository import RequirementRepository

router = APIRouter(prefix="/requirements")

# GET
# -------------------------------------------

# POST
# -------------------------------------------

# PUT
# -------------------------------------------

@router.put("", tags=["Requirement"], response_model=RequirementItem)
def create_requirement_item(
    requirement: RequirementItem,
    client: RequirementRepository = Depends(RequirementRepository)
) -> RequirementItem:
    saved_item = client.save_requirement_item(requirement)
    if saved_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    
    return saved_item

# DELETE
# -------------------------------------------