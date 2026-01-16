
from ..contracts import (
    RequirementItem as ContractRequirementItem
)
from AgenticAI.agentic.models import (
    RequirementItem as ModelRequirementItem
)

def requirement_item_to_model(contract: ContractRequirementItem) -> ModelRequirementItem:
    """
    Convert a contract RequirementItem (API layer) to a domain RequirementItem (business logic).
    
    Args:
        contract: The contract RequirementItem to convert
        
    Returns:
        Model RequirementItem
    """
    return ModelRequirementItem(
        id=contract.id,
        description=contract.description,
        rationale=contract.rationale,
        document_sources=contract.document_sources,
        online_sources=contract.online_sources
    )


def requirement_item_to_contract(domain: ModelRequirementItem) -> ContractRequirementItem:
    """
    Convert a domain RequirementItem (business logic) to a contract RequirementItem (API layer).
    
    Args:
        domain: The domain RequirementItem to convert
        
    Returns:
        Contract RequirementItem
    """
    return ContractRequirementItem(
        id=domain.id,
        description=domain.description,
        rationale=domain.rationale,
        document_sources=domain.document_sources,
        online_sources=domain.online_sources,
        type=domain.type
    )