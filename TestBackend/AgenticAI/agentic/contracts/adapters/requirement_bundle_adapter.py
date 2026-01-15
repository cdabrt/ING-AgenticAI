
from .contracts import (
    RequirementBundle as ContractRequirementBundle
)
from AgenticAI.agentic.models import (
    RequirementBundle as ModelRequirementBundle
)
from .requirement_item_adapter import requirement_item_to_contract, requirement_item_to_model

def requirement_bundle_to_model(contract: ContractRequirementBundle) -> ModelRequirementBundle:
    """
    Convert a contract RequirementBundle (API layer) to a domain RequirementBundle (business logic).
    
    Args:
        contract: The contract RequirementBundle to convert
        
    Returns:
        Model RequirementBundle
    """
    return ModelRequirementBundle(
        document=contract.document,
        document_type=contract.document_type,
        business_requirements=[
            requirement_item_to_model(req) for req in contract.business_requirements
        ],
        data_requirements=[
            requirement_item_to_model(req) for req in contract.data_requirements
        ],
        assumptions=contract.assumptions
    )


def requirement_bundle_to_contract(domain: ModelRequirementBundle) -> ContractRequirementBundle:
    """
    Convert a domain RequirementBundle (business logic) to a contract RequirementBundle (API layer).
    
    Args:
        domain: The domain RequirementBundle to convert
        
    Returns:
        Contract RequirementBundle
    """
    return ContractRequirementBundle(
        document=domain.document,
        document_type=domain.document_type,
        business_requirements=[
            requirement_item_to_contract(req) for req in domain.business_requirements
        ],
        data_requirements=[
            requirement_item_to_contract(req) for req in domain.data_requirements
        ],
        assumptions=domain.assumptions
    )