from ..models import (
    RequirementBundle as ContractRequirementBundle,
    Assumption as ContractAssumption,
    RequirementType
)
from AgenticAI.agentic.models import RequirementBundle as ModelRequirementBundle

from .requirement_item_adapter import adapt_to_model as _adapt_item_to_model
from .requirement_item_adapter import adapt_to_db_model as _adapt_item_to_db_model

def adapt_to_db_model(model_bundle: ModelRequirementBundle) -> ContractRequirementBundle:
    """
    Adapts a ModelRequirementBundle (Pydantic) to ContractRequirementBundle (SQLAlchemy)
    for database persistence.
    """
    # Create the contract bundle (SQLAlchemy model)
    contract_bundle = ContractRequirementBundle()
    contract_bundle.document = model_bundle.document
    contract_bundle.document_type = model_bundle.document_type
    
    # Convert requirements (both business and data)
    contract_bundle.requirements = []
    
    # Add business requirements
    for req in model_bundle.business_requirements:
        contract_bundle.requirements.append(_adapt_item_to_db_model(req))
    
    # Add data requirements
    for req in model_bundle.data_requirements:
        contract_bundle.requirements.append(_adapt_item_to_db_model(req))
    
    # Convert assumptions
    contract_bundle.assumptions = []
    for assumption_text in model_bundle.assumptions:
        assumption = ContractAssumption()
        assumption.text = assumption_text
        contract_bundle.assumptions.append(assumption)
    
    return contract_bundle


def adapt_to_model(contract_bundle: ContractRequirementBundle) -> ModelRequirementBundle:
    """
    Adapts a ContractRequirementBundle (SQLAlchemy) to ModelRequirementBundle (Pydantic)
    for use in the application layer.
    """
    # Separate requirements by type
    business_requirements = []
    data_requirements = []
    
    for contract_req in contract_bundle.requirements:
        model_req = _adapt_item_to_model(contract_req)
        # Add to appropriate list based on type
        if contract_req.type == RequirementType.BUSINESS:
            business_requirements.append(model_req)
        elif contract_req.type == RequirementType.DATA:
            data_requirements.append(model_req)
    
    # Extract assumption texts
    assumptions = [assumption.text for assumption in contract_bundle.assumptions]
    
    # Create and return the model bundle
    return ModelRequirementBundle(
        document=contract_bundle.document,
        document_type=contract_bundle.document_type,
        business_requirements=business_requirements,
        data_requirements=data_requirements,
        assumptions=assumptions
    )