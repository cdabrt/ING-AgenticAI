from ..models import (
    RequirementItem as ContractRequirementItem,
    Source as ContractSource,
    RequirementType,
    SourceType
)
from AgenticAI.agentic.models import RequirementItem as ModelRequirementItem


def adapt_to_db_model(model_item: ModelRequirementItem, item_type: RequirementType) -> ContractRequirementItem:
    """
    Adapts a ModelRequirementItem (Pydantic) to ContractRequirementItem (SQLAlchemy)
    for database persistence.
    
    Args:
        model_item: The Pydantic model item to convert
        item_type: RequirementType.BUSINESS or RequirementType.DATA to specify the requirement type
    """
    contract_item = ContractRequirementItem()
    contract_item.req_id = model_item.id
    contract_item.description = model_item.description
    contract_item.rationale = model_item.rationale
    contract_item.type = item_type
    
    # Convert sources
    contract_item.sources = []
    
    for doc_source in model_item.document_sources:
        source = ContractSource()
        source.source_type = SourceType.DOCUMENT
        source.reference = doc_source
        contract_item.sources.append(source)
    
    for online_source in model_item.online_sources:
        source = ContractSource()
        source.source_type = SourceType.ONLINE
        source.reference = online_source
        contract_item.sources.append(source)
    
    return contract_item


def adapt_to_model(contract_item: ContractRequirementItem) -> ModelRequirementItem:
    """
    Adapts a ContractRequirementItem (SQLAlchemy) to ModelRequirementItem (Pydantic)
    for use in the application layer.
    """
    # Separate sources by type
    document_sources = []
    online_sources = []
    
    for source in contract_item.sources:
        if source.source_type == SourceType.DOCUMENT:
            document_sources.append(source.reference)
        elif source.source_type == SourceType.ONLINE:
            online_sources.append(source.reference)
    
    return ModelRequirementItem(
        id=contract_item.req_id,
        description=contract_item.description,
        rationale=contract_item.rationale,
        document_sources=document_sources,
        online_sources=online_sources
    )