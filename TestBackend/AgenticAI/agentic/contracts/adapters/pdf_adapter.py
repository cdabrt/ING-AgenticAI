from ..contracts import (
    PDFDocument as ContractPDFDocument
)
from AgenticAI.agentic.models import (
    PDFDocument as ModelPDFDocument
)

def pdf_document_to_model(contract: ContractPDFDocument) -> ModelPDFDocument:
    """
    Convert a contract PDFDocument (API layer) to a domain PDFDocument (business logic).
    
    Args:
        contract: The contract PDFDocument to convert
        
    Returns:
        Model PDFDocument
    """
    return ModelPDFDocument(
        id=contract.id,
        filename=contract.filename,
        pdf_data=b"",
        sources=[] 
    )


def pdf_document_to_contract(domain: ModelPDFDocument) -> ContractPDFDocument:
    """
    Convert a domain PDFDocument (business logic) to a contract PDFDocument (API layer).
    
    Args:
        domain: The domain PDFDocument to convert
        
    Returns:
        Contract PDFDocument
    """
    return ContractPDFDocument(
        id=domain.id,
        filename=domain.filename
    )
