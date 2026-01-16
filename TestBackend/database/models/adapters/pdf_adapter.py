from AgenticAI.agentic.models import PDFDocument as PDFDocumentLocalModel, Source as SourceLocalModel
from database.models.models import PDFDocument as PDFDocumentDatabaseModel, Source as SourceDatabaseModel, SourceType

def adapt_to_db_model(model: PDFDocumentLocalModel) -> PDFDocumentDatabaseModel:
    return PDFDocumentDatabaseModel(
        id=model.id,
        filename=model.filename,
        pdf_data=model.pdf_data
    )


def adapt_to_model(db_model: PDFDocumentDatabaseModel) -> PDFDocumentLocalModel:
    return PDFDocumentLocalModel(
        id=db_model.id,
        filename=db_model.filename,
        pdf_data=db_model.pdf_data,
        sources=[
            SourceLocalModel(
                id=source.id,
                source_type=source.source_type.value,
                reference=source.reference
            )
            for source in db_model.sources
        ]
    )