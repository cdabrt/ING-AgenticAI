from typing import List, Optional
from sqlalchemy.orm import defer
from database.database_client import DatabaseClient
from AgenticAI.agentic.models import PDFDocument as PDFDocumentLocalModel
from database.models.models import PDFDocument as PDFDocumentDatabaseModel
from database.models.adapters.pdf_adapter import adapt_to_db_model, adapt_to_model

class PDFRepository(DatabaseClient):

    def save_pdf_document(self, pdf_document: PDFDocumentLocalModel) -> PDFDocumentLocalModel:
        """Save a PDF document to the database"""
        db_doc = adapt_to_db_model(pdf_document)
        saved_db_doc = self._db.save(db_doc)
        return adapt_to_model(saved_db_doc)
    
    def get_pdf_document(self, pdf_id: int) -> Optional[PDFDocumentLocalModel]:
        """Get a PDF document by ID (includes pdf_data)"""
        db_doc = self._db.get_first(PDFDocumentDatabaseModel, id=pdf_id)
        return adapt_to_model(db_doc) if db_doc else None
    
    def get_all_pdf_documents_deferred(self) -> List[PDFDocumentLocalModel]:
        """Get all PDF documents (metadata only, deferring pdf_data)
        """
        db_docs = self._db.get_all_deferred(PDFDocumentDatabaseModel, defer(PDFDocumentDatabaseModel.pdf_data))
        return [adapt_to_model(db_doc) for db_doc in db_docs]
    
    def delete_pdf_document(self, pdf_id: int) -> bool:
        """Delete a PDF document by ID"""
        pdf_doc = self.get_pdf_document(pdf_id)
        if pdf_doc:
            self._db.delete(pdf_doc)
            return True
        return False
    
    def get_pdf_by_filename(self, filename: str) -> Optional[PDFDocumentLocalModel]:
        """Get a PDF document by filename"""
        db_doc = self._db.get_first(PDFDocumentDatabaseModel, filename=filename)
        return adapt_to_model(db_doc) if db_doc else None