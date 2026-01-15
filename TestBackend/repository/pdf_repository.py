from typing import List, Optional
from database.database_client import DatabaseClient
from AgenticAI.agentic.models import PDFDocument as PDFDocumentLocalModel
from database.models.models import PDFDocument as PDFDocumentDatabaseModel

class PDFRepository(DatabaseClient):

    def save_pdf_document(self, pdf_document: PDFDocumentLocalModel) -> PDFDocumentLocalModel:
        """Save a PDF document to the database"""
        return self._db.save(pdf_document)
    
    def get_pdf_document(self, pdf_id: int) -> Optional[PDFDocumentLocalModel]:
        """Get a PDF document by ID"""
        return self._db.get_first(PDFDocumentDatabaseModel, id=pdf_id)
    
    def get_all_pdf_documents(self) -> List[PDFDocumentLocalModel]:
        """Get all PDF documents"""
        return self._db.get_all(PDFDocumentDatabaseModel)
    
    def delete_pdf_document(self, pdf_id: int) -> bool:
        """Delete a PDF document by ID"""
        pdf_doc = self.get_pdf_document(pdf_id)
        if pdf_doc:
            self._db.delete(pdf_doc)
            return True
        return False
    
    def get_pdf_by_filename(self, filename: str) -> Optional[PDFDocumentLocalModel]:
        """Get a PDF document by filename"""
        return self._db.get_first(PDFDocumentDatabaseModel, filename=filename)