from typing import List, Optional
from database.database_provider import DatabaseProvider
from contracts.models import PDFDocument

class PDFProvider(DatabaseProvider):

    def save_pdf_document(self, filename: str, pdf_data: bytes) -> PDFDocument:
        """Save a PDF document to the database"""
        pdf_doc = PDFDocument(
            filename=filename,
            pdf_data=pdf_data
        )
        self._db.db.add(pdf_doc)
        self._db.commit()
        self._db.db.refresh(pdf_doc)
        return pdf_doc
    
    def get_pdf_document(self, pdf_id: int) -> Optional[PDFDocument]:
        """Get a PDF document by ID"""
        return self._db.db.query(PDFDocument).filter(PDFDocument.id == pdf_id).first()
    
    def get_all_pdf_documents(self) -> List[PDFDocument]:
        """Get all PDF documents"""
        return self._db.db.query(PDFDocument).all()
    
    def delete_pdf_document(self, pdf_id: int) -> bool:
        """Delete a PDF document by ID"""
        pdf_doc = self.get_pdf_document(pdf_id)
        if pdf_doc:
            self._db.db.delete(pdf_doc)
            self._db.commit()
            return True
        return False
    
    def get_pdf_by_filename(self, filename: str) -> Optional[PDFDocument]:
        """Get a PDF document by filename"""
        return self._db.db.query(PDFDocument).filter(PDFDocument.filename == filename).first()
