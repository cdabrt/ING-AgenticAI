from fastapi import APIRouter
from fastapi import Depends, FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from repository.pdf_repository import PDFRepository

router = APIRouter(prefix="/pdfs")

# GET
# -------------------------------------------

@router.get("/")
def get_all_pdfs(pdf_repository: PDFRepository = Depends(PDFRepository)):
    """Get all uploaded PDF documents (metadata only, not the binary data)"""
    pdfs = pdf_repository.get_all_pdf_documents()
    return [
        {
            "id": pdf.id,
            "filename": pdf.filename,
            "size": len(pdf.pdf_data)
        }
        for pdf in pdfs
    ]

@router.get("/{pdf_id}")
def get_pdf(pdf_id: int, pdf_repository: PDFRepository = Depends(PDFRepository)):
    """Get a specific PDF document by ID"""
    pdf = pdf_repository.get_pdf_document(pdf_id)
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    return {
        "id": pdf.id,
        "filename": pdf.filename,
        "size": len(pdf.pdf_data)
    }

@router.get("/{pdf_id}/download")
def download_pdf(pdf_id: int, pdf_repository: PDFRepository = Depends(PDFRepository)):
    """Download a specific PDF document by ID"""
    pdf = pdf_repository.get_pdf_document(pdf_id)
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    return Response(
        content=pdf.pdf_data,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{pdf.filename}"'
        }
    )

# POST
# -------------------------------------------

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    pdf_repository: PDFRepository = Depends(PDFRepository)
):
    """
    Upload a single PDF file and save it to the database.
    Only PDF files are accepted.
    """
    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must have a .pdf extension")
    
    # Read file content
    contents = await file.read()
    
    # Check if file with same name already exists
    existing_pdf = pdf_repository.get_pdf_by_filename(file.filename)
    if existing_pdf:
        raise HTTPException(
            status_code=409, 
            detail=f"A PDF with filename '{file.filename}' already exists"
        )
    
    # Save to database
    saved_pdf = pdf_repository.save_pdf_document(
        filename=file.filename,
        pdf_data=contents
    )
    
    return {
        "id": saved_pdf.id,
        "filename": saved_pdf.filename,
        "size": len(contents),
        "message": "File uploaded successfully to database"
    }

# DELETE
# -------------------------------------------

@router.delete("/{pdf_id}")
def delete_pdf(pdf_id: int, pdf_repository: PDFRepository = Depends(PDFRepository)):
    """Delete a PDF document by ID"""
    success = pdf_repository.delete_pdf_document(pdf_id)
    if not success:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    return {"message": "PDF deleted successfully"}
