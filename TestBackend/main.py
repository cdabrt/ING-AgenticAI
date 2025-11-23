from typing import List
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from AgenticAI.agentic.models import RequirementBundle, RequirementItem

from database.bundle_provider import BundleProvider
from database.pdf_provider import PDFProvider
from contracts.models import Base
from database.engine.psycopg_connection import create_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database tables
    engine = create_engine()
    # Drop and recreate tables to ensure schema is up to date
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized!")
    yield
    # Shutdown: cleanup if needed
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/api/pipeline")
def run_pipeline(client: BundleProvider = Depends(BundleProvider)) -> RequirementBundle:
    req1 = RequirementItem(
        id="REQ-001",
        description="The system shall encrypt all user data at rest using AES-256 encryption.",
        rationale="To ensure data security and compliance with regulations.",
        document_sources=["Document A - Section 3.2", "Document B - Page 5"],
        online_sources=["https://example.com/security-guidelines"],
    )
    req2 = RequirementItem(
        id="REQ-002",
        description="The application shall support multi-factor authentication for all user accounts.",
        rationale="To enhance account security and prevent unauthorized access.",
        document_sources=["Document C - Section 4.1"],
        online_sources=["https://example.com/authentication-best-practices"],
    )
    
    data_req1 = RequirementItem(
        id="DATA-001",
        description="User profile data must be stored in a relational database.",
        rationale="To maintain data integrity and support complex queries.",
        document_sources=["Document D - Section 2.1"],
        online_sources=["https://example.com/database-design"],
    )

    bundle = RequirementBundle(
        document="Mock Document",
        document_type="Security Requirements Specification",
        business_requirements=[req1, req2],
        data_requirements=[data_req1],
        assumptions=[
            "All users have access to email for MFA verification",
            "The system will be deployed on cloud infrastructure with encryption support",
            "Compliance requirements follow GDPR and SOC2 standards"
        ]
    )

    saved_bundle = client.save_requirement_bundle(bundle)
    return saved_bundle

@app.get("/api/bundles")
def get_all_requirement_bundles(client: BundleProvider = Depends(BundleProvider)) -> List[RequirementBundle]:

    bundles = client.get_requirement_bundles()
    return bundles

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    pdf_client: PDFProvider = Depends(PDFProvider)
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
    existing_pdf = pdf_client.get_pdf_by_filename(file.filename)
    if existing_pdf:
        raise HTTPException(
            status_code=409, 
            detail=f"A PDF with filename '{file.filename}' already exists"
        )
    
    # Save to database
    saved_pdf = pdf_client.save_pdf_document(
        filename=file.filename,
        pdf_data=contents
    )
    
    return {
        "id": saved_pdf.id,
        "filename": saved_pdf.filename,
        "size": len(contents),
        "message": "File uploaded successfully to database"
    }

@app.get("/api/pdfs")
def get_all_pdfs(pdf_client: PDFProvider = Depends(PDFProvider)):
    """Get all uploaded PDF documents (metadata only, not the binary data)"""
    pdfs = pdf_client.get_all_pdf_documents()
    return [
        {
            "id": pdf.id,
            "filename": pdf.filename,
            "size": len(pdf.pdf_data)
        }
        for pdf in pdfs
    ]

@app.get("/api/pdfs/{pdf_id}")
def get_pdf(pdf_id: int, pdf_client: PDFProvider = Depends(PDFProvider)):
    """Get a specific PDF document by ID"""
    pdf = pdf_client.get_pdf_document(pdf_id)
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    return {
        "id": pdf.id,
        "filename": pdf.filename,
        "size": len(pdf.pdf_data)
    }

@app.get("/api/pdfs/{pdf_id}/download")
def download_pdf(pdf_id: int, pdf_client: PDFProvider = Depends(PDFProvider)):
    """Download a specific PDF document by ID"""
    pdf = pdf_client.get_pdf_document(pdf_id)
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    return Response(
        content=pdf.pdf_data,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{pdf.filename}"'
        }
    )

@app.delete("/api/pdfs/{pdf_id}")
def delete_pdf(pdf_id: int, pdf_client: PDFProvider = Depends(PDFProvider)):
    """Delete a PDF document by ID"""
    success = pdf_client.delete_pdf_document(pdf_id)
    if not success:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    return {"message": "PDF deleted successfully"}
