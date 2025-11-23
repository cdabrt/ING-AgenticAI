from typing import List
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from AgenticAI.agentic.models import RequirementBundle, RequirementItem

from database.bundle_provider import BundleProvider
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