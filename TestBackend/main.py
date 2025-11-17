from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequirementItem(BaseModel):
    id: str
    description: str
    rationale: str
    document_sources: List[str]
    online_sources: List[str]


class RequirementBundle(BaseModel):
    document: str
    document_type: str
    business_requirements: List[RequirementItem]
    data_requirements: List[RequirementItem]
    assumptions: List[str]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/api/pipeline")
def run_pipeline() -> RequirementBundle:
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

    return bundle