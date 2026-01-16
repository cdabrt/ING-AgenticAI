from typing import List
from fastapi import APIRouter, Depends
from AgenticAI.agentic.models import RequirementBundle, RequirementItem
from repository.bundle_repository import BundleRepository

router = APIRouter(prefix="/bundles")

# GET
# -------------------------------------------

@router.get("", tags=["Requirement Bundle"], response_model=List[RequirementBundle])
def get_all_requirement_bundles(client: BundleRepository = Depends(BundleRepository)) -> List[RequirementBundle]:
    bundles = client.get_requirement_bundles()
    return bundles

# POST
# -------------------------------------------

# TODO: Replace mock data with actual generation logic
@router.post("/generate", tags=["Requirement Bundle"], response_model=RequirementBundle)
def generate_bundle(client: BundleRepository = Depends(BundleRepository)) -> RequirementBundle:
    req1 = RequirementItem(
        id="REQ-001",
        description="The system shall encrypt all user data at rest using AES-256 encryption.",
        rationale="To ensure data security and compliance with regulations.",
        document_sources=["Document A - Section 3.2", "Document B - Page 5"],
        online_sources=["https://example.com/security-guidelines"],
        type="BUSINESS"
    )
    req2 = RequirementItem(
        id="REQ-002",
        description="The application shall support multi-factor authentication for all user accounts.",
        rationale="To enhance account security and prevent unauthorized access.",
        document_sources=["Document C - Section 4.1"],
        online_sources=["https://example.com/authentication-best-practices"],
        type="BUSINESS"
    )
    
    data_req1 = RequirementItem(
        id="DATA-001",
        description="User profile data must be stored in a relational database.",
        rationale="To maintain data integrity and support complex queries.",
        document_sources=["Document D - Section 2.1"],
        online_sources=["https://example.com/database-design"],
        type="DATA"
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
        ],
        type="DATA"

    )

    saved_bundle = client.save_requirement_bundle(bundle)
    return saved_bundle

# DELETE