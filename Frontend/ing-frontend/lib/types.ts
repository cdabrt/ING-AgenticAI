export interface RequirementItem {
    id: string;
    description: string;
    rationale: string;
    document_sources: string[];
    online_sources: string[];
}

export interface RequirementBundle {
    document: string;
    document_type: string;
    business_requirements: RequirementItem[];
    data_requirements: RequirementItem[];
    assumptions: string[];
}
