export interface RequirementItem {
    id: string;
    description: string;
    rationale: string;
    document_sources: string[];
    online_sources: string[];
    type: "BUSINESS" | "DATA";
}

export interface RequirementBundle {
    id: number;
    document: string;
    document_type: string;
    business_requirements: RequirementItem[];
    data_requirements: RequirementItem[];
    assumptions: string[];
}

export interface PDFItem {
    id: number;
    filename: string;
    upload_date: string;
    file_size?: number;
}
