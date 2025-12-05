from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from fpdf import FPDF

RequirementRecord = Mapping[str, Any]


class RequirementsPDF(FPDF):
    """Thin wrapper to centralize default fonts and spacing."""

    def __init__(self) -> None:  # noqa: D401 - single-line description already provided
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)


def render_requirements_pdf(records: Sequence[RequirementRecord], output_path: Path | str) -> Path:
    """Render requirement bundles into a readable PDF summary."""

    if not records:
        raise ValueError("No requirement bundles were supplied for PDF rendering")

    pdf = RequirementsPDF()
    for index, record in enumerate(records, start=1):
        pdf.add_page()
        _write_document_header(pdf, record, index)
        _write_requirement_section(pdf, "Business Requirements", record.get("business_requirements", []))
        _write_requirement_section(pdf, "Data Requirements", record.get("data_requirements", []))
        _write_assumptions(pdf, record.get("assumptions", []))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    return output_path


def _write_document_header(pdf: FPDF, record: RequirementRecord, index: int) -> None:
    doc_name = str(record.get("document", "Unknown document"))
    doc_type = str(record.get("document_type", "Unknown type"))

    pdf.set_font("Helvetica", "B", 16)
    _full_width_multicell(pdf, doc_name, height=8)
    pdf.ln(1)

    pdf.set_font("Helvetica", "", 12)
    _full_width_multicell(pdf, f"Document type: {doc_type}", height=6)
    _full_width_multicell(pdf, f"Bundle #{index}", height=6)
    pdf.ln(4)


def _write_requirement_section(pdf: FPDF, title: str, items: Sequence[Mapping[str, Any]]) -> None:
    pdf.set_font("Helvetica", "B", 14)
    _full_width_multicell(pdf, title, height=7)
    pdf.ln(1)

    if not items:
        pdf.set_font("Helvetica", "", 11)
        _full_width_multicell(pdf, "- None provided", height=6)
        pdf.ln(2)
        return

    for item in items:
        _write_requirement_item(pdf, item)
        pdf.ln(2)


def _write_requirement_item(pdf: FPDF, item: Mapping[str, Any]) -> None:
    req_id = str(item.get("id", "ID-unknown"))
    description = str(item.get("description", "No description supplied"))
    rationale = str(item.get("rationale", "No rationale supplied"))
    document_sources = item.get("document_sources", []) or []
    online_sources = item.get("online_sources", []) or []

    pdf.set_font("Helvetica", "B", 12)
    _write_indented_line(pdf, f"- {req_id}: {description}", indent=0)

    pdf.set_font("Helvetica", "", 11)
    _write_indented_line(pdf, f"Rationale: {rationale}", indent=4)

    _write_source_list(pdf, "Document sources", document_sources, indent=4)
    _write_source_list(pdf, "Online sources", online_sources, indent=4)


def _write_source_list(pdf: FPDF, label: str, values: Sequence[str], indent: int) -> None:
    pdf.set_font("Helvetica", "", 11)
    if not values:
        _write_indented_line(pdf, f"{label}: None provided", indent=indent)
        return

    _write_indented_line(pdf, f"{label}:", indent=indent)
    for value in values:
        _write_indented_line(pdf, f"- {value}", indent=indent + 4)


def _write_assumptions(pdf: FPDF, assumptions: Sequence[str]) -> None:
    pdf.set_font("Helvetica", "B", 14)
    _full_width_multicell(pdf, "Assumptions", height=7)
    pdf.ln(1)

    pdf.set_font("Helvetica", "", 11)
    if not assumptions:
        _full_width_multicell(pdf, "- None documented", height=6)
        return

    for assumption in assumptions:
        _write_indented_line(pdf, f"- {assumption}", indent=0)


def _write_indented_line(pdf: FPDF, text: str, *, indent: int) -> None:
    indent = max(indent, 0)
    x_position = pdf.l_margin + indent
    pdf.set_x(x_position)
    pdf.multi_cell(0, 5, text)


def _full_width_multicell(pdf: FPDF, text: str, *, height: float) -> None:
    available_width = max(pdf.w - pdf.l_margin - pdf.r_margin, 10)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(available_width, height, text)
