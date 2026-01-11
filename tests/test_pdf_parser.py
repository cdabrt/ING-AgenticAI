from pathlib import Path

from AgenticAI.PDF.PDFParser import PDFParser
from AgenticAI.PDF.Document import ElementType


def _word(
    text: str,
    *,
    top: float = 10.0,
    fontname: str = "Helvetica",
    size: float = 12.0,
):
    return {"text": text, "top": top, "fontname": fontname, "size": size}


def test_analyze_page_fonts_uses_median_as_body_size():
    words = [
        _word("small", size=10),
        _word("body", size=12),
        _word("large", size=14),
    ]
    stats = PDFParser._analyze_page_fonts(words)

    assert stats["body_size"] == 12
    assert stats["max_size"] == 14
    assert stats["min_size"] == 10


def test_group_into_lines_groups_by_vertical_position():
    words = [
        _word("Hello", top=10),
        _word("World", top=11),
        _word("Next", top=30),
    ]
    lines = PDFParser._group_into_lines(words)

    assert len(lines) == 2
    assert [w["text"] for w in lines[0]] == ["Hello", "World"]


def test_extract_headings_and_paragraphs_detects_heading_lines():
    pdf_file = Path("dummy.pdf")
    words = [
        _word("Section", top=10, fontname="Helvetica-Bold", size=14),
        _word("1", top=10, fontname="Helvetica-Bold", size=14),
        _word("This", top=30, size=10),
        _word("is", top=30, size=10),
        _word("body.", top=30, size=10),
    ]

    elements = PDFParser._extract_headings_and_paragraphs(words, pdf_file, 1)
    element_types = [element.meta_data.type for element in elements]

    assert ElementType.HEADING in element_types
    assert ElementType.PARAGRAPH in element_types
    assert any(element.page_content == "Section 1" for element in elements)
