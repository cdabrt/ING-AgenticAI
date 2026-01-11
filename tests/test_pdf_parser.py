from pathlib import Path

from AgenticAI.PDF.PDFParser import PDFParser
from AgenticAI.PDF.Document import ElementType


def _word(text: str, top: float = 10.0, fontname: str = "Calibri"):
    return {"text": text, "top": top, "fontname": fontname}


def test_process_paragraph_auto_flushes_long_blocks():
    pdf_file = Path("dummy.pdf")
    current = []
    produced_docs = []

    for idx in range(400):
        token = f"token{idx}"
        if idx % 30 == 29:
            token += "."
        current, emitted = PDFParser._process_paragraph(
            word=_word(token),
            current_paragraph=current,
            pdf_file=pdf_file,
            page_num=1,
            current_type=ElementType.PARAGRAPH,
            is_new_block=False,
        )
        produced_docs.extend(emitted)

    assert produced_docs, "Expected the parser to emit at least one paragraph"
    assert all(
        len(doc.page_content) <= PDFParser.AUTO_PARAGRAPH_CHAR_LIMIT + 120
        for doc in produced_docs
    )


def test_process_paragraph_flushes_on_new_block():
    pdf_file = Path("dummy.pdf")
    current = ["Existing", "paragraph", "content."]

    current, emitted = PDFParser._process_paragraph(
        word=_word("Next", top=40.0),
        current_paragraph=current,
        pdf_file=pdf_file,
        page_num=2,
        current_type=ElementType.PARAGRAPH,
        is_new_block=True,
    )

    assert emitted, "Expected previous paragraph to flush on layout break"
    assert emitted[0].page_content.startswith("Existing")
    assert current == ["Next"], "New paragraph should start with the current word"
