import re
from pathlib import Path
from typing import Iterator, List, Tuple

import pdfplumber

from AgenticAI.PDF.Document import Document, ElementType, Metadata

# Made static as the parser is built from pure functions. No state has to be saved.
class PDFParser:
    PAR_NUMBER_REGEX = re.compile(r"^\(\d+\)")
    SUBPAR_REGEX = re.compile(r"^\([a-z]+\)$|^\([ivxlcdm]+\)$|^\([a-z]{2}\)$")
    BULLET_MARKERS = {"•", "-", "–", "—", "·", "*"}
    PARAGRAPH_GAP_THRESHOLD = 12.0
    AUTO_PARAGRAPH_CHAR_LIMIT = 900
    AUTO_PARAGRAPH_FORCE_MARGIN = 100
    SENTENCE_ENDINGS = {".", "!", "?", ";", ":"}

    @staticmethod
    def _extract_tables(page, pdf_file, page_num):
        all_elements = []
        for table in page.extract_tables():
            table_text = "\n".join([" | ".join(cell if cell else "" for cell in row) for row in table])
            doc = Document(
                page_content=table_text,
                meta_data=Metadata(source=pdf_file.name, page=page_num, type=ElementType.TABLE)
            )
            all_elements.append(doc)
        return all_elements

    @staticmethod
    def _process_heading(word, current_heading, last_top, pdf_file, page_num):
        is_bold = "Bold" in word["fontname"]
        if not is_bold:
            return current_heading, last_top, None

        doc_to_add = None
        if last_top is not None and abs(word["top"] - last_top) > 3 and current_heading:
            doc_to_add = Document(
                page_content=" ".join(current_heading),
                meta_data=Metadata(source=pdf_file.name, page=page_num, type=ElementType.HEADING)
            )
            current_heading = []

        current_heading.append(word["text"].strip())
        last_top = word["top"]
        return current_heading, last_top, doc_to_add

    @staticmethod
    def _process_paragraph(
        word,
        current_paragraph: List[str],
        pdf_file,
        page_num: int,
        current_type: ElementType,
        is_new_block: bool,
    ) -> Tuple[List[str], List[Document]]:
        text = word["text"].strip()
        produced: List[Document] = []

        def flush_paragraph():
            nonlocal current_paragraph
            if not current_paragraph:
                return
            produced.append(
                Document(
                    page_content=" ".join(current_paragraph).strip(),
                    meta_data=Metadata(source=pdf_file.name, page=page_num, type=current_type),
                )
            )
            current_paragraph = []

        if not text:
            return current_paragraph, produced

        if is_new_block and current_paragraph:
            flush_paragraph()

        if PDFParser.PAR_NUMBER_REGEX.match(text):
            flush_paragraph()
            current_paragraph.append(text)
            return current_paragraph, produced

        if text in PDFParser.BULLET_MARKERS:
            flush_paragraph()
            current_paragraph.append(text)
            return current_paragraph, produced

        current_paragraph.append(text)

        if PDFParser.SUBPAR_REGEX.match(text):
            return current_paragraph, produced

        paragraph_len = len(" ".join(current_paragraph))
        if paragraph_len >= PDFParser.AUTO_PARAGRAPH_CHAR_LIMIT:
            should_flush = text[-1:] in PDFParser.SENTENCE_ENDINGS
            if not should_flush:
                should_flush = (
                    paragraph_len >= PDFParser.AUTO_PARAGRAPH_CHAR_LIMIT + PDFParser.AUTO_PARAGRAPH_FORCE_MARGIN
                )
            if should_flush:
                flush_paragraph()

        return current_paragraph, produced

    @staticmethod
    def _extract_headings_and_paragraphs(words, pdf_file, page_num):
        all_elements = []
        if not words:
            return all_elements

        current_paragraph = []
        current_heading = []
        last_top = None
        last_body_top = None
        current_type = ElementType.PARAGRAPH

        for word in words:
            # Process headings
            current_heading, last_top, heading_doc = PDFParser._process_heading(
                word, current_heading, last_top, pdf_file, page_num
            )
            if heading_doc:
                all_elements.append(heading_doc)

            # If the word is not bold, process paragraphs
            if "Bold" not in word["fontname"]:
                is_new_block = False
                if last_body_top is not None:
                    is_new_block = abs(word["top"] - last_body_top) >= PDFParser.PARAGRAPH_GAP_THRESHOLD

                current_paragraph, para_docs = PDFParser._process_paragraph(
                    word,
                    current_paragraph,
                    pdf_file,
                    page_num,
                    current_type,
                    is_new_block,
                )
                if para_docs:
                    all_elements.extend(para_docs)
                last_body_top = word["top"]

        # Add remaining headings and paragraphs
        if current_heading:
            doc = Document(
                page_content=" ".join(current_heading),
                meta_data=Metadata(source=pdf_file.name, page=page_num, type=ElementType.HEADING)
            )
            all_elements.append(doc)

        if current_paragraph:
            doc = Document(
                page_content=" ".join(current_paragraph),
                meta_data=Metadata(source=pdf_file.name, page=page_num, type=current_type)
            )
            all_elements.append(doc)

        return all_elements

    @staticmethod
    def _process_pdf(pdf_file: Path):
        all_elements = []
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                all_elements.extend(PDFParser._extract_tables(page, pdf_file, page_num))
                words = page.extract_words(extra_attrs=["fontname", "top"])
                all_elements.extend(PDFParser._extract_headings_and_paragraphs(words, pdf_file, page_num))
        return all_elements

    @staticmethod
    def load_structured_pdf(pdf_file: str | Path):
        pdf_path = Path(pdf_file)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file {pdf_file} does not exist")
        return PDFParser._process_pdf(pdf_path)

    @staticmethod
    def iter_structured_pdfs(folder_path: str) -> Iterator[Tuple[Path, list[Document]]]:
        pdf_dir = Path(folder_path)
        for pdf_file in pdf_dir.glob("*.pdf"):
            yield pdf_file, PDFParser._process_pdf(pdf_file)

    @staticmethod
    def load_structured_pdfs(folder_path: str):
        pdf_dir = Path(folder_path)
        all_elements = []

        for pdf_file in pdf_dir.glob("*.pdf"):
            all_elements.extend(PDFParser._process_pdf(pdf_file))

        return all_elements
