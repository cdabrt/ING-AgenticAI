import re
import pdfplumber
from pathlib import Path
from AgenticAI.PDF.Document import Document, Metadata, ElementType

class PDFParser:
    PAR_NUMBER_REGEX = re.compile(r"^\(\d+\)")
    SUBPAR_REGEX = re.compile(r"^\([a-z]+\)$|^\([ivxlcdm]+\)$|^\([a-z]{2}\)$")

    @staticmethod
    def extract_tables(page, pdf_file, page_num):
        all_elements = []
        #Extract tables
        for table in page.extract_tables():
            table_text = "\n".join([" | ".join(cell if cell else "" for cell in row) for row in table])
            doc = Document(
                page_content=table_text,
                meta_data=Metadata(source=pdf_file.name, page=page_num, type=ElementType.TABLE)
            )
            all_elements.append(doc)
        return all_elements

    @staticmethod
    def process_heading(word, current_heading, last_top, pdf_file, page_num):
        is_bold = "Bold" in word["fontname"]
        if not is_bold:
            return current_heading, last_top, None

        doc_to_add = None
        if last_top is not None and abs(word["top"] - last_top) > 3 and current_heading:
            # new line → save previous heading
            doc_to_add = Document(
                page_content=" ".join(current_heading),
                meta_data=Metadata(source=pdf_file.name, page=page_num, type=ElementType.HEADING)
            )
            current_heading = []

        current_heading.append(word["text"].strip())
        last_top = word["top"]
        return current_heading, last_top, doc_to_add

    @staticmethod
    def process_paragraph(word, current_paragraph, pdf_file, page_num, current_type):
        text = word["text"].strip()
        if not text:
            return current_paragraph, None

        doc_to_add = None
        if PDFParser.PAR_NUMBER_REGEX.match(text):
            if current_paragraph:
                doc_to_add = Document(
                    page_content=" ".join(current_paragraph),
                    meta_data=Metadata(source=pdf_file.name, page=page_num, type=current_type)
                )
                current_paragraph = []
            current_paragraph.append(text)
        elif PDFParser.SUBPAR_REGEX.match(text):
            current_paragraph.append(text)
        else:
            current_paragraph.append(text)

        return current_paragraph, doc_to_add

    @staticmethod
    def extract_headings_and_paragraphs(words, pdf_file, page_num):
        all_elements = []
        if not words:
            return all_elements

        current_paragraph = []
        current_heading = []
        last_top = None
        current_type = ElementType.PARAGRAPH

        for word in words:
            # Process headings
            current_heading, last_top, heading_doc = PDFParser.process_heading(
                word, current_heading, last_top, pdf_file, page_num
            )
            if heading_doc:
                all_elements.append(heading_doc)

            # If the word is not bold, process paragraphs
            if "Bold" not in word["fontname"]:
                current_paragraph, para_doc = PDFParser.process_paragraph(
                    word, current_paragraph, pdf_file, page_num, current_type
                )
                if para_doc:
                    all_elements.append(para_doc)

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
    def process_pdf(pdf_file):
        all_elements = []
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                all_elements.extend(PDFParser.extract_tables(page, pdf_file, page_num))
                words = page.extract_words(extra_attrs=["fontname", "top"])
                all_elements.extend(PDFParser.extract_headings_and_paragraphs(words, pdf_file, page_num))
        return all_elements

    @staticmethod
    def load_structured_pdfs(folder_path: str):
        pdf_dir = Path(folder_path)
        all_elements = []

        for pdf_file in pdf_dir.glob("*.pdf"):
            all_elements.extend(PDFParser.process_pdf(pdf_file))

        return all_elements
