from AgenticAI.Chunker.Chunker import Chunker
from AgenticAI.PDF.PDFParser import PDFParser

if __name__ == "__main__":
    elements = PDFParser.load_structured_pdfs("./data")

    chunks = Chunker.chunk_headings_with_paragraphs(
        documents=elements,
        chunk_size=1000,
        chunk_overlap=150,
        table_row_overlap=1
    )

    for chunk in chunks:
        print(f"id: {chunk.chunk_id}, type: {chunk.document.meta_data.type.value} (Page {chunk.document.meta_data.page}) \n"
              f"from: {chunk.document.meta_data.source}, parentHeader: \"{chunk.parent_heading}\":")
        print(chunk.document.page_content)
        print("-" * 80)
