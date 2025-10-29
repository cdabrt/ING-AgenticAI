from AgenticAI.Chunker.Chunker import Chunker
from AgenticAI.PDF.PDFParser import PDFParser

if __name__ == "__main__":
    elements = PDFParser.load_structured_pdfs("./data")

    CHUNK_SIZE : int = 1000
    CHUNK_OVERLAP : int = 150
    CHUNK_TABLE_OVERLAP : int = 1

    chunks = Chunker.chunk_headings_with_paragraphs(
        documents=elements,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        table_row_overlap=CHUNK_TABLE_OVERLAP
    )

    for chunk in chunks:
        print(f"id: {chunk.chunk_id}, type: {chunk.document.meta_data.type.value} (Page {chunk.document.meta_data.page}) \n"
              f"from: {chunk.document.meta_data.source}, parentHeader: \"{chunk.parent_heading}\":")
        print(chunk.document.page_content)
        print("-" * 80)
