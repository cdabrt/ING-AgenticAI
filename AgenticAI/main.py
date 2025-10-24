from AgenticAI.PDF.PDFParser import PDFParser

if __name__ == "__main__":
    elements = PDFParser.load_structured_pdfs("./data")

    for e in elements:
        print(f"{e.meta_data.type.value} (Page {e.meta_data.page}) from {e.meta_data.source}:")
        print(e.page_content)
        print("-" * 80)
