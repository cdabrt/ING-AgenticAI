from AgenticAI.Chunker.Chunker import Chunker
from AgenticAI.PDF.Document import Document, Metadata, ElementType


def _meta(element_type: ElementType) -> Metadata:
    return Metadata(source="test.pdf", page=1, type=element_type)


def test_table_chunker_advances_with_overlap():
    table_text = "row1\nrow2\nrow3"
    documents = [
        Document(page_content="Heading", meta_data=_meta(ElementType.HEADING)),
        Document(page_content=table_text, meta_data=_meta(ElementType.TABLE)),
    ]

    chunks = Chunker.chunk_headings_with_paragraphs(
        documents=documents,
        chunk_size=5,
        chunk_overlap=0,
        table_row_overlap=1,
    )

    assert len(chunks) == 3
    chunk_texts = [chunk.document.page_content for chunk in chunks]
    assert chunk_texts[0] == "row1"
    assert chunk_texts[-1] == "row3"


def test_paragraph_chunk_size_respected():
    long_text = " ".join(["sentence" + str(i) for i in range(500)])
    documents = [
        Document(page_content="Heading", meta_data=_meta(ElementType.HEADING)),
        Document(page_content=long_text, meta_data=_meta(ElementType.PARAGRAPH)),
    ]

    chunk_size = 200
    chunks = Chunker.chunk_headings_with_paragraphs(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=50,
        table_row_overlap=0,
    )

    assert len(chunks) > 1
    assert all(len(chunk.document.page_content) <= chunk_size for chunk in chunks)
