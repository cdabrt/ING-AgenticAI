from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from AgenticAI.PDF.Document import Document, ElementType

MAX_DOC_CHARS = 15000


@dataclass
class ParsedDocument:
    source: str
    text: str
    headings: List[str]
    pages: List[int]
    element_count: int
    truncated: bool

    def brief(self) -> str:
        prefix = self.text[:500].replace("\n", " ")
        return f"{self.source} | chars={len(self.text)} | headings={len(self.headings)} | preview={prefix}"


def group_documents_by_source(documents: List[Document]) -> List[ParsedDocument]:
    grouped: Dict[str, List[Document]] = {}
    for doc in documents:
        grouped.setdefault(doc.meta_data.source, []).append(doc)

    packages: List[ParsedDocument] = []
    for source, items in grouped.items():
        ordered = sorted(items, key=lambda d: (d.meta_data.page, d.meta_data.type))
        headings = [doc.page_content for doc in ordered if doc.meta_data.type == ElementType.HEADING]
        pages = sorted({doc.meta_data.page for doc in ordered})
        full_text = "\n\n".join(doc.page_content for doc in ordered)
        truncated = len(full_text) > MAX_DOC_CHARS
        if truncated:
            full_text = full_text[:MAX_DOC_CHARS]

        packages.append(
            ParsedDocument(
                source=source,
                text=full_text,
                headings=headings,
                pages=pages,
                element_count=len(ordered),
                truncated=truncated,
            )
        )

    return packages
