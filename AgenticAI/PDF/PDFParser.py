import re
from pathlib import Path
from typing import Iterator, List, Tuple, Dict, Any

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
    
    # Heading detection thresholds
    HEADING_MIN_FONT_SIZE_RATIO = 1.15  # Heading should be 15% larger than body text
    HEADING_MAX_LENGTH = 150  # Characters - headings are typically shorter
    HEADING_LINE_GAP_MULTIPLIER = 1.5  # Extra spacing before/after headings

    @staticmethod
    def _analyze_page_fonts(words: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze font characteristics across a page to establish baselines.
        Returns font statistics to help identify headings vs body text.
        """
        if not words:
            return {"body_size": 12.0, "sizes": {}}
        
        # Collect font sizes
        font_sizes = []
        for word in words:
            size = word.get("size", word.get("height", 12.0))
            if size > 0:
                font_sizes.append(size)
        
        if not font_sizes:
            return {"body_size": 12.0, "sizes": {}}
        
        # Use median as body text size (more robust than mean)
        sorted_sizes = sorted(font_sizes)
        median_idx = len(sorted_sizes) // 2
        body_size = sorted_sizes[median_idx]
        
        # Count frequency of each font size
        size_counts = {}
        for size in font_sizes:
            size_counts[size] = size_counts.get(size, 0) + 1
        
        return {
            "body_size": body_size,
            "sizes": size_counts,
            "max_size": max(font_sizes),
            "min_size": min(font_sizes)
        }
    
    @staticmethod
    def _is_likely_heading(
        word: Dict[str, Any],
        font_stats: Dict[str, Any],
        text_length: int,
        has_gap_before: bool
    ) -> bool:
        """
        Determine if text is likely a heading using multiple heuristics.
        Not just based on bold, but font size, length, spacing, etc.
        """
        # Get font characteristics
        is_bold = "Bold" in word.get("fontname", "")
        font_size = word.get("size", word.get("height", font_stats["body_size"]))
        body_size = font_stats["body_size"]
        
        # Calculate heading score based on multiple factors
        score = 0
        
        # Factor 1: Font size (most reliable indicator)
        if font_size >= body_size * PDFParser.HEADING_MIN_FONT_SIZE_RATIO:
            score += 3
        
        # Factor 2: Bold text (but not decisive on its own)
        if is_bold:
            score += 1
        
        # Factor 3: Short text (headings are typically concise)
        if text_length <= PDFParser.HEADING_MAX_LENGTH:
            score += 1
        
        # Factor 4: Extra spacing before (headings often have more space)
        if has_gap_before:
            score += 1
        
        # Factor 5: Text doesn't end with sentence-ending punctuation
        # (headings often don't end with periods)
        # This is checked at a higher level
        
        # Require multiple positive indicators, not just bold
        # Score >= 3 suggests heading (e.g., larger font + bold, or larger font + short + spacing)
        return score >= 3
    
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
    def _group_into_lines(words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group words into lines based on their vertical position.
        Words with similar 'top' values are on the same line.
        """
        if not words:
            return []
        
        lines = []
        current_line = [words[0]]
        current_top = words[0]["top"]
        
        for word in words[1:]:
            # If word is roughly at same height (within 3 points), it's on the same line
            if abs(word["top"] - current_top) <= 3:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
                current_top = word["top"]
        
        if current_line:
            lines.append(current_line)
        
        return lines

    @staticmethod
    def _extract_headings_and_paragraphs(words, pdf_file, page_num):
        """
        Extract headings and paragraphs using multiple heuristics, not just bold text.
        More PDF-agnostic approach that works with documents from various sources.
        All text is processed uniformly - no special treatment based on formatting alone.
        """
        all_elements = []
        if not words:
            return all_elements
        
        # Analyze page fonts to establish baselines
        font_stats = PDFParser._analyze_page_fonts(words)
        
        # Group words into lines
        lines = PDFParser._group_into_lines(words)
        
        current_paragraph = []
        current_type = ElementType.PARAGRAPH
        last_line_top = None
        
        for line_idx, line in enumerate(lines):
            if not line:
                continue
            
            # Get line text and properties
            line_text = " ".join([w["text"].strip() for w in line if w["text"].strip()])
            if not line_text:
                continue
            
            line_top = line[0]["top"]
            
            # Check for gap before this line (natural paragraph/section break)
            gap_before = 0
            if last_line_top is not None:
                gap_before = abs(line_top - last_line_top)
            
            has_significant_gap = gap_before >= PDFParser.PARAGRAPH_GAP_THRESHOLD * PDFParser.HEADING_LINE_GAP_MULTIPLIER
            has_paragraph_gap = gap_before >= PDFParser.PARAGRAPH_GAP_THRESHOLD
            
            # Determine if this line is a heading using multiple signals
            is_heading = PDFParser._is_likely_heading(
                line[0],  # Use first word for font characteristics
                font_stats,
                len(line_text),
                has_significant_gap
            )
            
            # Check if this is a special marker (numbered para or bullet)
            first_word = line[0]["text"].strip()
            is_numbered_para = PDFParser.PAR_NUMBER_REGEX.match(first_word)
            is_bullet = first_word in PDFParser.BULLET_MARKERS
            
            # Flush current paragraph if needed
            should_flush = False
            if current_paragraph:
                # Flush on: heading detection, significant gap, numbered para, or bullet
                if is_heading or has_paragraph_gap or is_numbered_para or is_bullet:
                    should_flush = True
                # Also flush if paragraph is getting long and ends with sentence
                elif len(" ".join(current_paragraph)) >= PDFParser.AUTO_PARAGRAPH_CHAR_LIMIT:
                    last_char = current_paragraph[-1][-1:] if current_paragraph else ""
                    if last_char in PDFParser.SENTENCE_ENDINGS:
                        should_flush = True
                    # Force flush if way over limit
                    elif len(" ".join(current_paragraph)) >= PDFParser.AUTO_PARAGRAPH_CHAR_LIMIT + PDFParser.AUTO_PARAGRAPH_FORCE_MARGIN:
                        should_flush = True
            
            if should_flush:
                doc = Document(
                    page_content=" ".join(current_paragraph).strip(),
                    meta_data=Metadata(source=pdf_file.name, page=page_num, type=current_type)
                )
                all_elements.append(doc)
                current_paragraph = []
                current_type = ElementType.PARAGRAPH
            
            # Add the heading as standalone element
            if is_heading:
                doc = Document(
                    page_content=line_text,
                    meta_data=Metadata(source=pdf_file.name, page=page_num, type=ElementType.HEADING)
                )
                all_elements.append(doc)
            else:
                # Add line content to current paragraph
                current_paragraph.append(line_text)
            
            last_line_top = line_top
        
        # Add remaining paragraph
        if current_paragraph:
            doc = Document(
                page_content=" ".join(current_paragraph).strip(),
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
                # Extract words with font characteristics (including size/height)
                words = page.extract_words(extra_attrs=["fontname", "top", "size", "height"])
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
