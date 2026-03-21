"""
Document loader and cleaner utility for PDF documents.
Handles loading PDFs and removing headers/footers for clean text extraction.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
from pypdf import PdfReader


def load_pdf_documents(pdf_dir: Path) -> List[Dict[str, str]]:
    """
    Load all PDF documents from a directory.

    Args:
        pdf_dir: Path to directory containing PDF files

    Returns:
        List of dicts with 'source' (filename) and 'content' (raw text) keys
    """
    pdf_dir = Path(pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    documents = []
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
            pdf_text = "\n".join(
                (page.extract_text() or "") for page in reader.pages
            )
            documents.append({
                "source": pdf_path.name,
                "content": pdf_text
            })
            print(f"✓ Loaded {pdf_path.name}: {len(pdf_text):,} chars from {len(reader.pages)} pages")
        except Exception as e:
            print(f"✗ Failed to load {pdf_path.name}: {e}")

    return documents


def remove_headers_footers(text: str, threshold: int = 100) -> str:
    """
    Remove common header/footer patterns from document text.

    Args:
        text: Raw document text with potential headers/footers
        threshold: Min character count to keep a line (removes very short lines at page boundaries)

    Returns:
        Cleaned text with headers/footers removed
    """
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        # Skip empty lines
        if not line.strip():
            cleaned_lines.append('')
            continue

        stripped = line.strip()

        # Skip common header/footer patterns
        # Page numbers (e.g., "1", "Page 1", "1 of 50")
        if re.match(r'^(?:page\s*)?(\d+)(?:\s+of\s+\d+)?$', stripped, re.IGNORECASE):
            continue

        # Company name/report headers (very common)
        if re.match(r'^(?:confidential|internal use|©|copyright|all rights reserved)', stripped, re.IGNORECASE):
            continue

        # Date footers (e.g., "March 2024", "2024-03-15")
        if re.match(r'^(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4})$', stripped, re.IGNORECASE):
            continue

        # URLs and email patterns (common in headers)
        if re.match(r'^(?:https?://|www\.|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})', stripped, re.IGNORECASE):
            continue

        # Section/chapter indicators at start of line (e.g., "- 1 -", "Section 1")
        if re.match(r'^[-–—]?\s*\d+\s*[-–—]?$', stripped):
            continue

        # Very short lines at document boundaries are likely headers/footers
        # (but keep reasonable short lines like "Yes" or "No")
        if 0 < len(stripped) < 3 and not stripped.isalpha():
            continue

        cleaned_lines.append(line)

    # Join lines and clean up excessive whitespace
    cleaned_text = '\n'.join(cleaned_lines)

    # Remove multiple consecutive blank lines (reduce to max 2)
    cleaned_text = re.sub(r'\n\n\n+', '\n\n', cleaned_text)

    return cleaned_text.strip()


def clean_documents(documents: List[Dict[str, str]], remove_footers: bool = True) -> List[Dict[str, str]]:
    """
    Clean documents by removing headers/footers and normalizing text.

    Args:
        documents: List of document dicts with 'source' and 'content' keys
        remove_footers: Whether to remove headers/footers

    Returns:
        List of cleaned document dicts
    """
    cleaned_docs = []

    for doc in documents:
        content = doc['content']
        original_len = len(content)

        if remove_footers:
            content = remove_headers_footers(content)

        # Normalize whitespace
        # Replace multiple spaces with single space (but preserve line breaks)
        lines = content.split('\n')
        normalized_lines = [' '.join(line.split()) for line in lines]
        content = '\n'.join(normalized_lines)

        cleaned_docs.append({
            "source": doc['source'],
            "content": content,
            "original_chars": original_len,
            "cleaned_chars": len(content)
        })

        reduction = ((original_len - len(content)) / original_len * 100) if original_len > 0 else 0
        print(f"✓ Cleaned {doc['source']}: {original_len:,} → {len(content):,} chars ({reduction:.1f}% removed)")

    return cleaned_docs


def save_cleaned_text(documents: List[Dict[str, str]], output_dir: Path) -> None:
    """
    Save cleaned documents to text files.

    Args:
        documents: List of cleaned document dicts
        output_dir: Directory to save text files in
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for doc in documents:
        # Generate output filename
        source_name = Path(doc['source']).stem
        output_file = output_dir / f"{source_name}_cleaned.txt"

        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(doc['content'])

        print(f"✓ Saved to {output_file}")
