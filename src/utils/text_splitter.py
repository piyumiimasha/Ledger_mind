"""Simple text splitters without external dependencies."""
from typing import List, Dict, Any


def recursive_split(
    text: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
    separators: List[str] = None
) -> List[str]:
    """
    Recursively split text on separators.

    Args:
        text: Text to split
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between consecutive chunks
        separators: List of separators to try (defaults to paragraphs, lines, sentences, spaces)

    Returns:
        List of text chunks
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", ", ", " ", ""]

    chunks = []
    current_separator = separators[0] if separators else ""

    # Base case: if text is small enough, return it
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Try splitting on current separator
    splits = text.split(current_separator) if current_separator else [text]

    # Process splits
    current_chunk = ""
    for i, split in enumerate(splits):
        # If this split is too large, try next separator
        if len(split) > chunk_size and len(separators) > 1:
            # Save current chunk if exists
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            # Recursively split this piece
            sub_chunks = recursive_split(
                split,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators[1:]
            )
            chunks.extend(sub_chunks)
        else:
            # Try adding to current chunk
            separator_to_add = current_separator if current_chunk else ""
            test_chunk = current_chunk + separator_to_add + split

            if len(test_chunk) <= chunk_size:
                current_chunk = test_chunk
            else:
                # Current chunk is full, save it
                if current_chunk:
                    chunks.append(current_chunk)

                # Start new chunk with overlap
                if chunk_overlap > 0 and current_chunk:
                    # Take last chunk_overlap characters from previous chunk
                    overlap_text = current_chunk[-chunk_overlap:]
                    current_chunk = overlap_text + separator_to_add + split
                else:
                    current_chunk = split

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def fixed_split(
    text: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
    separator: str = " "
) -> List[str]:
    """
    Split text into fixed-size chunks on a separator.

    Args:
        text: Text to split
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between consecutive chunks
        separator: Separator to split on (default: space)

    Returns:
        List of text chunks
    """
    # Split on separator
    splits = text.split(separator)

    chunks = []
    current_chunk = ""

    for i, split in enumerate(splits):
        # Try adding to current chunk
        separator_to_add = separator if current_chunk else ""
        test_chunk = current_chunk + separator_to_add + split

        if len(test_chunk) <= chunk_size:
            current_chunk = test_chunk
        else:
            # Current chunk is full, save it
            if current_chunk:
                chunks.append(current_chunk)

            # Start new chunk with overlap
            if chunk_overlap > 0 and current_chunk:
                # Take last chunk_overlap characters from previous chunk
                overlap_text = current_chunk[-chunk_overlap:]
                current_chunk = overlap_text + separator + split
            else:
                current_chunk = split

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


class Document:
    """Simple document class compatible with LangChain."""

    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"


class TextSplitter:
    """Base text splitter class."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks. Override in subclasses."""
        raise NotImplementedError

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        chunks = []
        for doc in documents:
            text_chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(text_chunks):
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "chunk_index": i}
                )
                chunks.append(chunk_doc)
        return chunks


class RecursiveCharacterTextSplitter(TextSplitter):
    """Recursively split text on multiple separators."""

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
        separators: List[str] = None
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ". ", ", ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split text recursively."""
        return recursive_split(
            text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )


class CharacterTextSplitter(TextSplitter):
    """Split text on a single separator."""

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
        separator: str = " "
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.separator = separator

    def split_text(self, text: str) -> List[str]:
        """Split text on separator."""
        return fixed_split(
            text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=self.separator
        )


def get_splitter(strategy: str, chunk_size: int = 1500, chunk_overlap: int = 150) -> TextSplitter:
    """
    Return a text splitter based on the specified strategy.

    Args:
        strategy: One of "recursive", "fixed"
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        A TextSplitter instance
    """
    if strategy == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
    elif strategy == "fixed":
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" "
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'recursive' or 'fixed'")
