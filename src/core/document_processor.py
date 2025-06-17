"""
Process documents for the RAG system.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import logging
from typing import List, Dict, Any
import re

from config.config import settings


class DocumentProcessor:
    """Process documents for the RAG system."""

    def __init__(self):
        """Initialize the document processor."""
        self.logger = logging.getLogger(__name__)
        self.data_dir = settings.DATA_DIR
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

    def _read_file(self, file_path: Path) -> str:
        """Read a file and return its contents.

        Args:
            file_path: Path to the file

        Returns:
            str: File contents
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            raise

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List[str]: List of text chunks
        """
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file.

        Args:
            file_path: Path to the file

        Returns:
            Dict[str, Any]: Processed document
        """
        try:
            text = self._read_file(file_path)
            chunks = self._chunk_text(text)
            
            return {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "chunks": chunks
            }
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            raise

    def process_all_documents(self) -> List[Dict[str, Any]]:
        """Process all documents in the data directory.

        Returns:
            List[Dict[str, Any]]: List of processed documents
        """
        try:
            documents = []
            for file_path in self.data_dir.glob("*.txt"):
                doc = self.process_file(file_path)
                documents.append(doc)
                self.logger.info(f"Processed document {file_path.name}")
            return documents
        except Exception as e:
            self.logger.error(f"Error processing documents: {e}")
            raise


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Test the document processor
    processor = DocumentProcessor()
    documents = processor.process_all_documents()
    
    for doc in documents:
        print(f"\nDocument: {doc['file_name']}")
        print(f"Number of chunks: {len(doc['chunks'])}")
        print(f"First chunk: {doc['chunks'][0][:200]}...") 