"""
Document processor for reading and processing text files.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
from typing import List, Dict, Any

from config.config import settings


class DocumentProcessor:
    """Processes documents from the data directory."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.logger = logging.getLogger(__name__)
        self.data_dir = settings.DATA_DIR
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
    
    def read_document(self, file_path: Path) -> str:
        """Read a single document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            str: Content of the document
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    def get_all_documents(self) -> List[Path]:
        """Get all .txt files from the data directory.
        
        Returns:
            List[Path]: List of paths to text files
        """
        try:
            return list(self.data_dir.glob("*.txt"))
        except Exception as e:
            self.logger.error(f"Error getting documents from {self.data_dir}: {e}")
            raise
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List[str]: List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Find the end of the chunk
            end = start + self.chunk_size
            
            # If this is not the first chunk, try to find a good breaking point
            if start > 0:
                # Look for the last newline in the overlap region
                overlap_start = end - self.chunk_overlap
                last_newline = text.rfind('\n', overlap_start, end)
                if last_newline != -1:
                    end = last_newline + 1
            
            # If this is not the last chunk, try to find a good breaking point
            if end < text_length:
                # Look for the last newline in the chunk
                last_newline = text.rfind('\n', start, end)
                if last_newline != -1:
                    end = last_newline + 1
            
            # Add the chunk
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move to the next chunk
            start = end
        
        return chunks
    
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """Process a single document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict[str, Any]: Document metadata and chunks
        """
        try:
            content = self.read_document(file_path)
            chunks = self.chunk_text(content)
            
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "chunks": chunks,
                "num_chunks": len(chunks)
            }
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {e}")
            raise
    
    def process_all_documents(self) -> List[Dict[str, Any]]:
        """Process all documents in the data directory.
        
        Returns:
            List[Dict[str, Any]]: List of processed documents
        """
        documents = []
        for file_path in self.get_all_documents():
            try:
                doc = self.process_document(file_path)
                documents.append(doc)
                self.logger.info(f"Processed document: {file_path.name}")
            except Exception as e:
                self.logger.error(f"Failed to process document {file_path}: {e}")
                continue
        
        return documents


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the document processor
    processor = DocumentProcessor()
    documents = processor.process_all_documents()
    
    # Print summary
    print(f"\nProcessed {len(documents)} documents:")
    for doc in documents:
        print(f"- {doc['file_name']}: {doc['num_chunks']} chunks") 