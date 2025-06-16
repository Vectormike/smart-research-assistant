"""
Generate embeddings for document chunks using sentence-transformers.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from config.config import settings


class EmbeddingGenerator:
    """Generate embeddings for document chunks."""

    def __init__(self):
        """Initialize the embedding generator."""
        self.logger = logging.getLogger(__name__)
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.embeddings_dir = settings.EMBEDDINGS_DIR
        self.embeddings_dir.mkdir(exist_ok=True)

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text chunks.

        Args:
            chunks: List of text chunks

        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            embeddings = self.model.encode(chunks, batch_size=settings.EMBEDDING_BATCH_SIZE)
            return embeddings.tolist()
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise

    def process_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document to generate embeddings for its chunks.

        Args:
            doc: Document metadata and chunks

        Returns:
            Dict[str, Any]: Document with embeddings
        """
        try:
            embeddings = self.generate_embeddings(doc["chunks"])
            doc["embeddings"] = embeddings
            self.logger.info(f"Generated embeddings for {doc['file_name']}")
            return doc
        except Exception as e:
            self.logger.error(f"Error processing document {doc['file_name']}: {e}")
            raise

    def save_embeddings(self, doc: Dict[str, Any]) -> None:
        """Save document embeddings to a file.

        Args:
            doc: Document with embeddings
        """
        try:
            output_file = self.embeddings_dir / f"{doc['file_name']}.embeddings"
            with open(output_file, "w") as f:
                for i, embedding in enumerate(doc["embeddings"]):
                    f.write(f"Chunk {i}: {embedding}\n")
            self.logger.info(f"Saved embeddings for {doc['file_name']}")
        except Exception as e:
            self.logger.error(f"Error saving embeddings for {doc['file_name']}: {e}")
            raise


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Test the embedding generator
    from document_processor import DocumentProcessor

    processor = DocumentProcessor()
    generator = EmbeddingGenerator()

    documents = processor.process_all_documents()
    for doc in documents:
        doc_with_embeddings = generator.process_document(doc)
        generator.save_embeddings(doc_with_embeddings) 