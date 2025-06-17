"""
Vector store for document embeddings using ChromaDB.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import logging
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from config.config import settings


class VectorStore:
    """Vector store for document embeddings."""

    def __init__(self):
        """Initialize the vector store."""
        self.logger = logging.getLogger(__name__)
        self.client = chromadb.PersistentClient(
            path=str(settings.CHROMA_PERSIST_DIR),
            settings=ChromaSettings(
                anonymized_telemetry=False
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_document(self, doc: Dict[str, Any]) -> None:
        """Add a document and its embeddings to the vector store.

        Args:
            doc: Document with chunks and embeddings
        """
        try:
            # Prepare metadata for each chunk
            metadatas = [
                {
                    "source": doc["file_name"],
                    "chunk_index": i,
                    "file_path": doc["file_path"]
                }
                for i in range(len(doc["chunks"]))
            ]

            # Add documents to collection
            self.collection.add(
                embeddings=doc["embeddings"],
                documents=doc["chunks"],
                metadatas=metadatas,
                ids=[f"{doc['file_name']}_{i}" for i in range(len(doc["chunks"]))]
            )
            self.logger.info(f"Added document {doc['file_name']} to vector store")
        except Exception as e:
            self.logger.error(f"Error adding document {doc['file_name']} to vector store: {e}")
            raise

    def search(self, query: str, n_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for similar documents.

        Args:
            query: Query text
            n_results: Number of results to return (defaults to settings.TOP_K_RESULTS)

        Returns:
            List[Dict[str, Any]]: List of similar documents with their metadata
        """
        try:
            n_results = n_results or settings.TOP_K_RESULTS
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })
            
            return formatted_results
        except Exception as e:
            self.logger.error(f"Error searching vector store: {e}")
            raise


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Test the vector store
    from document_processor import DocumentProcessor
    from embeddings import EmbeddingGenerator

    # Process documents and generate embeddings
    processor = DocumentProcessor()
    generator = EmbeddingGenerator()
    documents = processor.process_all_documents()
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Add documents to vector store
    for doc in documents:
        doc_with_embeddings = generator.process_document(doc)
        vector_store.add_document(doc_with_embeddings)
    
    # Test search
    test_query = "What is RAG architecture?"
    results = vector_store.search(test_query)
    
    print(f"\nSearch results for: {test_query}")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. From: {result['metadata']['source']}")
        print(f"Text: {result['text'][:200]}...")
        print(f"Similarity: {1 - result['distance']:.2f}") 