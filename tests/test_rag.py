"""
Test script for the RAG system.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
from src.core.vector_store import VectorStore
from src.core.generator import Generator
from config.config import settings


def main():
    # Set up logging
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        # Initialize components
        logger.info("Initializing RAG components...")
        vector_store = VectorStore()
        generator = Generator()

        # Test queries
        test_queries = [
            "What is RAG architecture?",
            "How do attention mechanisms work?",
            "What are the key components of a vector database?",
            "How are embeddings generated?",
            "What metrics are used to evaluate RAG systems?"
        ]

        for query in test_queries:
            print("\n" + "="*80)
            print(f"Query: {query}")
            print("="*80)

            # Retrieve relevant documents
            logger.info(f"Searching for relevant documents...")
            retrieved_docs = vector_store.search(query)
            
            print("\nRetrieved Documents:")
            print("-"*40)
            for i, doc in enumerate(retrieved_docs, 1):
                print(f"\nDocument {i}:")
                print(f"Source: {doc['metadata']['source']}")
                print(f"Text: {doc['text'][:200]}...")
                print(f"Similarity: {1 - doc['distance']:.2f}")

            # Generate response
            logger.info("Generating response...")
            response = generator.generate_response(query, retrieved_docs)
            
            print("\nGenerated Response:")
            print("-"*40)
            print(response)

    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise


if __name__ == "__main__":
    main() 