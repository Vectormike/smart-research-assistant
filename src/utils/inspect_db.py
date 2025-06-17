"""
Utility script to inspect ChromaDB collection contents and vectors.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
from typing import Dict, Any, List
import numpy as np
from pprint import pprint

import chromadb
from chromadb.config import Settings as ChromaSettings
from config.config import settings


def inspect_collection() -> None:
    """Inspect the contents of the ChromaDB collection."""
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path=str(settings.CHROMA_PERSIST_DIR),
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    
    # Get collection
    collection = client.get_collection(settings.CHROMA_COLLECTION_NAME)
    
    # Get collection info
    print("\n=== Collection Info ===")
    print(f"Name: {collection.name}")
    print(f"Count: {collection.count()} documents")
    
    # Get all documents with their metadata and embeddings
    results = collection.get(include=['embeddings', 'documents', 'metadatas'])
    
    # Group documents by source file
    documents_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for i in range(len(results["ids"])):
        metadata = results["metadatas"][i]
        source = metadata["source"]
        if source not in documents_by_source:
            documents_by_source[source] = []
        
        documents_by_source[source].append({
            "chunk_index": metadata["chunk_index"],
            "text": results["documents"][i],
            "id": results["ids"][i],
            "embedding": results["embeddings"][i]
        })
    
    # Print documents grouped by source
    print("\n=== Documents by Source ===")
    for source, chunks in documents_by_source.items():
        print(f"\n{source} ({len(chunks)} chunks):")
        for chunk in sorted(chunks, key=lambda x: x["chunk_index"]):
            print(f"\nChunk {chunk['chunk_index']}:")
            print(f"ID: {chunk['id']}")
            print(f"Text: {chunk['text'][:200]}...")
            print(f"Embedding shape: {len(chunk['embedding'])}")
            print(f"Embedding (first 5 values): {chunk['embedding'][:5]}")
            print(f"Embedding stats:")
            print(f"  - Mean: {np.mean(chunk['embedding']):.4f}")
            print(f"  - Std: {np.std(chunk['embedding']):.4f}")
            print(f"  - Min: {np.min(chunk['embedding']):.4f}")
            print(f"  - Max: {np.max(chunk['embedding']):.4f}")


def search_example(query: str, n_results: int = 3) -> None:
    """Perform a search and show results with vector similarities.
    
    Args:
        query: Search query
        n_results: Number of results to show
    """
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path=str(settings.CHROMA_PERSIST_DIR),
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    
    # Get collection
    collection = client.get_collection(settings.CHROMA_COLLECTION_NAME)
    
    # Perform search
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=['distances', 'documents', 'metadatas']
    )
    
    # Print results
    print(f"\n=== Search Results for: {query} ===")
    for i in range(len(results["documents"][0])):
        print(f"\nResult {i + 1}:")
        print(f"Source: {results['metadatas'][0][i]['source']}")
        print(f"Chunk Index: {results['metadatas'][0][i]['chunk_index']}")
        print(f"Similarity: {1 - results['distances'][0][i]:.4f}")
        print(f"Text: {results['documents'][0][i][:200]}...")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Inspect collection
    inspect_collection()
    
    # Example searches
    print("\n=== Example Searches ===")
    search_example("What is RAG architecture?")
    search_example("How do vector databases work?")
    search_example("What are the evaluation metrics for RAG?") 