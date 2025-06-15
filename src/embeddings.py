"""
Text embedding functionality for the RAG Research Notes project.
"""
from pathlib import Path
from typing import List

from langchain_openai import OpenAIEmbeddings

from .config import settings


class EmbeddingManager:
    """Manages text embeddings for the application."""
    
    def __init__(self):
        """Initialize the embedding manager."""
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )
        self.embeddings_dir = settings.embeddings_dir
        self.embeddings_dir.mkdir(exist_ok=True)
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        return self.embeddings.embed_query(text)
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        return self.embeddings.embed_documents(texts) 