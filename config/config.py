"""
Configuration settings for the RAG system.
"""
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Settings for the RAG system."""
    
    # Base paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    EMBEDDINGS_DIR: Path = BASE_DIR / "embeddings"
    
    # Ollama settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2"
    OLLAMA_TIMEOUT: int = 120
    OLLAMA_MAX_RETRIES: int = 3
    
    # Embedding settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32
    
    # Vector database settings
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "chroma_db"
    CHROMA_COLLECTION_NAME: str = "research_notes"
    
    # Document processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Retrieval settings
    TOP_K_RESULTS: int = 3
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Generation settings
    MAX_TOKENS: int = 1000
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    
    # System prompt
    SYSTEM_PROMPT: str = """You are a helpful AI assistant that provides accurate and relevant information based on the given context. 
    If you don't know the answer or the context doesn't contain relevant information, say so. 
    Always cite your sources when providing information."""
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[Path] = BASE_DIR / "logs" / "app.log"
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create global settings instance
settings = Settings() 