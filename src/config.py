"""
Configuration management for the RAG Research Notes project.
"""
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    openai_api_key: str
    
    # Paths
    data_dir: Path = Path("data")
    embeddings_dir: Path = Path("embeddings")
    
    # Model settings
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-3.5-turbo"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create global settings instance
settings = Settings() 