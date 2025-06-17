# RAG System Configuration

This directory contains configuration settings for the RAG system.

## Configuration Structure

The system uses a combination of:
1. Default settings in `config.py`
2. Environment variables (optional)
3. Command-line arguments (optional)

## Key Settings

### Ollama Settings
- `OLLAMA_BASE_URL`: URL for Ollama API (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model to use (default: llama2)
- `OLLAMA_TIMEOUT`: Request timeout in seconds (default: 120)
- `OLLAMA_MAX_RETRIES`: Maximum number of retries (default: 3)

### Embedding Settings
- `EMBEDDING_MODEL`: Model for generating embeddings (default: all-MiniLM-L6-v2)
- `EMBEDDING_DIMENSION`: Dimension of embedding vectors (default: 384)
- `EMBEDDING_BATCH_SIZE`: Batch size for embedding generation (default: 32)

### Vector Database Settings
- `CHROMA_PERSIST_DIR`: Directory for ChromaDB persistence
- `CHROMA_COLLECTION_NAME`: Name of the collection (default: research_notes)

### Document Processing
- `CHUNK_SIZE`: Size of text chunks (default: 500)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)

### Retrieval Settings
- `TOP_K_RESULTS`: Number of results to retrieve (default: 3)
- `SIMILARITY_THRESHOLD`: Minimum similarity score (default: 0.7)

### Generation Settings
- `MAX_TOKENS`: Maximum tokens in response (default: 1000)
- `TEMPERATURE`: Generation temperature (default: 0.7)
- `TOP_P`: Top-p sampling parameter (default: 0.9)

### System Prompts
- `SYSTEM_PROMPT`: Default system prompt for the LLM

### Logging
- `LOG_LEVEL`: Logging level (default: INFO)
- `LOG_FILE`: Path to log file
