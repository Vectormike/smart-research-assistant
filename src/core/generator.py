"""
Generator component for RAG system using Ollama.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import logging
from typing import List, Dict, Any
import requests
from config.config import settings


class Generator:
    """Generator component for RAG system."""

    def __init__(self):
        """Initialize the generator."""
        self.logger = logging.getLogger(__name__)
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT
        self.max_retries = settings.OLLAMA_MAX_RETRIES

    def _format_prompt(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format the prompt for the LLM.

        Args:
            query: User's query
            retrieved_docs: Retrieved documents from vector store

        Returns:
            str: Formatted prompt
        """
        # Format context from retrieved documents
        context = "\n\n".join([
            f"Document {i+1} from {doc['metadata']['source']}:\n{doc['text']}"
            for i, doc in enumerate(retrieved_docs)
        ])

        # Create the prompt
        prompt = f"""You are a helpful research assistant. Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {query}

Answer:"""

        return prompt

    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Generate a response using Ollama.

        Args:
            query: User's query
            retrieved_docs: Retrieved documents from vector store

        Returns:
            str: Generated response
        """
        try:
            prompt = self._format_prompt(query, retrieved_docs)
            
            # Prepare the request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": settings.TEMPERATURE,
                        "top_p": settings.TOP_P,
                        "num_predict": settings.MAX_TOKENS
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API returned status code {response.status_code}: {response.text}")
            
            return response.json()["response"].strip()
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Test the generator
    from vector_store import VectorStore

    # Initialize components
    vector_store = VectorStore()
    generator = Generator()

    # Test query
    test_query = "What is RAG architecture?"
    
    # Retrieve relevant documents
    retrieved_docs = vector_store.search(test_query)
    
    # Generate response
    response = generator.generate_response(test_query, retrieved_docs)
    
    print(f"\nQuery: {test_query}")
    print(f"\nResponse: {response}") 