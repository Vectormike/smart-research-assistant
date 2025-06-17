# RAG Research Notes

A Python project for managing and querying research notes using Retrieval-Augmented Generation (RAG).

## Project Structure

```
.
├── config/         # Configuration files
├── data/          # Data storage
├── embeddings/    # Vector embeddings
├── src/           # Source code
└── utils/         # Utility functions
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate 
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your configuration:
```env
OPENAI_API_KEY=your_api_key_here
```

## Development

- Use `black` for code formatting
- Use `isort` for import sorting
- Use `flake8` for linting
- Use `pytest` for testing

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- Add other environment variables as needed

## License

MIT 