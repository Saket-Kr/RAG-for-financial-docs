# Document Query Answerer

A production-grade document query answering system for financial documents using RAG (Retrieval Augmented Generation) with local, open-source tools.

## Features

- PDF parsing with multiple parser backends (pdfplumber, PyMuPDF, unstructured)
- Intelligent document chunking (hierarchical, semantic, sentence-based)
- Nomic Embed for embeddings
- Vector database support (Chroma, FAISS, Qdrant)
- Ollama integration for local LLM inference
- FastAPI REST API
- SQLite metadata storage
- Docker Compose deployment

## Architecture

The system follows a factory-based architecture with clear separation of concerns:

- **Parsers**: Extract text and structure from PDFs
- **Chunking**: Split documents into manageable chunks
- **Embeddings**: Generate vector representations
- **Vector DB**: Store and search embeddings
- **Query Engine**: RAG-based query answering
- **Gatekeeping**: Relevance validation

## Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)

### Docker Deployment

1. Clone the repository
2. Copy `.env.example` to `.env` and configure if needed
3. Start services:

```bash
docker-compose up -d
```

4. Pull Ollama model (in ollama container):

```bash
docker exec -it document-qa-ollama ollama pull mistral:7b-instruct
docker exec -it document-qa-ollama ollama pull nomic-embed-text
```

5. Access API at `http://localhost:8000`

Qdrant vector database will be available at `http://localhost:6333` (inside Docker network as `http://qdrant:6333`).

### Local Development

0. Create Environment (conda preferred)
```bash
conda create -n docqa python=3.11 -y
conda activate docqa
# (Optional but recommended) Update pip and conda, install common build tools:
python -m pip install --upgrade pip
conda install -c conda-forge setuptools wheel --yes
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start Ollama locally and pull model:

```bash
ollama pull mistral:7b-instruct
```

3. Update `config/settings.yaml` with local Ollama URL

4. Run application:

```bash
uvicorn app.main:app --reload
```

## API Endpoints

- `POST /api/v1/documents/upload` - Upload and process PDF
- `POST /api/v1/documents/{document_id}/query` - Query document
- `GET /api/v1/documents/{document_id}/status` - Get processing status
- `GET /api/v1/documents/{document_id}` - Get document info
- `DELETE /api/v1/documents/{document_id}` - Delete document
- `GET /health` - Health check

## Configuration

Edit `config/settings.yaml` to configure:

- PDF parser type
- Chunking strategy
- Embedding model
- Vector database
- Query strategy
- LLM settings

## Project Structure

```
document-query-answerer/
├── app/              # Application code
├── config/           # Configuration files
├── data/             # Data storage
├── logs/             # Log files
└── tests/            # Test files
```

## License

MIT
