# Document Query Answerer

A RAG system for querying financial documents. Upload a PDF, ask questions, get answers using semantic search and LLM reasoning.

## Setup Instructions

### Option 1: Local Development

This setup runs the API and Ollama locally while using a Docker container for Qdrant.

**Prerequisites:**
- Python 3.11+
- Docker and Docker Compose (for Qdrant)
- Ollama installed locally
- (Optional) Conda for environment management

**Steps:**

1. **Create and activate environment:**
```bash
conda create -n docqa python=3.11 -y
conda activate docqa
pip install -r requirements.txt
```

2. **Install Ollama locally** (if not already installed):
   - Visit https://ollama.ai and follow installation instructions
   - Or use: `curl -fsSL https://ollama.ai/install.sh | sh` (Linux/Mac)

3. **Start Qdrant container:**
```bash
docker-compose up -d qdrant
```

4. **Start Ollama service locally and pull required models:**
```bash
# Start Ollama service (if not already running)
ollama serve

# In another terminal, pull models:
ollama pull mistral:7b-instruct
ollama pull nomic-embed-text
```

5. **Verify Ollama is accessible:**
```bash
curl http://localhost:11434/api/tags
```

6. **Update configuration** (`config/settings.yaml`):
   - Set `embeddings.api_url` to `http://localhost:11434`
   - Set `query_answering.llm.base_url` to `http://localhost:11434`
   - Set `vector_db.host` to `localhost`

7. **Run the API:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### Option 2: Full Docker Compose Deployment

All services run in containers. Note: This requires significant resources (16GB+ RAM recommended) due to running Mistral 7B and Nomic Embed.

**Steps:**

1. **Create .env:**
```bash
cp env.example .env  # Edit if needed
```

2. **Start all services:**
```bash
docker-compose up -d
```

3. **Pull models in the Ollama container:**
```bash
docker exec -it document-qa-ollama ollama pull mistral:7b-instruct
docker exec -it document-qa-ollama ollama pull nomic-embed-text
```

4. **Wait for services to be healthy:**
```bash
docker-compose ps
```

5. **Access the API:**
- API: `http://localhost:8000`
- Qdrant Dashboard: `http://localhost:6333`

**Resource Requirements:**
- Minimum 16GB RAM
- 10GB+ disk space for models
- CPU with multiple cores (GPU optional but recommended)

---

## Design Choices

### Code Architecture

**Factory Pattern Implementation:**
The system uses factory patterns throughout to enable easy swapping of components. I have included all the experimentations that I could think of, and we can easily switch between them to see what works best, or even execute multiple options in parallel and merge the results:
- **ParserFactory**: Switch between `pdfplumber`, `pymupdf`, and `unstructured` parsers, we can choose one to see what works better for our format or pattern of documents.
- **ChunkingFactory**: Choose from `hierarchical`, `sentence`, `fixed_size`, `semantic`, or `sliding_window` chunking. We can again choose one based on what works based for maybe different kind of documents. 
- **EmbeddingFactory**: Currently supports `nomic_embed` (Ollama-based), it has shown better results among open-source models.
- **VectorDBFactory**: Switch between `qdrant`, `chroma`, and `faiss` backends
- **QueryFactory**: Support for `rag`, `direct_retrieval`, `multi_query`, and `reranking` strategies

This design makes experimentation straightforward—change the type in `config/settings.yaml` and restart. No code changes needed.

### Document Parsing

**Parser Selection:**
- **pdfplumber** (default): Good balance of accuracy and speed for most documents
- **pymupdf**: Faster for simple documents, occasionally misses complex layouts
- **unstructured**: Best for complex layouts but requires additional dependencies

All parsers extract both text and tables. Tables are normalized into a consistent format and stored separately with rich metadata (headers, row counts, page numbers).

**Text Cleaning:**
A custom `TextCleaner` utility post-processes extracted text to:
- Remove PDF extraction artifacts (stray characters, formatting noise)
- Improve semantic search accuracy by making key information more prominent

**Chunking Strategy:**
- **Hierarchical** (default): Respects document structure (sections, paragraphs), best for long documents
- **Sentence-based**: Good for question-answering tasks
- **Table-specific**: Tables are chunked separately with enhanced formatting—headers, section titles, and column names are repeated multiple times to improve semantic matching

Tables get special treatment: row counts are explicit, headers are emphasized, and rows are formatted as "ColumnName: value" to improve LLM comprehension for queries like "how many X were made?"

### Embedding Generation
**Nomic Embed via Ollama:**
- **Why Nomic**: Small model size (~274MB), runs efficiently on CPU, produces good quality embeddings (768 dimensions)
- **Implementation**: Direct HTTP API calls to Ollama's `/api/embed` endpoint for batch processing
- **Adaptive Batching**: Automatically splits large batches if Ollama returns errors, ensuring even very large documents are processed
- **Retry Logic**: Built-in HTTP retry decorator with exponential backoff handles transient failures

The system uses batch embedding by default (configurable batch size) for efficiency, with automatic fallback to smaller batches on errors.-=

### Vector Database

**Qdrant Selection:**
- **Why Qdrant over FAISS**: Qdrant supports rich metadata filtering (e.g., filter by `document_id`, `chunk_type`). FAISS is in-process only and doesn't support metadata queries well.
- **Why Qdrant over Chroma**: Qdrant handles list-valued metadata (like table headers) without issues. Chroma only accepts scalar metadata values.
- **Features Used**: Metadata filtering for per-document isolation, chunk type filtering (text vs. table), distance-based similarity search

**Search Strategy:**
The query process performs parallel searches for text and table chunks separately, applies similarity thresholds to each, takes top-k from each type, then merges results. This ensures both narrative text and structured table data are considered when answering questions.

### Query Answering

**Mistral 7B via Ollama:**
- **Why Mistral**: Good balance of capability and size, runs well on consumer hardware, produces coherent answers
- **Configuration**: Temperature set to 0.0 for deterministic, factual responses

**RAG Strategy:**
The default RAG strategy:
1. Embeds the user query
2. Searches vector database for relevant chunks (text and tables separately)
3. Filters by similarity threshold
4. Constructs a prompt with context chunks
5. Sends to Mistral for answer generation

**Prompt Engineering:**
The prompt explicitly instructs the LLM to extract exact values when information is clearly stated (e.g., "VANTAGESCORE 3.0: 609" → answer should be "609"). This reduces hallucination and improves accuracy for factual queries.

**Gatekeeping:**
Optional relevance checking ensures queries are actually related to the document content before processing, preventing irrelevant answers.

---

## Sample Input/Output

### 1. Upload a Document

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@credit_report.pdf"
```

**Response:**
```json
{
  "document_id": "556eae50-8437-4191-ada5-a48d47f39ec3",
  "status": "processing",
  "message": "Document uploaded successfully"
}
```

### 2. Check Document Status

**Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/documents/556eae50-8437-4191-ada5-a48d47f39ec3/status"
```

**Response:**
```json
{
  "document_id": "556eae50-8437-4191-ada5-a48d47f39ec3",
  "filename": "credit_report.pdf",
  "status": "completed",
  "created_at": "2024-01-07T10:30:00",
  "updated_at": "2024-01-07T10:32:15",
  "metadata": {
    "total_pages": 18,
    "total_tables": 26,
    "parser": "pdfplumber"
  }
}
```

### 3. Query the Document

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/documents/556eae50-8437-4191-ada5-a48d47f39ec3/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is my VANTAGESCORE?"
  }'
```

**Response:**
```json
{
  "answer": "609",
  "sources": [
    {
      "chunk_id": "dbc6e55e-fb26-470a-ba89-de8009d60564",
      "text": "VANTAGESCORE 3.0: 609",
      "metadata": {
        "chunk_type": "text",
        "chunk_index": 86,
        "document_id": "556eae50-8437-4191-ada5-a48d47f39ec3"
      }
    }
  ],
  "confidence": 1.0,
  "metadata": {
    "strategy": "rag",
    "num_sources": 1
  }
}
```

**Another Query Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/documents/556eae50-8437-4191-ada5-a48d47f39ec3/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many Credit Inquiries were made?"
  }'
```

**Response:**
```json
{
  "answer": "2 credit inquiries were made. The dates are 1/3/2021 (Super Lender Usa) and 1/5/2021 (Cool Lender).",
  "sources": [
    {
      "chunk_id": "27498908-90ca-4ffa-9742-5955e248a147",
      "text": "Section: Credit Inquiries\nTable: Credit Inquiries\n\nTable: Credit Inquiries\n\nTable with columns: Date of Inquiry, Member Name\nColumn names: Date of Inquiry, Member Name\nThis table contains 2 row(s) of data:\n\nRow 1: Date of Inquiry: 1/3/2021, Member Name: Super Lender Usa\nRow 2: Date of Inquiry: 1/5/2021, Member Name: Cool Lender\n\nTotal: 2 row(s) in this Credit Inquiries table with columns: Date of Inquiry, Member Name.",
      "metadata": {
        "chunk_type": "table",
        "table_index": 23,
        "num_rows": 2,
        "has_headers": true
      }
    }
  ],
  "confidence": 0.8,
  "metadata": {
    "strategy": "rag",
    "num_sources": 1
  }
}
```

### 4. Get All Chunks (Debugging)

**Request:**
```bash
# Get all chunks
curl -X GET "http://localhost:8000/api/v1/documents/556eae50-8437-4191-ada5-a48d47f39ec3/chunks"

# Get only table chunks
curl -X GET "http://localhost:8000/api/v1/documents/556eae50-8437-4191-ada5-a48d47f39ec3/chunks?chunk_type=table"

# Get only text chunks
curl -X GET "http://localhost:8000/api/v1/documents/556eae50-8437-4191-ada5-a48d47f39ec3/chunks?chunk_type=text"
```

### 5. Get Document Info

**Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/documents/556eae50-8437-4191-ada5-a48d47f39ec3"
```

**Response:**
```json
{
  "document_id": "556eae50-8437-4191-ada5-a48d47f39ec3",
  "filename": "credit_report.pdf",
  "status": "completed",
  "query_count": 5,
  "created_at": "2024-01-07T10:30:00",
  "updated_at": "2024-01-07T10:35:22"
}
```

### 6. Delete Document

**Request:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/556eae50-8437-4191-ada5-a48d47f39ec3"
```

**Response:** `204 No Content`

### 7. Health Check

**Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "ollama": "healthy"
  }
}
```

---

## Configuration

Key settings in `config/settings.yaml`:

- **PDF Parser**: Set `pdf_parser.type` to `pdfplumber`, `pymupdf`, or `unstructured`
- **Chunking**: Set `chunking.strategy` and adjust `chunk_size` and `chunk_overlap`
- **Embeddings**: Configure `embeddings.batch_size` and `api_url` for Ollama
- **Vector DB**: Set `vector_db.type`, `host`, `port`, and `collection_name`
- **Query**: Adjust `query_answering.top_k`, `similarity_threshold`, and LLM settings
- **Gatekeeping**: Toggle `gatekeeping.enabled` and adjust `threshold`

---

## Troubleshooting

### Ollama Connection Issues

**Error**: `Connection refused` or `nodename nor servname provided`

**Solutions:**
- Ensure Ollama container is running: `docker ps | grep ollama`
- Check if Ollama is accessible: `curl http://localhost:11434/api/tags`
- For local setup, verify `api_url` in `settings.yaml` is `http://localhost:11434`
- For Docker setup, verify it's `http://ollama:11434`

### Model Not Found

**Error**: Model not loaded in Ollama

**Solution:**
```bash
docker exec -it document-qa-ollama ollama list
# If models missing, pull them:
docker exec -it document-qa-ollama ollama pull mistral:7b-instruct
docker exec -it document-qa-ollama ollama pull nomic-embed-text
```

### Qdrant Connection Issues

**Error**: `nodename nor servname provided` when connecting to Qdrant

**Solutions:**
- Ensure Qdrant container is running: `docker ps | grep qdrant`
- For local setup, set `vector_db.host` to `localhost` in `settings.yaml`
- For Docker setup, set it to `qdrant`
- Check Qdrant dashboard: `http://localhost:6333`

### Large Document Processing Fails

**Error**: Embedding batch size errors or timeouts

**Solutions:**
- Reduce `embeddings.batch_size` in `settings.yaml` (default: 8)
- System will auto-split batches, but smaller initial batch helps
- Check Ollama logs: `docker logs document-qa-ollama`

### Low Quality Answers

**Solutions:**
- Adjust `query_answering.similarity_threshold` (lower = more results, higher = stricter)
- Increase `query_answering.top_k` to include more context
- Check retrieved chunks via `/documents/{id}/chunks` endpoint
- Try different chunking strategies

---

## Logging

Logs are written to `logs/app.log` in JSON format for easy parsing. Log levels can be adjusted in `config/settings.yaml`:

```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "json"
  file: "./logs/app.log"
```

To view logs:
```bash
# Local setup
tail -f logs/app.log

# Docker setup
docker logs -f document-qa-api
```

---

## Development Tips

### Switching Parsers

Change `pdf_parser.type` in `settings.yaml` and restart. No code changes needed.

### Testing Different Chunking Strategies

Update `chunking.strategy` in `settings.yaml`. Available options:
- `hierarchical` (default)
- `sentence`
- `fixed_size`
- `semantic` (requires embedder)
- `sliding_window`

### Debugging Queries

1. Check document status: `GET /documents/{id}/status`
2. View all chunks: `GET /documents/{id}/chunks`
3. View specific chunk types: `GET /documents/{id}/chunks?chunk_type=table`
4. Check logs for embedding/search details

### Running Tests

```bash
pytest tests/
```

---

## Additional Notes

### Experimentation Opportunities

Given more time, I would explore:
- **Better table parsing**: Some complex tables (especially with merged cells) could benefit from more sophisticated extraction
- **Alternative chunking**: Testing semantic chunking with embeddings to find natural boundaries
- **Query expansion**: Implementing the multi-query strategy more thoroughly to handle query variations
- **Re-ranking**: Fine-tuning reranking to improve result quality

### Scaling Considerations

**Multiple Documents:**
- Current design supports multiple documents via metadata filtering in Qdrant
- Consider document-level collections if strict isolation is needed
- Batch processing could be added for bulk uploads

**Large Documents:**
- Hierarchical chunking already handles long documents well
- Adaptive batch splitting for embeddings prevents memory issues
- For extremely large documents, consider streaming processing or document splitting strategies

**Performance:**
- Qdrant can be scaled horizontally
- Ollama supports GPU acceleration for faster inference
- Consider caching frequently accessed document embeddings

### Limitations

- **Complex layouts**: Very complex PDFs with non-standard layouts may not parse perfectly
- **Image-based PDFs**: OCR not currently implemented (would require Tesseract or similar)
- **Multi-language**: Optimized for English; other languages may need different embedding models
- **Real-time**: Current design is request-response; real-time streaming would require architecture changes

---

## Project Structure

```
document-query-answerer/
├── app/
│   ├── api/              # FastAPI routes and schemas
│   ├── chunking/         # Chunking strategies and table chunker
│   ├── core/             # Configuration and database
│   ├── embeddings/       # Embedding generation
│   ├── gatekeeping/      # Relevance checking
│   ├── parsers/          # PDF parsers and text cleaner
│   ├── query/            # Query strategies and LLM client
│   ├── services/         # Business logic
│   ├── utils/            # Utilities (retry, validators, etc.)
│   └── vector_db/        # Vector database implementations
├── config/               # Configuration files
├── data/                 # Data storage (SQLite, Qdrant, uploads)
├── tests/                # Test suite
└── requirements.txt      # Python dependencies
```

---

## License

MIT
