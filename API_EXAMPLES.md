# API cURL Examples

Base URL: `http://localhost:8000`

## 1. Health Check

Check if the API and services are healthy.

```bash
curl -X GET "http://localhost:8000/health" \
  -H "Content-Type: application/json"
```

**Expected Response:**
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

## 2. Upload Document

Upload a PDF document for processing.

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf"
```

**Example with a file:**
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@./Senior\ Python\ Developer\ Assignment/Input\ Docs/credit_report.pdf"
```

**Expected Response:**
```json
{
  "document_id": "abc123-def456-ghi789",
  "status": "processing",
  "message": "Document uploaded successfully"
}
```

**Save the `document_id` for subsequent requests!**

---

## 3. Query Document

Ask a question about an uploaded document.

```bash
curl -X POST "http://localhost:8000/api/v1/documents/{document_id}/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the total outstanding balance?"
  }'
```

**Example with actual document_id:**
```bash
curl -X POST "http://localhost:8000/api/v1/documents/abc123-def456-ghi789/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the total outstanding balance?"
  }'
```

**More example queries:**
```bash
# Financial document query
curl -X POST "http://localhost:8000/api/v1/documents/abc123-def456-ghi789/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key financial metrics mentioned in this document?"
  }'

# Credit report query
curl -X POST "http://localhost:8000/api/v1/documents/abc123-def456-ghi789/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the credit score and what factors affect it?"
  }'
```

**Expected Response:**
```json
{
  "answer": "The total outstanding balance is $15,234.56...",
  "sources": [
    {
      "chunk_id": "chunk_001",
      "text": "Outstanding Balance: $15,234.56...",
      "metadata": {
        "page": 3,
        "section": "Financial Summary"
      }
    }
  ],
  "confidence": 0.85,
  "metadata": {
    "query_time": 1.23,
    "chunks_retrieved": 5
  }
}
```

---

## 4. Get Document Status

Check the processing status of an uploaded document.

```bash
curl -X GET "http://localhost:8000/api/v1/documents/{document_id}/status" \
  -H "Content-Type: application/json"
```

**Example:**
```bash
curl -X GET "http://localhost:8000/api/v1/documents/abc123-def456-ghi789/status" \
  -H "Content-Type: application/json"
```

**Expected Response:**
```json
{
  "document_id": "abc123-def456-ghi789",
  "filename": "credit_report.pdf",
  "status": "completed",
  "created_at": "2024-01-15T10:30:00",
  "updated_at": "2024-01-15T10:32:15",
  "metadata": {
    "pages": 10,
    "file_size": 245678,
    "processing_time": 2.15
  }
}
```

**Status values:** `processing`, `completed`, `failed`

---

## 5. Get Document Info

Get detailed information about a document including query count.

```bash
curl -X GET "http://localhost:8000/api/v1/documents/{document_id}" \
  -H "Content-Type: application/json"
```

**Example:**
```bash
curl -X GET "http://localhost:8000/api/v1/documents/abc123-def456-ghi789" \
  -H "Content-Type: application/json"
```

**Expected Response:**
```json
{
  "document_id": "abc123-def456-ghi789",
  "filename": "credit_report.pdf",
  "status": "completed",
  "created_at": "2024-01-15T10:30:00",
  "updated_at": "2024-01-15T10:32:15",
  "metadata": {
    "pages": 10,
    "file_size": 245678,
    "processing_time": 2.15
  },
  "query_count": 3
}
```

---

## 6. Delete Document

Delete a document and all its associated data.

```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/{document_id}" \
  -H "Content-Type: application/json"
```

**Example:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/abc123-def456-ghi789" \
  -H "Content-Type: application/json"
```

**Expected Response:**
- Status Code: `204 No Content`
- No response body

---

## Complete Workflow Example

```bash
# 1. Check health
curl -X GET "http://localhost:8000/health"

# 2. Upload a document
DOCUMENT_ID=$(curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@./credit_report.pdf" | jq -r '.document_id')

echo "Document ID: $DOCUMENT_ID"

# 3. Wait a moment for processing, then check status
sleep 5
curl -X GET "http://localhost:8000/api/v1/documents/$DOCUMENT_ID/status"

# 4. Query the document
curl -X POST "http://localhost:8000/api/v1/documents/$DOCUMENT_ID/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of this document?"
  }'

# 5. Get document info
curl -X GET "http://localhost:8000/api/v1/documents/$DOCUMENT_ID"

# 6. Clean up (optional)
curl -X DELETE "http://localhost:8000/api/v1/documents/$DOCUMENT_ID"
```

---

## Error Responses

### 400 Bad Request
```json
{
  "error": "Validation error",
  "detail": "Query text must be between 1 and 10000 characters"
}
```

### 404 Not Found
```json
{
  "error": "Document not found",
  "detail": "Document with ID 'abc123' not found"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error",
  "detail": "Failed to process query"
}
```

---

## Notes

- **Query field constraints:** Must be between 1 and 10,000 characters (per `QueryRequest` schema)
- **File upload:** Only PDF files are accepted
- **File size limit:** Default is 50MB (configurable in settings)
- **Document processing:** Documents are processed asynchronously; check status endpoint to see when ready
- **Model requirement:** Make sure `mistral:7b-instruct` is pulled in Ollama before querying documents
