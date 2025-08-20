# API Documentation

## Overview

The Advanced Document Q&A System provides a RESTful API built with FastAPI that processes documents and answers questions using a sophisticated RAG (Retrieval-Augmented Generation) pipeline.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API accepts Bearer tokens but does not validate them. This is designed for the hackathon environment.

```http
Authorization: Bearer your_token_here
```

## Endpoints

### 1. Main Processing Endpoint

#### `POST /hackrx/run`

Processes documents from URLs and answers questions based on their content.

**Description:**
This is the primary endpoint that implements the complete RAG pipeline. It downloads documents, processes them into chunks, creates vector embeddings, and uses OpenAI's language model to answer questions.

**Request Format:**

```http
POST /hackrx/run HTTP/1.1
Content-Type: application/json
Authorization: Bearer optional_token

{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the main topic of this document?",
    "Who are the key stakeholders mentioned?",
    "What are the important dates or deadlines?"
  ]
}
```

**Request Schema:**

```json
{
  "type": "object",
  "properties": {
    "documents": {
      "type": "string",
      "description": "URL of the document to process",
      "format": "uri",
      "examples": [
        "https://example.com/policy.pdf",
        "https://drive.google.com/file/d/xyz/view",
        "https://company.com/manual.docx"
      ]
    },
    "questions": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "List of questions to answer based on the document",
      "minItems": 1,
      "maxItems": 10,
      "examples": [
        "What is the coverage amount?",
        "What are the eligibility criteria?",
        "How do I file a claim?"
      ]
    }
  },
  "required": ["documents", "questions"]
}
```

**Response Format:**

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "answers": [
    "The main topic of this document is the comprehensive health insurance policy...",
    "The key stakeholders mentioned include policyholders, insurance providers, and healthcare facilities...",
    "Important dates include the policy effective date of January 1, 2024, and claim filing deadline of 30 days..."
  ]
}
```

**Response Schema:**

```json
{
  "type": "object",
  "properties": {
    "answers": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Corresponding answers to the input questions in the same order"
    }
  },
  "required": ["answers"]
}
```

**Response Characteristics:**
- Answers are returned in the same order as the input questions
- Each answer is concise (typically 1-2 sentences, max 200 characters)
- Answers are based on the document content and context
- If information is not available, returns "Unable to determine from provided context."

### 2. Health Check Endpoint

#### `GET /`

Basic health check endpoint.

**Request:**
```http
GET / HTTP/1.1
```

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "message": "Advanced Document Q&A System is running"
}
```

### 3. API Documentation Endpoints

#### `GET /docs`

Interactive Swagger UI documentation.

#### `GET /redoc`

Alternative ReDoc documentation interface.

#### `GET /openapi.json`

OpenAPI specification in JSON format.

## Supported Document Formats

| Format | Extension | Description | Notes |
|--------|-----------|-------------|-------|
| PDF | `.pdf` | Portable Document Format | Most common format, supports text extraction |
| Word Document | `.docx` | Microsoft Word Open XML | Modern Word documents |
| Email | `.eml` | Email Message Format | Email files with attachments |

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages.

### Status Codes

| Code | Description | Scenario |
|------|-------------|----------|
| `200` | Success | Request processed successfully |
| `400` | Bad Request | Invalid input data or document processing error |
| `422` | Unprocessable Entity | Request validation failed |
| `500` | Internal Server Error | Unexpected server error |

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Examples

#### Document Processing Error
```http
HTTP/1.1 400 Bad Request
Content-Type: application/json

{
  "detail": "Error: Could not process the provided document."
}
```

#### Validation Error
```http
HTTP/1.1 422 Unprocessable Entity
Content-Type: application/json

{
  "detail": [
    {
      "loc": ["body", "questions"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

#### Server Error
```http
HTTP/1.1 500 Internal Server Error
Content-Type: application/json

{
  "detail": "Error processing document: OpenAI API rate limit exceeded"
}
```

## Request Examples

### cURL Examples

#### Basic Request
```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer optional_token" \
  -d '{
    "documents": "https://example.com/insurance-policy.pdf",
    "questions": ["What is the coverage limit?"]
  }'
```

#### Multiple Questions
```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/employee-handbook.pdf",
    "questions": [
      "What is the vacation policy?",
      "What are the working hours?",
      "How do I request time off?"
    ]
  }'
```

### Python Examples

#### Using requests library
```python
import requests

url = "http://localhost:8000/hackrx/run"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer optional_token"
}
data = {
    "documents": "https://example.com/policy.pdf",
    "questions": [
        "What are the key benefits?",
        "What are the limitations?"
    ]
}

response = requests.post(url, json=data, headers=headers)
result = response.json()

print("Answers:")
for i, answer in enumerate(result["answers"]):
    print(f"{i+1}. {answer}")
```

#### Using aiohttp (async)
```python
import aiohttp
import asyncio

async def query_document():
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/hackrx/run",
            json={
                "documents": "https://example.com/manual.pdf",
                "questions": ["What is the main purpose?"]
            }
        ) as response:
            return await response.json()

result = asyncio.run(query_document())
print(result["answers"][0])
```

### JavaScript Examples

#### Using fetch API
```javascript
const response = await fetch('http://localhost:8000/hackrx/run', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer optional_token'
  },
  body: JSON.stringify({
    documents: 'https://example.com/document.pdf',
    questions: [
      'What is this document about?',
      'Who is the target audience?'
    ]
  })
});

const result = await response.json();
console.log('Answers:', result.answers);
```

#### Using axios
```javascript
const axios = require('axios');

const response = await axios.post('http://localhost:8000/hackrx/run', {
  documents: 'https://example.com/guide.pdf',
  questions: ['What are the main features?']
}, {
  headers: {
    'Authorization': 'Bearer optional_token'
  }
});

console.log('Answer:', response.data.answers[0]);
```

## Performance Considerations

### Request Optimization

1. **Document Size**: Large documents (>10MB) take longer to process but are handled automatically
2. **Question Count**: Processing time scales with the number of questions, but parallel processing optimizes this
3. **Question Complexity**: More complex questions may require additional processing time

### Response Times

| Scenario | Typical Response Time |
|----------|----------------------|
| Small document (<1MB), 1 question | 5-10 seconds |
| Medium document (1-5MB), 3 questions | 10-15 seconds |
| Large document (>5MB), 5+ questions | 20-40 seconds |

### Rate Limiting

The API inherits OpenAI's rate limits:
- **Requests per minute**: Varies by OpenAI plan
- **Tokens per minute**: Varies by OpenAI plan
- **Concurrent requests**: Limited by the system's max_workers setting

### Best Practices

1. **Batch Questions**: Send multiple questions in a single request for better performance
2. **Document Caching**: The same document URL is cached, so subsequent requests are faster
3. **Question Optimization**: Be specific and clear in your questions for better answers
4. **Error Handling**: Implement retry logic for transient errors

## Data Models

### HackathonInput

```python
class HackathonInput(BaseModel):
    documents: str = Field(description="URL of the document to process")
    questions: List[str] = Field(description="List of questions to answer")
```

### HackathonOutput

```python
class HackathonOutput(BaseModel):
    answers: List[str] = Field(description="List of answers corresponding to the questions")
```

### Internal Models

#### ParsedQuery
```python
class ParsedQuery(BaseModel):
    procedure: Optional[str] = Field(None, description="Main procedure or activity")
    entities: List[str] = Field(description="Key entities and terms")
```

#### FinalAnswer
```python
class FinalAnswer(BaseModel):
    answer: str = Field(description="Direct, concise answer to the question")
```

## Security Considerations

### Current Implementation
- Bearer token acceptance without validation (hackathon environment)
- No rate limiting beyond OpenAI's limits
- Document URLs are accessed directly

### Production Recommendations
- Implement proper authentication and authorization
- Add rate limiting and request throttling
- Validate and sanitize document URLs
- Add input validation and sanitization
- Implement audit logging
- Add CORS configuration as needed

## Integration Examples

### Web Application Integration

```html
<!DOCTYPE html>
<html>
<head>
    <title>Document Q&A</title>
</head>
<body>
    <div id="app">
        <input type="url" id="docUrl" placeholder="Document URL">
        <textarea id="questions" placeholder="Enter questions (one per line)"></textarea>
        <button onclick="processDocument()">Get Answers</button>
        <div id="answers"></div>
    </div>

    <script>
    async function processDocument() {
        const docUrl = document.getElementById('docUrl').value;
        const questions = document.getElementById('questions').value
            .split('\n').filter(q => q.trim());
        
        const response = await fetch('/hackrx/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ documents: docUrl, questions })
        });
        
        const result = await response.json();
        document.getElementById('answers').innerHTML = 
            result.answers.map((answer, i) => 
                `<p><strong>Q:</strong> ${questions[i]}<br>
                 <strong>A:</strong> ${answer}</p>`
            ).join('');
    }
    </script>
</body>
</html>
```

### Mobile App Integration (React Native)

```javascript
import React, { useState } from 'react';
import { View, TextInput, TouchableOpacity, Text } from 'react-native';

const DocumentQA = () => {
  const [docUrl, setDocUrl] = useState('');
  const [questions, setQuestions] = useState('');
  const [answers, setAnswers] = useState([]);

  const processDocument = async () => {
    try {
      const response = await fetch('http://your-api-server.com/hackrx/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          documents: docUrl,
          questions: questions.split('\n').filter(q => q.trim())
        })
      });
      
      const result = await response.json();
      setAnswers(result.answers);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <View>
      <TextInput 
        value={docUrl}
        onChangeText={setDocUrl}
        placeholder="Document URL"
      />
      <TextInput 
        value={questions}
        onChangeText={setQuestions}
        placeholder="Questions (one per line)"
        multiline
      />
      <TouchableOpacity onPress={processDocument}>
        <Text>Get Answers</Text>
      </TouchableOpacity>
      {answers.map((answer, index) => (
        <Text key={index}>{answer}</Text>
      ))}
    </View>
  );
};
```

## Testing the API

### Unit Tests Example

```python
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_hackrx_run_endpoint():
    response = client.post(
        "/hackrx/run",
        json={
            "documents": "https://example.com/test-doc.pdf",
            "questions": ["What is this document about?"]
        }
    )
    assert response.status_code == 200
    assert "answers" in response.json()
    assert len(response.json()["answers"]) == 1

def test_invalid_request():
    response = client.post(
        "/hackrx/run",
        json={"documents": "invalid-url"}
    )
    assert response.status_code == 422
```

### Load Testing Example

```python
import asyncio
import aiohttp
import time

async def test_concurrent_requests():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(10):
            task = session.post(
                "http://localhost:8000/hackrx/run",
                json={
                    "documents": "https://example.com/test.pdf",
                    "questions": [f"Question {i}?"]
                }
            )
            tasks.append(task)
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        print(f"Processed 10 requests in {end_time - start_time:.2f} seconds")
        
        for response in responses:
            assert response.status == 200

# Run the test
asyncio.run(test_concurrent_requests())
```

## Troubleshooting API Issues

### Common Problems and Solutions

#### 1. Connection Refused
```
ConnectionError: [Errno 111] Connection refused
```
**Solution**: Ensure the server is running on the correct host and port.

#### 2. Timeout Errors
```
TimeoutError: Request timed out
```
**Solutions**:
- Increase client timeout for large documents
- Check document URL accessibility
- Verify OpenAI API connectivity

#### 3. Invalid Document URL
```
HTTPException: Error: Could not process the provided document
```
**Solutions**:
- Verify the URL is accessible and returns the document
- Check the document format is supported (PDF, DOCX, EML)
- Ensure the URL doesn't require authentication

#### 4. OpenAI API Errors
```
OpenAI API error: Rate limit exceeded
```
**Solutions**:
- Check your OpenAI API quota and billing
- Implement exponential backoff for retries
- Reduce the number of concurrent requests

### Debug Headers

Add these headers to get more information:

```http
X-Debug-Mode: true
X-Request-ID: unique-request-id
```

---

For more information, see the main [README.md](./README.md) and [ARCHITECTURE.md](./ARCHITECTURE.md).