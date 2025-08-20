# Usage Examples and Testing Guide

This document provides comprehensive examples of how to use the Advanced Document Q&A System in various scenarios, along with testing strategies and troubleshooting tips.

## Table of Contents

- [Basic Usage Examples](#basic-usage-examples)
- [Advanced Usage Patterns](#advanced-usage-patterns)
- [Integration Examples](#integration-examples)
- [Testing Strategies](#testing-strategies)
- [Performance Optimization](#performance-optimization)
- [Common Use Cases](#common-use-cases)
- [Troubleshooting Examples](#troubleshooting-examples)

## Basic Usage Examples

### 1. Simple Single Question

**Scenario**: Ask one question about a document

```python
import requests
import json

# Basic API call
url = "http://localhost:8000/hackrx/run"
payload = {
    "documents": "https://example.com/employee-handbook.pdf",
    "questions": ["What is the vacation policy?"]
}

response = requests.post(url, json=payload)
result = response.json()

print("Question:", payload["questions"][0])
print("Answer:", result["answers"][0])
```

**Expected Output**:
```
Question: What is the vacation policy?
Answer: Employees are entitled to 15 days of paid vacation annually, accruing at 1.25 days per month.
```

### 2. Multiple Questions on Same Document

**Scenario**: Ask several related questions efficiently

```python
import requests

# Multiple questions for comprehensive analysis
payload = {
    "documents": "https://example.com/insurance-policy.pdf",
    "questions": [
        "What is the maximum coverage amount?",
        "What are the main exclusions?", 
        "How do I file a claim?",
        "What is the deductible amount?",
        "Who do I contact for customer service?"
    ]
}

response = requests.post("http://localhost:8000/hackrx/run", json=payload)
result = response.json()

# Display Q&A pairs
for i, (question, answer) in enumerate(zip(payload["questions"], result["answers"])):
    print(f"\nQ{i+1}: {question}")
    print(f"A{i+1}: {answer}")
```

**Expected Output**:
```
Q1: What is the maximum coverage amount?
A1: The maximum coverage amount is $500,000 per incident with an annual aggregate limit of $1,000,000.

Q2: What are the main exclusions?
A2: Main exclusions include pre-existing conditions, experimental treatments, and cosmetic procedures.

Q3: How do I file a claim?
A3: Claims can be filed online at the customer portal or by calling 1-800-CLAIMS within 30 days.

Q4: What is the deductible amount?
A4: The deductible is $500 per incident for in-network providers and $1,000 for out-of-network.

Q5: Who do I contact for customer service?
A5: Customer service is available 24/7 at 1-800-SUPPORT or support@insurance-company.com.
```

### 3. Error Handling Example

**Scenario**: Handle various error conditions gracefully

```python
import requests
from requests.exceptions import RequestException

def safe_document_query(document_url, questions):
    """
    Safely query documents with comprehensive error handling.
    """
    try:
        response = requests.post(
            "http://localhost:8000/hackrx/run",
            json={
                "documents": document_url,
                "questions": questions
            },
            timeout=60  # 1 minute timeout for large documents
        )
        
        # Check HTTP status
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        elif response.status_code == 400:
            return {"success": False, "error": "Document processing failed", "details": response.json()}
        elif response.status_code == 422:
            return {"success": False, "error": "Invalid input format", "details": response.json()}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}", "details": response.text}
            
    except RequestException as e:
        return {"success": False, "error": "Network error", "details": str(e)}
    except Exception as e:
        return {"success": False, "error": "Unexpected error", "details": str(e)}

# Example usage with error handling
result = safe_document_query(
    "https://example.com/document.pdf",
    ["What is this document about?"]
)

if result["success"]:
    print("Answers:", result["data"]["answers"])
else:
    print(f"Error: {result['error']}")
    print(f"Details: {result['details']}")
```

## Advanced Usage Patterns

### 1. Batch Document Processing

**Scenario**: Process multiple documents with different questions

```python
import asyncio
import aiohttp
from typing import List, Dict, Any

async def process_multiple_documents(document_queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process multiple documents concurrently for improved performance.
    
    Args:
        document_queries: List of dicts with 'url' and 'questions' keys
        
    Returns:
        List of results with answers and metadata
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for i, query in enumerate(document_queries):
            task = process_single_document(session, query, i)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

async def process_single_document(session, query, index):
    """Process a single document asynchronously."""
    try:
        async with session.post(
            "http://localhost:8000/hackrx/run",
            json={
                "documents": query["url"],
                "questions": query["questions"]
            },
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "index": index,
                    "url": query["url"],
                    "success": True,
                    "answers": data["answers"],
                    "questions": query["questions"]
                }
            else:
                return {
                    "index": index,
                    "url": query["url"], 
                    "success": False,
                    "error": f"HTTP {response.status}",
                    "details": await response.text()
                }
    except Exception as e:
        return {
            "index": index,
            "url": query["url"],
            "success": False,
            "error": "Processing error",
            "details": str(e)
        }

# Example usage
document_queries = [
    {
        "url": "https://example.com/policy1.pdf",
        "questions": ["What is the coverage?", "What are the terms?"]
    },
    {
        "url": "https://example.com/manual.docx", 
        "questions": ["How do I install?", "What are the requirements?"]
    },
    {
        "url": "https://example.com/contract.pdf",
        "questions": ["What is the duration?", "What are the obligations?"]
    }
]

# Run batch processing
results = asyncio.run(process_multiple_documents(document_queries))

# Display results
for result in results:
    print(f"\nDocument {result['index'] + 1}: {result['url']}")
    if result["success"]:
        for q, a in zip(result["questions"], result["answers"]):
            print(f"  Q: {q}")
            print(f"  A: {a}")
    else:
        print(f"  Error: {result['error']}")
```

### 2. Question Quality Analysis

**Scenario**: Analyze and improve question quality before processing

```python
import re
from typing import List, Tuple

def analyze_question_quality(questions: List[str]) -> List[Tuple[str, str, List[str]]]:
    """
    Analyze question quality and provide improvement suggestions.
    
    Returns:
        List of tuples: (question, quality_score, suggestions)
    """
    results = []
    
    for question in questions:
        score, suggestions = evaluate_single_question(question)
        results.append((question, score, suggestions))
    
    return results

def evaluate_single_question(question: str) -> Tuple[str, List[str]]:
    """Evaluate a single question's quality."""
    suggestions = []
    
    # Check length
    if len(question.strip()) < 5:
        suggestions.append("Question is too short - add more context")
    elif len(question.strip()) > 200:
        suggestions.append("Question is too long - try to be more concise")
    
    # Check for question words
    question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which', 'can', 'is', 'are', 'do', 'does']
    if not any(word in question.lower() for word in question_words) and not question.strip().endswith('?'):
        suggestions.append("Consider adding question words (what, how, when, etc.) or ending with '?'")
    
    # Check specificity
    vague_terms = ['stuff', 'things', 'something', 'anything', 'everything']
    if any(term in question.lower() for term in vague_terms):
        suggestions.append("Try to be more specific - avoid vague terms")
    
    # Check for multiple questions
    if question.count('?') > 1:
        suggestions.append("Break into separate questions for better results")
    
    # Determine quality score
    if not suggestions:
        quality = "Good"
    elif len(suggestions) <= 2:
        quality = "Fair"
    else:
        quality = "Poor"
    
    return quality, suggestions

# Example usage
questions = [
    "What is the coverage amount?",  # Good question
    "stuff?",  # Poor question
    "How do I file a claim and what documents do I need and how long does it take?",  # Multiple questions
    "Tell me everything about this policy",  # Missing question format
    "What are the specific eligibility requirements for coverage?"  # Good question
]

analysis = analyze_question_quality(questions)

print("Question Quality Analysis:")
print("=" * 50)

for question, quality, suggestions in analysis:
    print(f"\nQuestion: {question}")
    print(f"Quality: {quality}")
    if suggestions:
        print("Suggestions:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")
    else:
        print("✓ Question looks good!")
```

### 3. Response Quality Enhancement

**Scenario**: Post-process responses for better quality

```python
import re
from typing import List

def enhance_responses(questions: List[str], answers: List[str]) -> List[dict]:
    """
    Enhance responses with additional context and formatting.
    
    Returns:
        List of enhanced response objects
    """
    enhanced = []
    
    for question, answer in zip(questions, answers):
        enhanced_response = {
            "original_question": question,
            "original_answer": answer,
            "enhanced_answer": enhance_single_answer(answer),
            "confidence": estimate_confidence(answer),
            "category": categorize_question(question),
            "follow_up_suggestions": generate_follow_ups(question, answer)
        }
        enhanced.append(enhanced_response)
    
    return enhanced

def enhance_single_answer(answer: str) -> str:
    """Enhance a single answer with better formatting."""
    # Clean up the answer
    enhanced = answer.strip()
    
    # Ensure proper capitalization
    if enhanced and not enhanced[0].isupper():
        enhanced = enhanced[0].upper() + enhanced[1:]
    
    # Ensure proper ending punctuation
    if enhanced and enhanced[-1] not in '.!?':
        enhanced += '.'
    
    # Format numbers and currency
    enhanced = re.sub(r'\$(\d+)', r'$\1', enhanced)  # Format currency
    enhanced = re.sub(r'(\d+)%', r'\1%', enhanced)   # Format percentages
    
    return enhanced

def estimate_confidence(answer: str) -> str:
    """Estimate confidence level based on answer characteristics."""
    # Low confidence indicators
    uncertainty_words = ['may', 'might', 'possibly', 'unclear', 'unable to determine']
    if any(word in answer.lower() for word in uncertainty_words):
        return "Low"
    
    # High confidence indicators
    certainty_words = ['specifically', 'exactly', 'precisely', '$', '%', 'must', 'required']
    if any(word in answer.lower() for word in certainty_words):
        return "High"
    
    return "Medium"

def categorize_question(question: str) -> str:
    """Categorize the type of question."""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['amount', 'cost', 'price', '$', 'fee']):
        return "Financial"
    elif any(word in question_lower for word in ['how', 'process', 'steps', 'procedure']):
        return "Process"
    elif any(word in question_lower for word in ['when', 'date', 'time', 'deadline']):
        return "Temporal"
    elif any(word in question_lower for word in ['who', 'contact', 'responsible']):
        return "Contact"
    elif any(word in question_lower for word in ['what', 'definition', 'meaning']):
        return "Definitional"
    else:
        return "General"

def generate_follow_ups(question: str, answer: str) -> List[str]:
    """Generate relevant follow-up questions."""
    category = categorize_question(question)
    
    follow_ups = {
        "Financial": [
            "Are there any additional fees or charges?",
            "How are payments processed?",
            "What payment methods are accepted?"
        ],
        "Process": [
            "What documents are required for this process?",
            "How long does this process typically take?",
            "Who can help if there are issues with this process?"
        ],
        "Temporal": [
            "Are there any exceptions to this timeline?",
            "What happens if deadlines are missed?",
            "How are dates calculated?"
        ],
        "Contact": [
            "What are the contact hours?",
            "Are there alternative contact methods?",
            "What information should I have ready when contacting?"
        ],
        "Definitional": [
            "Are there any related terms I should know?",
            "How does this apply to my specific situation?",
            "Are there any exceptions to this definition?"
        ]
    }
    
    return follow_ups.get(category, [
        "Can you provide more details about this topic?",
        "Are there any important exceptions or conditions?",
        "How does this relate to other policies or procedures?"
    ])

# Example usage
questions = [
    "What is the coverage amount?",
    "How do I file a claim?",
    "When is the payment due?"
]

# Simulate API response
answers = [
    "the coverage amount is $100000 per incident",
    "You can file claims online or by phone within 30 days",
    "payments are due on the 15th of each month"
]

enhanced_responses = enhance_responses(questions, answers)

print("Enhanced Response Analysis:")
print("=" * 50)

for response in enhanced_responses:
    print(f"\nQuestion: {response['original_question']}")
    print(f"Category: {response['category']}")
    print(f"Original Answer: {response['original_answer']}")
    print(f"Enhanced Answer: {response['enhanced_answer']}")
    print(f"Confidence: {response['confidence']}")
    print("Follow-up Suggestions:")
    for follow_up in response['follow_up_suggestions'][:2]:  # Show first 2
        print(f"  - {follow_up}")
```

## Integration Examples

### 1. Web Application Integration

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Q&A System</title>
    <style>
        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .form-group input, .form-group textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        .loading { display: none; text-align: center; margin: 20px 0; }
        .results { margin-top: 20px; }
        .qa-pair { background: #f8f9fa; padding: 15px; margin-bottom: 10px; border-radius: 4px; }
        .question { font-weight: bold; color: #495057; }
        .answer { margin-top: 5px; color: #212529; }
        .error { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 4px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Document Q&A System</h1>
        <p>Upload a document URL and ask questions to get instant answers!</p>
        
        <form id="qaForm">
            <div class="form-group">
                <label for="documentUrl">Document URL:</label>
                <input type="url" id="documentUrl" required 
                       placeholder="https://example.com/document.pdf">
            </div>
            
            <div class="form-group">
                <label for="questions">Questions (one per line):</label>
                <textarea id="questions" rows="5" required 
                          placeholder="What is the main topic?&#10;Who are the key stakeholders?&#10;What are the important dates?"></textarea>
            </div>
            
            <button type="submit" class="btn">Get Answers</button>
        </form>
        
        <div class="loading" id="loading">
            <p>Processing your document and questions... This may take 30-60 seconds for large documents.</p>
        </div>
        
        <div class="results" id="results"></div>
    </div>

    <script>
        document.getElementById('qaForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const documentUrl = document.getElementById('documentUrl').value;
            const questionsText = document.getElementById('questions').value;
            const questions = questionsText.split('\n').filter(q => q.trim());
            
            // Validate input
            if (!questions.length) {
                showError('Please enter at least one question.');
                return;
            }
            
            if (questions.length > 10) {
                showError('Maximum 10 questions allowed.');
                return;
            }
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';
            
            try {
                const response = await fetch('/hackrx/run', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        documents: documentUrl,
                        questions: questions
                    })
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.detail || `HTTP ${response.status}`);
                }
                
                displayResults(questions, data.answers);
                
            } catch (error) {
                showError(`Error: ${error.message}`);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });
        
        function displayResults(questions, answers) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<h2>Results:</h2>';
            
            questions.forEach((question, index) => {
                const qaDiv = document.createElement('div');
                qaDiv.className = 'qa-pair';
                qaDiv.innerHTML = `
                    <div class="question">Q${index + 1}: ${question}</div>
                    <div class="answer">A${index + 1}: ${answers[index] || 'No answer provided'}</div>
                `;
                resultsDiv.appendChild(qaDiv);
            });
        }
        
        function showError(message) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `<div class="error">${message}</div>`;
        }
    </script>
</body>
</html>
```

### 2. Python Client Library

```python
import requests
import time
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class DocumentQAResult:
    """Result from document Q&A processing."""
    questions: List[str]
    answers: List[str]
    processing_time: float
    success: bool
    error: Optional[str] = None

class DocumentQAClient:
    """
    Python client for the Document Q&A API.
    
    Provides a convenient interface for interacting with the API
    with built-in error handling, retries, and response validation.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 120):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def query_document(
        self, 
        document_url: str, 
        questions: Union[str, List[str]],
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> DocumentQAResult:
        """
        Query a document with questions.
        
        Args:
            document_url: URL of the document to process
            questions: Single question or list of questions
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            DocumentQAResult with answers and metadata
        """
        # Normalize questions to list
        if isinstance(questions, str):
            questions = [questions]
        
        # Validate input
        if not questions:
            return DocumentQAResult(
                questions=[], answers=[], processing_time=0.0,
                success=False, error="No questions provided"
            )
        
        if len(questions) > 10:
            return DocumentQAResult(
                questions=questions, answers=[], processing_time=0.0,
                success=False, error="Maximum 10 questions allowed"
            )
        
        start_time = time.time()
        
        for attempt in range(max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/hackrx/run",
                    json={
                        "documents": document_url,
                        "questions": questions
                    },
                    timeout=self.timeout
                )
                
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    return DocumentQAResult(
                        questions=questions,
                        answers=data["answers"],
                        processing_time=processing_time,
                        success=True
                    )
                else:
                    error_detail = response.json().get("detail", f"HTTP {response.status_code}")
                    if attempt < max_retries:
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        return DocumentQAResult(
                            questions=questions, answers=[], 
                            processing_time=processing_time,
                            success=False, error=error_detail
                        )
                        
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                else:
                    processing_time = time.time() - start_time
                    return DocumentQAResult(
                        questions=questions, answers=[],
                        processing_time=processing_time,
                        success=False, error=str(e)
                    )
    
    def health_check(self) -> bool:
        """Check if the API server is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()

# Example usage
if __name__ == "__main__":
    # Create client
    client = DocumentQAClient()
    
    # Check if server is running
    if not client.health_check():
        print("❌ API server is not accessible")
        exit(1)
    
    print("✅ API server is running")
    
    # Query document
    result = client.query_document(
        document_url="https://example.com/policy.pdf",
        questions=[
            "What is the coverage amount?",
            "How do I file a claim?",
            "What are the exclusions?"
        ]
    )
    
    # Display results
    if result.success:
        print(f"✅ Processing completed in {result.processing_time:.2f} seconds")
        print("\nResults:")
        for i, (question, answer) in enumerate(zip(result.questions, result.answers)):
            print(f"\nQ{i+1}: {question}")
            print(f"A{i+1}: {answer}")
    else:
        print(f"❌ Error: {result.error}")
```

This comprehensive usage guide provides real-world examples for integrating and using the Document Q&A system effectively. It covers basic usage, advanced patterns, error handling, quality enhancement, and complete integration examples for different platforms.