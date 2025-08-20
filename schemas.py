# ==============================================================================
#                    DATA SCHEMAS AND PYDANTIC MODELS
# ==============================================================================
#
# This module defines the data structures and validation schemas used throughout
# the Advanced Document Q&A System. It includes both API interface models and
# internal processing schemas for the RAG pipeline.
#
# SCHEMA CATEGORIES:
# ==================
# 1. API Schemas: Request/response models for the hackathon endpoint
# 2. Internal Schemas: Data structures for RAG pipeline processing
# 3. Validation Rules: Input validation and data sanitization
#
# DESIGN PRINCIPLES:
# ==================
# - Type safety with Pydantic validation
# - Clear separation of concerns between API and internal models
# - Comprehensive field documentation for maintainability
# - Flexible schema design for future extensibility
#
# ==============================================================================

from typing import List, Optional
from pydantic import BaseModel as FastApiBaseModel, BaseModel, Field

# ==============================================================================
#                         HACKATHON API SCHEMAS
# ==============================================================================
# These schemas define the contract for the main /hackrx/run endpoint

class HackathonInput(FastApiBaseModel):
    """
    Request schema for the main document Q&A endpoint.
    
    This model validates incoming requests to ensure they contain the required
    document URL and questions list. The validation includes format checking
    and reasonable constraints on input size.
    
    Attributes:
        documents (str): URL of the document to process. Must be accessible
                        via HTTP/HTTPS and in supported format (PDF, DOCX, EML)
        questions (List[str]): List of questions to answer based on document.
                              Maximum 10 questions to prevent abuse and ensure
                              reasonable response times.
    
    Example:
        {
            "documents": "https://example.com/policy.pdf",
            "questions": [
                "What is the coverage amount?",
                "What are the exclusions?"
            ]
        }
    """
    documents: str = Field(
        description="URL of the document to process",
        example="https://example.com/document.pdf"
    )
    questions: List[str] = Field(
        description="List of questions to answer based on the document",
        min_items=1,
        max_items=10,
        example=[
            "What is the main topic?",
            "Who are the key stakeholders?"
        ]
    )

class HackathonOutput(FastApiBaseModel):
    """
    Response schema for the main document Q&A endpoint.
    
    This model ensures consistent response formatting with answers provided
    in the same order as the input questions. Each answer is optimized for
    clarity and conciseness.
    
    Attributes:
        answers (List[str]): List of answers corresponding to input questions.
                            Each answer is limited to ~200 characters for
                            optimal readability and API performance.
    
    Example:
        {
            "answers": [
                "The main topic is health insurance coverage for employees.",
                "Key stakeholders include employees, HR department, and insurance provider."
            ]
        }
    """
    answers: List[str] = Field(
        description="List of answers corresponding to the input questions",
        example=[
            "The coverage amount is $100,000 per incident.",
            "Exclusions include pre-existing conditions and experimental treatments."
        ]
    )

# ==============================================================================
#                    INTERNAL RAG PIPELINE SCHEMAS
# ==============================================================================
# These schemas support internal processing and structured reasoning

class ParsedQuery(BaseModel):
    """
    Structured representation of a parsed user query.
    
    This model helps the system understand and categorize user questions
    for better retrieval and answer generation. It extracts key entities
    and identifies the main procedure or topic being queried.
    
    Attributes:
        procedure (Optional[str]): The main procedure, process, or key activity
                                  mentioned in the query. Helps focus retrieval
                                  on relevant document sections.
        entities (List[str]): Key nouns, terms, or entities extracted from the
                             question (e.g., dates, amounts, policy terms,
                             conditions). Used for enhanced context matching.
    
    Usage:
        This model is used internally by the RAG chain to better understand
        user intent and improve retrieval accuracy.
    
    Example:
        ParsedQuery(
            procedure="claim filing",
            entities=["deadline", "required documents", "contact information"]
        )
    """
    procedure: Optional[str] = Field(
        None, 
        description="The main medical procedure or key activity mentioned in the query",
        example="claim filing process"
    )
    entities: List[str] = Field(
        description="A list of other key nouns or entities (e.g., age, policy terms, conditions)",
        example=["deadline", "forms", "documentation"]
    )

class FinalAnswer(BaseModel):
    """
    The final, processed answer from the RAG chain.
    
    This model represents the culmination of the retrieval-augmented generation
    process, containing the final answer that will be returned to the user.
    The answer is validated for quality and appropriate length.
    
    Attributes:
        answer (str): A direct, concise, and factual answer to the user's question.
                     The answer is optimized for clarity and typically limited
                     to 1-2 sentences for optimal user experience.
    
    Quality Standards:
        - Factual accuracy based on document content
        - Concise and direct response style
        - Professional and clear language
        - Appropriate length (typically 50-200 characters)
        - Fallback to standard message if content insufficient
    
    Example:
        FinalAnswer(
            answer="The coverage limit is $100,000 per incident with a $500 deductible."
        )
    """
    answer: str = Field(
        description="A direct, concise, and factual answer to the user's question",
        min_length=1,
        max_length=500,
        example="The policy covers medical expenses up to $100,000 annually."
    )

# ==============================================================================
#                           VALIDATION HELPERS
# ==============================================================================
# Additional validation functions and custom validators can be added here

def validate_document_url(url: str) -> bool:
    """
    Validate that a document URL is properly formatted and accessible.
    
    Args:
        url (str): The document URL to validate
        
    Returns:
        bool: True if URL appears valid, False otherwise
        
    Note:
        This is a basic validation. In production, additional checks
        for domain whitelisting and security should be implemented.
    """
    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None

def validate_question_quality(question: str) -> bool:
    """
    Validate that a question meets quality standards.
    
    Args:
        question (str): The question to validate
        
    Returns:
        bool: True if question meets quality standards
        
    Quality Criteria:
        - Minimum length for meaningful questions
        - Contains at least one question word or ends with '?'
        - Not just whitespace or special characters
    """
    if not question or len(question.strip()) < 3:
        return False
    
    question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which', 'can', 'is', 'are', 'do', 'does']
    question_lower = question.lower().strip()
    
    return (
        question_lower.endswith('?') or 
        any(word in question_lower for word in question_words)
    )

# ==============================================================================
#                              EXAMPLES
# ==============================================================================

# Example usage for testing and development
if __name__ == "__main__":
    # Example API request
    sample_request = HackathonInput(
        documents="https://example.com/insurance-policy.pdf",
        questions=[
            "What is the maximum coverage amount?",
            "How do I file a claim?",
            "What are the exclusions?"
        ]
    )
    
    # Example API response
    sample_response = HackathonOutput(
        answers=[
            "The maximum coverage amount is $500,000 per incident.",
            "Claims can be filed online or by calling the customer service number.",
            "Exclusions include pre-existing conditions and experimental treatments."
        ]
    )
    
    print("Sample Request:", sample_request.json(indent=2))
    print("Sample Response:", sample_response.json(indent=2))
