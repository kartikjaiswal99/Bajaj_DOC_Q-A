# ==============================================================================
#
#           DATA SCHEMAS (Pydantic Models)
#
# Schemas for the external API and our internal reasoning process.
# ==============================================================================

from typing import List, Optional
from pydantic import BaseModel as FastApiBaseModel, BaseModel, Field

# --- Hackathon API Schemas (Request and Response) ---
class HackathonInput(FastApiBaseModel):
    documents: str  # Changed from List[str] to str for single document
    questions: List[str]

class HackathonOutput(FastApiBaseModel):
    answers: List[str]

# --- Internal RAG Pipeline Schemas for Structured Reasoning ---
class ParsedQuery(BaseModel):
    """Structured representation of a user's query."""
    procedure: Optional[str] = Field(None, description="The main medical procedure or key activity mentioned in the query.")
    entities: List[str] = Field(description="A list of other key nouns or entities (e.g., age, policy terms, conditions).")

class FinalAnswer(BaseModel):
    """The final, concise answer from our internal RAG chain."""
    answer: str = Field(description="A direct, concise, and factual answer to the user's question.")
