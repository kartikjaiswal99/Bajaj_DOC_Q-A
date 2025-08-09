from pydantic import BaseModel as FastApiBaseModel, Field
from typing import List

class HackathonInput(FastApiBaseModel):
    documents: str
    questions: List[str]

class HackathonOutput(FastApiBaseModel):
    answers: List[str]