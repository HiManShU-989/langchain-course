from typing import List
from pydantic import BaseModel, Field

class Reflection(BaseModel): #Helps LLM reflect on its answer and improve it. The desciption is used to generate the prompt for the LLM.
    missing: str = Field(description="Critique of what is missing.")
    superflous: str = Field(description="Critique of what is superflous.")
    
class AnswerQuestion(BaseModel): #Represents the structured output for answering a question.
    """Answer the question."""
    answer: str = Field(description="~250 words detailed answer to the question.")
    reflection: str = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )