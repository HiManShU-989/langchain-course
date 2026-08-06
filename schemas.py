from typing import List
from pydantic import BaseModel, Field

class Reflection(BaseModel):
    """
    Helps the LLM reflect on its answer and improve it. 
    The descriptions provided in the Field definitions act as prompts 
    that tell the LLM exactly what kind of content to generate for each field.
    """
    # Instructs the LLM to identify gaps or missing details in its draft
    missing: str = Field(description="Critique of what is missing.")
    # Instructs the LLM to identify unnecessary fluff or off-topic information
    superflous: str = Field(description="Critique of what is superflous.")


class AnswerQuestion(BaseModel):
    """
    Represents the structured output expected from the FIRST pass of the LLM.
    It forces the LLM to not just answer, but also reflect and propose web searches.
    """
    # The actual response to the user's prompt
    answer: str = Field(description="~250 words detailed answer to the question.")
    # The LLM's self-critique of the answer it just generated
    reflection: str = Field(description="Your reflection on the initial answer.")
    # A list of search queries the LLM wants to run to fill in the gaps identified in the reflection
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class ReviseAnswer(AnswerQuestion):
    """
    Represents the structured output expected from the REVISION pass of the LLM.
    It inherits from AnswerQuestion (meaning it still needs an answer, reflection, and search_queries),
    but adds a requirement for citations.
    """
    # Forces the LLM to cite its sources based on the web search results
    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )