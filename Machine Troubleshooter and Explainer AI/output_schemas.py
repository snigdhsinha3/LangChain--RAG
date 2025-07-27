# output_schemas.py
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

# Define a Pydantic model for the expected structured output from the agent
class AgentResponse(BaseModel):
    """
    Structured response from the AI assistant."""
    content: str = Field(description="The comprehensive answer to the user's question.")
    answer_source: Literal["manual", "web_search", "general_knowledge", "mixed", "none"] = Field(
        description= "The primary source of the answer."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=" THe AI's confidence level in the provided answer based on available information."
    )
    follow_up_questions: Optional[List[str]] = Field(
        default= None,
        description="A list of relevant follow-up questions the user might as based on the answer."
    )
