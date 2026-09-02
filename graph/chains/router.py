from typing import Literal #Literal is used when a variable can take only predefined values only, for type checking.
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["websearch", "vectorstore"] = Field(
        ...,
        description="Given a user question, choose to route it to websearch or a vectorstore."
    )
    
llm = ChatGoogleGenerativeAI(model = "gemini-3.5-flash", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains document related to agents, prompt engineering, and advarsarial attacks.
Use the vectorstore for questions on these topics. For all else, use web search."""

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{question}"),
        ("system", system),
    ]
)

question_router = router_prompt | structured_llm_router