from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0)

class GradeHallucinations(BaseModel):
    """Binary Score for Hallucinations present in Generation answer."""
    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'."
    )
    
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an answer addresses / resolves a question \n 
Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
     
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
            ("system",system),
            ("human", "Set of facts: \n\n {document} \n\n LLM Generation: {generation}")
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader