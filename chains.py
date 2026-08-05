import datetime
from dotenv import load_dotenv
load_dotenv()

from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser
)

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from schemas import AnswerQuestion

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

#Preparing the prompt template for the first responder, which will answer the question and provide reflection and search queries.
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
            (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time = lambda:datetime.datetime.now().isoformat()
)
#Partial is used to fill in the current time dynamically when the prompt is generated.

first_responder_prompt_template = actor_prompt_template.partial( #Fills in the first instruction for the first responder to provide a detailed answer.
    first_instruction = "Provide a detailed 250 word answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion],
    tool_choice="AnswerQuestion"
)

if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC/ autonomous soc problem domain,"
        "list startups that do that and raised capital."
    )
    
    chain = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion],
    tool_choice="AnswerQuestion"
) | parser_pydantic
    
    res = chain.invoke(input={"messages":[human_message]})
    print(res)