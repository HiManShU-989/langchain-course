import datetime
from dotenv import load_dotenv

# Load environment variables (like GOOGLE_API_KEY) from the .env file
load_dotenv()

from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from schemas import AnswerQuestion, ReviseAnswer

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Parsers to extract the structured data from the LLM's raw output (used primarily for testing below)
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

# ---------------------------------------------------------
# CORE PROMPT TEMPLATE
# ---------------------------------------------------------
# This base prompt is shared by both the drafter and the revisor.
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time} # Injects current time so the LLM knows the current date for context.

1. {first_instruction} # A placeholder for specific instructions (drafting vs revising).
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        # This placeholder will inject the entire conversation history (the MessagesState)
        MessagesPlaceholder(variable_name="messages"),
        # A final strict reminder to output data using the bound tool schema
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())
# .partial() dynamically injects the current timestamp every time the prompt is invoked.

# ---------------------------------------------------------
# DRAFTER CHAIN (First Responder)
# ---------------------------------------------------------
# Fill in the {first_instruction} variable for the initial drafting phase
first_responder_prompt_template = actor_prompt_template.partial(  
    first_instruction="Provide a detailed 250 word answer."
)

# Create the drafter chain. 
# bind_tools forces the LLM to output its response using the AnswerQuestion schema.
# tool_choice="AnswerQuestion" guarantees it will ALWAYS use this format.
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

# ---------------------------------------------------------
# REVISOR CHAIN
# ---------------------------------------------------------
# Specific instructions for the revision phase, demanding citations and adherence to critiques.
revise_instructions = """Revise your previous answers using the new information.
    -You should use the previous critique to add important information to your answer.
        -You MUST include numerical citations in your revised answer to ensure it can be verified.
        -Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            -[1] https://example.com
            -[2] https://example.com
    -You should use previous critique to remove superflous information from your answer and make SURE it is not more than 250 words.
"""

# Create the revisor chain. 
# bind_tools forces the LLM to output its response using the ReviseAnswer schema.
revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


# ---------------------------------------------------------
# TESTING BLOCK
# ---------------------------------------------------------
# This block only runs if you execute chains.py directly. It does not run when imported by main.py.
if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC/ autonomous soc problem domain,"
        "list startups that do that and raised capital."
    )

    # A simple test chain to see if the LLM correctly outputs the Pydantic model
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)