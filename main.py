from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
load_dotenv()


def main():
    print("Welcome to the Ollama LangChain example!")
    information = """Elon Musk cofounded seven companies, including electric car maker Tesla, rocket producer SpaceX and artificial intelligence startup xAI.
He owns about 12% of Tesla, which he first backed in 2004, and which he's led as CEO since 2008. He also owns options to acquire another 8%.
Musk led a group that bought Twitter for $44 billion in 2022. He merged it with xAI in 2025 in deal that valued the combined company at $113 billion (net of debt).
SpaceX, founded in 2002, acquired xAI in February 2026 in a deal that valued the combined company at $1.25 trillion. Musk owns a nearly 40% stake.
Musk also founded tunneling startup The Boring Company and brain implant outfit Neuralink. The two startups have raised around $2 billion from private investors combined.
    """
    
    summary_template = """Given the information {information} about a person, I want you to create:
    1. A short summary
    2. Two interesting facts about the person
    """
    
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    ) # This creates a prompt template that takes "information" as an input variable and uses the defined summary_template as the template for generating the prompt. The PromptTemplate class allows us to easily create and manage prompts in a structured way, making it easier to generate consistent and well-formatted prompts for our language model.

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )
    chain = summary_prompt_template | llm #LCEL(Langchain Expression Language) Creates a runnable object in which the output of one component is the input to the other component.
    response = chain.invoke(input={"information": information})
    print(response.content)
if __name__ == "__main__":
    main()