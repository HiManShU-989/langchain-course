from dotenv import load_dotenv
load_dotenv()
from typing import List
from pydantic import BaseModel, Field
from langchain.agents import create_agent #Modern way to create agents. This function abstracts away the details of how to set up an agent, and allows you to create agents with just a few lines of code.
from langchain.tools import tool #Under the hood, tools are callable functions with well-defined inputs and outputs that get passed to a chat model. The model decides when to invoke a tool based on the conversation context, and what input arguments to provide.
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch 
from langchain_google_genai import ChatGoogleGenerativeAI

# @tool     #Makes a tool with the description, the arguments which LLM can use to decide if it can call the tool or not.
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query (str): The search query
#     Returns:
#         str: The search results
#     """
#     print(f"Seraching for {query}...")
#     return tavily.search(query = query) #Calls the search method of the Tavily client with the query provided by the LLM. The result is returned as a string which will be passed back to the LLM as the output of the tool call.

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")
    
class AgentResponse(BaseModel):
    """Schema for the response of the agent"""
    answer: str = Field(description="The answer to the user's question")
    sources: List[Source] = Field(default_factory=list, description="The sources used by the agent to answer the question")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") #Initialize the LLM
tools = [TavilySearch()] #List of tools to be used by the agent
agent = create_agent(model = llm, tools = tools, response_format=AgentResponse) #Create an agent with the LLM and the tools
#Added response_format to specify the format of the agent's response. This allows us to have a structured response that includes both the answer and the sources used by the agent.

def main():
    result = agent.invoke({"messages": HumanMessage(content="search for 3 job postings for an ai engineer using langchain in Gurugram or Bangalore on linkedin and list their details?")}) #Invoke the agent with a human message. The agent will decide whether to call the tool or not based on the message and the tool's description.
    print(result)
    # print("Hello from langchain-course!")


if __name__ == "__main__":
    main()
