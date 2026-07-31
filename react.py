from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(num:float)->float:
    """param num: a number to triple
    returns: the triple of the input number
    """
    return 3*float(num)

tools = [TavilySearch(max_results=1),triple]

llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature = 0).bind_tools(tools)