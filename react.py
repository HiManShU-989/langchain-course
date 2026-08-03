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

tools = [TavilySearch(max_results=1),triple] #tavily search tool and a custom tool to triple a number.

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature = 0).bind_tools(tools) #llm binded with the tools.