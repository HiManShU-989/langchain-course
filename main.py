import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

stdio_server_params = StdioServerParameters(
    command = "python",
    args = ["C:\\Users\\himanshu.singh13\\Desktop\\Learnings\\LangChainLanggraphUdemy\\langchain-course\\servers\\math_server.py"],
)

async def main():
    async with stdio_client(stdio_server_params) as (read,write):
        async with ClientSession(read_stream =  read, write_stream = write) as session:
            await session.initialize()
            print("Session initialized")
            tools = await load_mcp_tools(session)
            # print(f"Tools loaded with tools: {tools}")
            agent = create_agent(llm, tools)
            result = await agent.ainvoke({"messages":[HumanMessage(content="What is 54+2*3?")]})
            print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())