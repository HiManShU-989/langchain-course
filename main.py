from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END
from nodes import run_agent_reasoning, tool_node
load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1

def should_continue(state: MessagesState)-> str: #if it's the first time, we want to continue to the tool node, otherwise we want to end the flow.
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState) #Initializing a state graph with MessagesState as the state type. This will be used to manage the flow of messages and tool calls in the application.
#Below steps are used to build the flow of the application. We add nodes, set entry points, and define conditional edges to control the flow based on the state of messages and tool calls. Finally, we compile the flow and generate a visual representation of it in a PNG file.
flow.add_node(AGENT_REASON,run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT,tool_node)


flow.add_conditional_edges(AGENT_REASON, should_continue,{
    END:END,
    ACT:ACT
})

flow.add_edge(ACT,AGENT_REASON)
app = flow.compile()

app.get_graph().draw_mermaid_png(output_file_path="flow.png")


if __name__ == "__main__":
    print("Hello from langchain-course!")
    res = app.invoke({"messages":[HumanMessage(content="What is the temperature in Tokyo? List it and then triple it.")]})
    print(res["messages"][LAST].content)