from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph, END
from graph.consts import GENERATE, GRADEDOCUMENTS, RETRIEVE, WEBSEARCH
from graph.state import GraphState
from graph.nodes import generate, grade_documents, retrieve, web_search

def decide_to_generate(state):
    print("--ASSESS GRADED DOCUMENTS--")
    if state["web_search"]:
        print("--DECISION: NOT ALL DOCUMNETS WERE RELEVANT TO QUESTION, INCLUDE WEB SEARCH--")
        return WEBSEARCH
    else:
        print("--DECISION: GENERATE--")
        return GENERATE
    
workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADEDOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADEDOCUMENTS)
workflow.add_conditional_edges(GRADEDOCUMENTS, decide_to_generate, path_map={
    WEBSEARCH: WEBSEARCH,
    GENERATE: GENERATE
},)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")
