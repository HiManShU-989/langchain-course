from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph, END
from graph.consts import GENERATE, GRADEDOCUMENTS, RETRIEVE, WEBSEARCH
from graph.state import GraphState
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery

def route_question(state: GraphState):
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question":question})
    if source.datasource == "websearch":
        print("---ROUTE QUESTION TO WEBSEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE

def decide_to_generate(state: GraphState):
    print("--ASSESS GRADED DOCUMENTS--")
    if state["web_search"]:
        print("--DECISION: NOT ALL DOCUMNETS WERE RELEVANT TO QUESTION, INCLUDE WEB SEARCH--")
        return WEBSEARCH
    else:
        print("--DECISION: GENERATE--")
        return GENERATE

def grade_generation_grounded_in_documents_and_generation(state: GraphState) -> str:
    print("-----CHECK HALLUCINATIONS-----")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallucination_grader.invoke({
        "document":documents, "generation":generation
    })
    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS.---")
        print("---GRADE GENERATION VS QUESTION---")
        score = answer_grader.invoke({
            "question": question, "generation": generation
        })
        if answer_grade := score.binary_score:
            print("---DECISION: GENERARION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERARION DOES NOT ADDRESSES QUESTION---")
            return "not useful"
    else:
       print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS RE-TRY.---")
       return "not supported"
            
workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADEDOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

# workflow.set_entry_point(RETRIEVE)
workflow.set_conditional_entry_point(route_question, path_map={
    WEBSEARCH: WEBSEARCH,
    RETRIEVE: RETRIEVE
})
workflow.add_edge(RETRIEVE, GRADEDOCUMENTS)
workflow.add_conditional_edges(GRADEDOCUMENTS, decide_to_generate, path_map={
    WEBSEARCH: WEBSEARCH,
    GENERATE: GENERATE
},)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_generation,
    path_map={
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")