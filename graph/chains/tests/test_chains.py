from dotenv import load_dotenv

load_dotenv()

from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from ingestion import retriever
from pprint import pprint
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucinations
from graph.chains.router import RouteQuery, question_router


def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    assert len(docs) > 0, "Retriever returned 0 documents"
    doc_txt = docs[0].page_content  # Use top document (index 0)
    
    res: GradeDocuments = retrieval_grader.invoke({
        "question": question, "document": doc_txt
    })
    
    assert res.binary_score == "yes"
    

def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    assert len(docs) > 0, "Retriever returned 0 documents"
    doc_txt = docs[0].page_content
    
    res: GradeDocuments = retrieval_grader.invoke({
        "question": "How to make pizza", "document": doc_txt
    })
    
    assert res.binary_score == "no"
    

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({
        "context": docs, "question": question 
    })
    pprint(generation)
    assert generation is not None
    

def test_hallucination_grade_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({
        "context": docs, "question": question 
    })    
    res: GradeHallucinations = hallucination_grader.invoke(
        {"document": docs, "generation": generation}
    )
    assert res.binary_score
    

def test_hallucination_grade_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    res: GradeHallucinations = hallucination_grader.invoke(
        {"document": docs, "generation": "In order to make pizza we need first dough"}
    )
    assert not res.binary_score
    
def test_router_to_vectorstore() -> None:
    question = "agent memory"
    res: RouteQuery =  question_router.invoke({"question":question})
    assert res.datasource == "vectorstore"
    
def test_router_to_websearch() -> None:
    question = "How to make pizza?"
    res: RouteQuery =  question_router.invoke({"question":question})
    assert res.datasource == "websearch"