from typing import Any, Dict
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState
from dotenv import load_dotenv
load_dotenv() 

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print('---WEB SEARCH---')
    question = state["question"]
    documents = state.get("documents") or [] 
    
    # 1. Invoke Tavily
    tavily_results = web_search_tool.invoke({"query": question})
    
    # 2. Extract the 'results' list from the dictionary
    search_results = tavily_results.get("results", [])
    
    # 3. Create a distinct LangChain Document for EACH search result (Best Practice)
    web_docs = [
        Document(
            page_content=result["content"], 
            metadata={"source": result.get("url", "tavily_search")} # Save the URL for citations!
        )
        for result in search_results
    ]
    
    # 4. Safely combine the old documents with the newly searched documents
    updated_documents = documents + web_docs
    
    # 5. Return the updated state
    return {"question": question, "documents": updated_documents}
    

if __name__ == "__main__":
    web_search(state = {"question": "agent memory", "documents":None})