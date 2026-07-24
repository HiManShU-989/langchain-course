import asyncio
# When you define a function with async def, it becomes a coroutine. If you try to call a coroutine like a normal function (e.g., typing main()), it will not actually run the code inside. Instead, it just returns a "coroutine object" that sits there doing nothing.
# To actually execute a coroutine, something needs to manage it. That "something" is the Event Loop—a continuous loop that monitors and runs asynchronous tasks, pausing them when they are waiting for something (like a network response) and resuming them when they are ready.
# asyncio.run(main()) acts as the bridge. It does three critical things automatically:
# Creates a brand-new event loop.
# Runs your main() coroutine until it finishes.
# Shuts down the event loop safely when done.
import os
import ssl
from typing import List, Any, Dict
import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import (Colors, log_error, log_info, log_success, log_warning, log_header)


load_dotenv()

# Configure SSL context to use certifi's CA bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = GoogleGenerativeAIEmbeddings( #Settings are same as we set in Dify.
    model="gemini-embedding-001",
    show_progress_bar=True,
    chunk_size = 50,
    retry_min_seconds=10
    )

# chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings) #For local storage, but we are using Pinecone for vector storage in this case.
vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings
)

# tavily_extract = TavilyExtract()
# tavily_map = TavilyMap(
#     max_depth=5, #How deep to crawl (default: 3)
#     max_breadth=20, #How many links per page (default: 10)
#     max_pages=1000 # Maximum total pages to discover (default: 100)
# )

tavily_crawl = TavilyCrawl() #Used to crawl web pages easily and directly.


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(f"VectorStore Indexing: Preparing to add {len(documents)} documents to the vector store.",
             Colors.DARKCYAN)
    
    #Create batches
    batches = [
        documents[i:i+batch_size] for i in range(0,len(documents),batch_size)
    ]
    log_info(f"VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each.")
    
    #Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num:int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} {len(batch)} documents")
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True
    
    #Process batches concurrently
    tasks = [add_batch(batch,i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    #Count successful batches
    successful = sum(1 for result in results if result is True)
    if successful==len(batches):
        log_success(f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})")
    else:
        log_warning(f"VectorStore Indexing: Processed {successful}/{len(batches)} successfully")
    
async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")
    
    log_info("TavilyCrawl: Starting the Crawl documentation from https://python.langchain.com/",
             Colors.PURPLE)
    
    #Crawl the documentation site
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 5, #Keep it low for testing, but can be increased for full crawl.
        "extract_depth": "advanced", #Extract all content from the pages, increased latency but higher quality of content.
        "instructions": "Content on AI Agents" #Used as a filter to crawl specific content so give exact instructions.
    })
    all_docs = [Document(page_content=result["raw_content"], metadata = {"source": result["url"]}) for result in res["results"]] #This is done to convert the results into langchain documents that can be used later while embedding and storing to pinecone.
    log_success(f"TavilyCrawl: Successfully crawled {len(all_docs)} URL from documentation Site")
   
#Total Size of chuncking is input prompt with context + output tokens.
    # Split Document into chunks.
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(f"Text Splitter: Processing {len(all_docs)} documments with 4000 chunk size and 200 overlap", Colors.YELLOW)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 4000, chunk_overlap = 200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents") 
    #Modern day LLMs have millions of token as a limit but still RAG is important because it saves us tokens and cost by providing the relevant chunks as context also it is easier to debug as we can see the chunks and their related meta data.
    #Process documents asyncronously
    await index_documents_async(splitted_docs,batch_size=500)
    
    log_header("PIPELINE COMPLETE")
    log_success("Documentation ingestion pipeline finished successfully!")
    log_info("Summary:",Colors.BOLD)
    # log_info(f"   URLs mapped: {len(all_docs["results"])}")
    log_info(f"   Documents extracted: {len(all_docs)}")
    log_info(f"   Chunks created: {len(splitted_docs)}")
    
if __name__ == "__main__":
    asyncio.run(main())