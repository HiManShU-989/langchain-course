import os
import time
from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Pass the entire 'urls' list to a single WebBaseLoader.
# This automatically returns a flat list of Documents: [Doc1, Doc2, Doc3]
docs = (WebBaseLoader(web_paths=urls)).load()

# Preparing the text splitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 250,
    chunk_overlap = 0
)

#Splitting the docs obtained from url int0 chunks
doc_lists = text_splitter.split_documents(docs)

# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# vectorstore = Chroma(
#     collection_name="rag-chroma",
#     embedding_function=embeddings,
#     persist_directory="./.chroma"
# )

# # Add chunks in batches of 20 with 3-second pauses to prevent 429 API rate limits
# batch_size = 20
# for i in range(0, len(doc_lists), batch_size):
#     batch = doc_lists[i : i + batch_size]
#     vectorstore.add_documents(batch)
#     time.sleep(15)
# # ==============================================================================

retriever = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001"),
    collection_name="rag-chroma",
    persist_directory="./.chroma"
    ).as_retriever() #Makes it as a langchain retriever for doing similarity search etc.