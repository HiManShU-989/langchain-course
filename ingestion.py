import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader(
        "C:\\Users\\himanshu.singh13\\Desktop\\Learnings\\LangChainLanggraphUdemy\\langchain-course\\mediumblog1.txt",
        encoding="utf-8") #Loads the text file from the specified path with UTF-8 encoding, can also be used to load whatsapp chats, slack messages etc.
    document = loader.load() #Contains the loaded document, which is a list of Document objects. Each Document object represents a single document and contains the text content and metadata associated with that document.
    print("Splitting...")
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) #Creates a splitter that splits the document into chunks of 1000 characters with no overlap between chunks. This is useful for processing large documents in smaller, manageable pieces.
    texts = text_splitter.split_documents(document) #Splits the loaded document into smaller chunks using the specified chunk size and overlap. The result is a list of Document objects, each representing a chunk of text.
    print(f"created {len(texts)} chunks")
    
    embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001") #Creates an instance of the GoogleGenerativeAIEmbeddings class, which is used to generate embeddings for the text chunks. Embeddings are numerical representations of text that capture semantic meaning and can be used for various NLP tasks.
    print("Ingesting")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME")) #Creates a Pinecone vector store from the text chunks and their corresponding embeddings. The vector store is used to efficiently store and retrieve embeddings for similarity search and other tasks. The index name is specified using an environment variable.
    print("Finish.")