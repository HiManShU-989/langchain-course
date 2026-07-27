import streamlit as st # Imports the Streamlit library for building the web app UI
import re # Imports Python's regex module to help strip out inline sources
from typing import Any, Dict, List # Imports type hints for better code readability and linting
from backend.core import run_llm # Imports the custom backend function that runs the LangChain/LLM logic

def _format_sources(context_docs: List[Any]) -> List[str]:
    """Extract source URLs from retrieved document metadata."""
    sources = [] # Initializes an empty list to store unique source URLs
    for doc in (context_docs or []): # Iterates through the provided documents, defaulting to an empty list if None
        # Extract metadata from the document object, defaulting to an empty dict if missing
        meta = getattr(doc, "metadata", None) or {}
        
        # Attempt to get the URL from the 'source' key first, then fallback to 'sources'
        source = meta.get("source") or meta.get("sources")
        
        # Check if a source exists and hasn't been added yet (prevents duplicates)
        if source and str(source) not in sources:  
            sources.append(str(source)) # Add the unique source URL to our list as a string
            
    return sources # Return the final list of unique source URLs

# Configure the Streamlit page title (browser tab) and set the layout width to centered
st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("LangChain Documentation Helper") # Renders the main H1 heading on the web page

# --- Sidebar Configuration ---
with st.sidebar: # Opens a context manager to place UI elements inside the left sidebar
    st.subheader("Session") # Renders a smaller heading inside the sidebar
    # Creates a button spanning the full width; evaluates to True only when clicked
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.pop("messages", None) # Removes the 'messages' list from session state to clear history
        st.rerun() # Forces Streamlit to rerun the script from top to bottom, refreshing the UI

# --- Initialize Chat History ---
if "messages" not in st.session_state: # Checks if this is the first time the app is loading
    st.session_state.messages = [ # If so, initializes the message history list
        { # Adds a default greeting message from the AI assistant
            "role": "assistant", # Specifies the sender's role
            "content": "Ask me anything about LangChain docs. I'll retrieve relevant context and cite sources.", # The greeting text
            "sources": [] # Initializes an empty list for sources for this specific message
        }
    ]

# --- Display Existing Chat Messages ---
for msg in st.session_state.messages: # Loops through all saved messages in the session state history
    with st.chat_message(msg["role"]): # Creates a chat UI container based on the sender's role (user or assistant)
        st.markdown(msg["content"]) # Renders the message text using Markdown formatting
        
        if msg.get("sources"): # Checks if this specific message has associated source URLs
            with st.expander("Sources"): # Creates a collapsible UI accordion titled "Sources"
                for s in msg["sources"]: # Loops through each source URL
                    st.markdown(f"- {s}") # Renders each source as a Markdown bullet point

# --- Handle User Input ---
# Displays a chat input box at the bottom of the screen; captures text when the user hits Enter
prompt = st.chat_input("Ask a question about LangChain...") 

if prompt: # Proceeds only if the user actually typed and submitted text
    # Appends the new user message to the session state history
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    
    with st.chat_message("user"): # Opens a chat container for the user's message
        st.markdown(prompt) # Displays the user's submitted text on the screen immediately

    with st.chat_message("assistant"): # Opens a chat container for the assistant's upcoming response
        try: # Starts a try-except block to handle potential API or backend errors gracefully
            with st.spinner("Retrieving docs and generating answer..."): # Shows a loading spinner while processing
                # Calls the backend LLM function with the user's prompt and expects a dictionary back
                result: Dict[str, Any] = run_llm(prompt)
                
                # --- NEW CLEANUP LOGIC ---
                # 1. Grab the raw object (which might be a string, a dict, or a list of dicts containing signatures)
                raw_answer = result.get("answer") or result.get("result") or ""
                
                # 2. Extract ONLY the text, leaving behind signatures and extras
                if isinstance(raw_answer, list):
                    # If it's a list of blocks, extract the 'text' key from each dictionary block
                    answer = "".join(block.get("text", "") for block in raw_answer if isinstance(block, dict))
                elif isinstance(raw_answer, dict):
                    # If it's a single dictionary block, just grab the 'text' key
                    answer = raw_answer.get("text", "")
                else:
                    # If it's already a string (older LLM models), just cast it safely
                    answer = str(raw_answer)
                
                # 2. NEW FIX: Safely remove ONLY the specific lines where the LLM cited a source.
                # Uses regex to match lines starting with "Source:" or "Sources:" and deletes them.
                # '(?im)' makes it case-insensitive and applies to every individual line in a multiline block.
                answer = re.sub(r'(?im)^\s*\**sources?\**:\s*.*$', '', answer).strip()
                
                # 4. Fallback if the extracted answer ended up completely empty
                answer = answer or "(No answer returned.)"
                
                # Extracts retrieved documents, trying modern 'context' key first, then legacy 'source_documents'
                docs = result.get("context") or result.get("source_documents") or []
                
                # Passes the raw documents to our helper function to extract clean, unique URLs
                sources = _format_sources(docs)

            st.markdown(answer) # Displays the final, clean generated answer text to the user
            
            if sources: # If any valid sources were successfully extracted
                with st.expander("Sources"): # Creates a collapsible section for the citations
                    for s in sources: # Loops through the sources
                        st.markdown(f"- {s}") # Displays each source as a bullet point
                        
            # Saves the assistant's clean response and its sources into the session history
            st.session_state.messages.append(
                {
                    "role": "assistant", # Marks this as an assistant message
                    "content": answer, # Stores the clean answer text
                    "sources": sources # Stores the list of formatted sources
                }
            )      
        except Exception as e: # Catches any errors that occur during backend execution
            st.error("Failed to generate a response.") # Shows a red error banner to the user
            st.exception(e) # Prints the technical stack trace of the error below the banner for debugging