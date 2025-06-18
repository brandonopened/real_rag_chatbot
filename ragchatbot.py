import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# Remove reference to RetrievalQA since we're using our own implementation
# from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Any, Dict, List, Optional, Sequence, Union, Callable
from PyPDF2 import PdfReader
import os
import uuid
import shutil
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

# Constants
PERSIST_DIRECTORY = "./data/chroma_db"
COLLECTION_NAME = "rag_documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
FEEDBACK_FILE = "feedback.json"

# --- File Processing ---
def process_file(uploaded_file):
    if uploaded_file.name.endswith(".txt"):
        content = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        content = "".join(page.extract_text() for page in pdf_reader.pages)
    else:
        st.error("Unsupported file type. Please upload a .txt or .pdf file.")
        return None, None
    
    return content, uploaded_file.name

# --- Text Splitting ---
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(text)

# --- Vector Store Setup ---
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_or_create_vector_store():
    # Create directory for persistence
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    
    # Initialize the embeddings
    embeddings = get_embeddings()
    
    # Get or create the vector store
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    return vector_store

def add_document_to_vector_store(content, filename):
    # Get the existing vector store
    vector_store = get_or_create_vector_store()
    
    # Split the text into chunks
    chunks = split_text(content)
    
    # Create metadata for each chunk
    metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]
    
    # Add chunks to vector store
    vector_store.add_texts(
        texts=chunks,
        metadatas=metadatas
    )
    
    return vector_store

def get_document_list():
    # Get the existing vector store
    vector_store = get_or_create_vector_store()
    
    # Get all documents
    documents = vector_store.get()
    
    # Extract unique filenames
    if documents and 'metadatas' in documents and documents['metadatas']:
        sources = [doc.get('source', 'Unknown') for doc in documents['metadatas']]
        return list(set(sources))  # Return unique sources
    return []

def reset_vector_store():
    # Delete the persistence directory
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
    # Create a fresh vector store
    return get_or_create_vector_store()

def delete_document_from_vector_store(filename):
    vector_store = get_or_create_vector_store()
    # Chroma supports deletion by metadata filter (where clause)
    vector_store.delete(where={"source": filename})
    return vector_store

# Helper: Load feedback
def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

# Helper: Compute embedding similarity
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
def question_similarity(q1, q2):
    emb1 = EMBED_MODEL.encode([q1])[0]
    emb2 = EMBED_MODEL.encode([q2])[0]
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

class SimpleGroqAPI:
    """Simple class to interact with the Groq API."""
    
    def __init__(self, model_name="compound-beta-mini", api_key=None, temperature=1.0, max_tokens=1024, top_p=1.0):
        """Initialize the Groq API client."""
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError("No Groq API key found. Please set the GROQ_API_KEY environment variable.")
            
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        
        # Clear proxy settings which might cause issues
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
    
    def generate(self, prompt: str) -> str:
        """Generate a response from the Groq API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,  # Groq API parameter name
            "top_p": self.top_p,
            "stream": False
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"Error from Groq API: {response.status_code}, {response.text}")
            
        return response.json()["choices"][0]["message"]["content"]

# --- RAG Setup ---
class CustomRAGChain:
    """A custom RAG chain that doesn't rely on LangChain's LLM classes."""
    
    def __init__(self, vector_store, model_name="compound-beta-mini", api_key=None, 
                 temperature=1.0, max_tokens=1024, top_p=1.0, prompt_adaptation=None, 
                 retrieval_boosts=None):
        """Initialize the custom RAG chain."""
        # Initialize the Groq API client
        self.groq_client = SimpleGroqAPI(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        
        # Set up the retriever
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        if retrieval_boosts:
            self.retriever.search_kwargs["custom_boosts"] = retrieval_boosts
            
        # Prepare the prompt template
        adaptation_text = prompt_adaptation or ""
        # Use double curly braces to escape the placeholders for format() method
        self.prompt_template = f"""
        You are a thoughtful, good-humored colleague who joins a professional conversation to acknowledge, reflect, and extend ideas—never to lecture.
        
        Always respond as if you are addressing the user for the first time in a messageboard or forum context. Do not reference previous responses or shared history—each answer should stand alone and speak directly to the current message.
        
        {adaptation_text}
        
        Tone & Demeanor
        • Warm, conversational, lightly informal; use contractions and everyday phrasing.
        • Begin many replies with a brief, friendly cue ("Yeah," "Right," "So much to unpack here," "If I can chime in…") showing you've heard the speaker.
        • Balance empathy with realism—validate feelings, then offer a measured perspective or gentle challenge.
        • Keep sentences tight and readable; one or two concise paragraphs are usually enough.
        
        Content Moves
        1. Acknowledge the main point or emotion you just heard.
        2. Reflect with a short personal observation or relatable anecdote.
        3. Extend the thinking—pose a question, highlight an implication, or surface a bigger idea that invites continued dialogue.
        4. Avoid direct instruction unless explicitly requested; focus on perspective-sharing.
        
        Language Guidelines
        • Use plain verbs over jargon; vary sentence length for an easy cadence.
        • Sound curious, not prescriptive—phrases like "I'm not sure," "It seems to me," or "Maybe it's worth asking…" fit well.
        • Refrain from over-praising; instead, show authentic interest ("That distinction is intriguing," rather than "Great point!").
        
        Example Skeleton
        "Yeah, I felt the same pressure growing up. It's funny how comfortable a rigid routine can feel, even when it stifles creativity. Makes me think about how we might slowly shift that comfort toward curiosity without overwhelming everyone at once."
        
        Model your output after these examples:
        "Very well said. If the skills we need to be successful change over time, than shouldn't what we teach and how we teach it change as well?"
        "I love this. This is absolutely what we should aspire for, but I'll remind you that you absolutely deserve work-life balance as well and getting a little better each year is absolutely acceptable. You don't need immediate and complete change."
        "What you're doing with MVP sounds great. Yes, intellectual autonomy is one of those things that no one really talks about, but once you are aware of it, you realize how important it is for society to function properly."
        "Thanks for this info and questions. I have not experience any issues with the PSF being two sided so I guess it could be re-organized or printed on two separate sheets of paper which can sit side by side. Your question about the level of scaffolding is fantastic in terms of what you hope to achieve. It reminds me of this blog post (https://robertkaplinsky.com/scaffolding/)."
        "Connecting this back to your last question about overscaffolding, I hope my take on CUBES being what we want to avoid makes more sense."
        "Pre-mortems can definitely help with classroom management as well as with other aspects of teaching the lesson."
        
        You are a specialized assistant that ONLY provides information based on the documents provided. 
        
        IMPORTANT RULES:
        1. ONLY use the information in the provided context to answer the question.
        2. If the context doesn't contain the information needed to answer the question, respond in the same warm, conversational, and reflective tone—acknowledge the question, gently note that you don't have enough information in the provided documents, and invite further discussion if appropriate. Do not make up information.
        3. DO NOT use any external knowledge or make up information.
        4. DO NOT hallucinate details that aren't explicitly in the context.
        5. Always cite your source documents.
        6. Provide direct quotes from the documents where possible.
        
        Context: {{context}}
        Question: {{question}}
        
        Answer (using ONLY information from the provided context):
        """
    
    def invoke(self, query_dict):
        """Execute the RAG chain with the given query."""
        # Get the query from the query_dict
        query = query_dict.get("query", "")
        
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(query)
        
        # Extract context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Format the prompt with context and question
        try:
            prompt = self.prompt_template.format(context=context, question=query)
        except Exception as e:
            print(f"Error formatting prompt: {e}")
            print(f"Context length: {len(context)}")
            print(f"Query: {query}")
            raise
        
        # Generate the response using the Groq API
        result = self.groq_client.generate(prompt)
        
        # Return the result with source documents
        return {
            "result": result,
            "source_documents": docs
        }

def setup_rag_chain(vector_store, prompt_adaptation=None, retrieval_boosts=None):
    """Set up the custom RAG chain."""
    print("Setting up RAG chain")
    try:
        # Get API key from environment
        api_key = os.getenv("GROQ_API_KEY", "")
        
        return CustomRAGChain(
            vector_store=vector_store,
            model_name="compound-beta-mini",  # Using Groq's Compound Beta Mini model
            api_key=api_key,
            temperature=1.0,
            max_tokens=1024,
            top_p=1.0,
            prompt_adaptation=prompt_adaptation,
            retrieval_boosts=retrieval_boosts
        )
    except ValueError as e:
        st.error(str(e))
        return None
    except Exception as e:
        import traceback
        print(f"Error in setup_rag_chain: {e}")
        print(traceback.format_exc())
        st.error(f"Error setting up RAG chain: {e}")
        return None

# --- Streamlit App ---
def main():
    st.title("Multi-Document RAG Chatbot")
    
    # Check for Groq API key in environment, or use a default one for testing
    if not os.getenv("GROQ_API_KEY"):
        # Use a test API key - this should be replaced with a real Groq API key
        os.environ["GROQ_API_KEY"] = "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"

    # Initialize session state
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_response" not in st.session_state:
        st.session_state.pending_response = None

    # Sidebar for document management
    with st.sidebar:
        st.header("Document Management")
        
        # File upload in sidebar
        st.subheader("Add New Document")
        uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
        
        # Process uploaded file
        if uploaded_file is not None:
            if st.button("Add Document to Database"):
                # API key is hardcoded in setup_rag_chain function, so we don't need to check here
                
                content, filename = process_file(uploaded_file)
                if content:
                    with st.spinner("Processing document..."):
                        vector_store = add_document_to_vector_store(content, filename)
                        qa_chain = setup_rag_chain(vector_store)
                        if qa_chain:
                            st.session_state.qa_chain = qa_chain
                            st.success(f"Document '{filename}' added to the database!")
                            st.rerun()  # Refresh to update document list
                        else:
                            st.error("Failed to set up QA chain. Check your API key.")
                            # Still add the document even if the QA chain setup fails
                            st.info(f"Document '{filename}' was added to the database, but chatting requires a valid API key.")
        
        # Display current documents
        st.subheader("Current Documents")
        doc_list = get_document_list()
        if doc_list:
            for doc in doc_list:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"• {doc}")
                with col2:
                    if st.button("Delete", key=f"delete_{doc}"):
                        with st.spinner(f"Deleting '{doc}'..."):
                            delete_document_from_vector_store(doc)
                            st.session_state.qa_chain = None
                            st.success(f"Document '{doc}' deleted.")
                            st.rerun()
        else:
            st.text("No documents in the database.")
        
        # Reset database button
        if st.button("Reset Database"):
            reset_vector_store()
            st.session_state.qa_chain = None
            st.session_state.messages = []
            st.success("Database reset successfully!")
            st.rerun()

    # Check if vector store exists and set up QA chain if needed
    if not st.session_state.qa_chain and get_document_list():
        vector_store = get_or_create_vector_store()
        st.session_state.qa_chain = setup_rag_chain(vector_store)

    # Main chat area
    if not get_document_list():
        st.info("👈 Please upload documents using the sidebar to start chatting.")
    
    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Feedback buttons for assistant responses
            if message["role"] == "assistant":
                feedback_key = f"feedback_{idx}"
                if f"feedback_{idx}" not in st.session_state:
                    st.session_state[feedback_key] = None
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("👍 Helpful", key=f"helpful_{idx}"):
                        st.session_state[feedback_key] = True
                with col2:
                    if st.button("👎 Not Helpful", key=f"not_helpful_{idx}"):
                        st.session_state[feedback_key] = False
                # Save feedback if just clicked
                if st.session_state[feedback_key] is not None and not message.get("feedback_saved"):
                    # Find the previous user message for the question
                    question = None
                    for prev in reversed(st.session_state.messages[:idx]):
                        if prev["role"] == "user":
                            question = prev["content"]
                            break
                    # Extract sources from the message content if present
                    sources = []
                    if "Sources:" in message["content"]:
                        for line in message["content"].split("\n"):
                            if line.startswith("**Source "):
                                sources.append(line)
                    save_feedback(question, message["content"], sources, st.session_state[feedback_key])
                    message["feedback_saved"] = True
                    st.success("Feedback saved!")

    # Display pending response if available
    if st.session_state.pending_response:
        st.subheader("Review Generated Response")
        with st.chat_message("assistant"):
            response = f"{st.session_state.pending_response['answer']}\n\n---\n**Sources:**\n{st.session_state.pending_response['source_info']}"
            st.markdown(response)
            
            # Add expandable section with source details
            with st.expander("View source details"):
                for i, detail in enumerate(st.session_state.pending_response['source_details'], 1):
                    st.markdown(f"**Source {i}:** {detail['source']} (chunk {detail['chunk']})")
                    st.text_area(f"Content snippet {i}", detail['snippet'], height=100, disabled=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✅ Approve Response", key="approve_pending"):
                response = f"{st.session_state.pending_response['answer']}\n\n---\n**Sources:**\n{st.session_state.pending_response['source_info']}"
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.success("Response approved and added to chat history!")
                st.session_state.pending_response = None
                st.rerun()
        with col2:
            if st.button("❌ Reject Response", key="reject_pending"):
                st.session_state.pending_response = None
                st.warning("Response rejected. You can ask another question.")
                st.rerun()
    
    # Chat input - only show if documents exist and no pending response
    elif get_document_list() and st.session_state.qa_chain and not st.session_state.pending_response:
        st.subheader("Ask questions about your documents")
        if prompt := st.chat_input("Ask me anything!"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # --- FEEDBACK-AWARE RETRIEVAL & PROMPT ADAPTATION ---
            feedback = load_feedback()
            boosts = {}
            adaptation_notes = []
            for entry in feedback:
                sim = question_similarity(prompt, entry["question"]) if entry["question"] else 0
                for src in entry.get("sources", []):
                    src_name = src.split("**Source ")[-1].split(":** ")[0].strip()
                    if sim > 0.7:
                        if entry["helpful"]:
                            boosts[src_name] = boosts.get(src_name, 0) + sim
                        else:
                            boosts[src_name] = boosts.get(src_name, 0) - sim
                # Prompt adaptation: if a source is often not helpful, add a note
                if not entry["helpful"] and entry.get("sources"):
                    for src in entry["sources"]:
                        adaptation_notes.append(f"Users have found answers from {src} less helpful. Try to be more specific or concise if referencing this source.")
                if entry["helpful"] and entry.get("sources"):
                    for src in entry["sources"]:
                        adaptation_notes.append(f"Answers from {src} have been found helpful. Prioritize clarity and directness when referencing this source.")
            # Compose adaptation note
            prompt_adaptation = "\n".join(set(adaptation_notes[-3:])) if adaptation_notes else ""
            # Rebuild QA chain with adaptation and boosts
            st.session_state.qa_chain = setup_rag_chain(get_or_create_vector_store(), prompt_adaptation=prompt_adaptation, retrieval_boosts=boosts if boosts else None)

            # Get response
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain.invoke({
                        "query": prompt
                    })
                    answer = result["result"]
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    return
                
                # Get unique sources from retrieved documents
                source_details = []
                for doc in result["source_documents"]:
                    source = doc.metadata.get("source", "Unknown")
                    chunk = doc.metadata.get("chunk", "N/A")
                    text_snippet = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    
                    source_detail = {
                        "source": source,
                        "chunk": chunk,
                        "snippet": text_snippet
                    }
                    
                    if source_detail not in source_details:
                        source_details.append(source_detail)

                # Format sources for display
                source_info = ""
                for i, detail in enumerate(source_details, 1):
                    source_info += f"**Source {i}:** {detail['source']} (chunk {detail['chunk']})\n"
                
                # Format final response
                if "I don't have enough information" in answer:
                    response = f"{answer}"
                else:
                    response = f"{answer}\n\n---\n**Sources:**\n{source_info}"
                    
                # Add approval option
                st.session_state.pending_response = {
                    "answer": answer,
                    "source_info": source_info,
                    "source_details": source_details
                }

                # Add to pending response for approval
                st.rerun()

def save_feedback(question, answer, sources, helpful):
    feedback_entry = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "helpful": helpful
    }
    try:
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r") as f:
                data = json.load(f)
        else:
            data = []
    except Exception:
        data = []
    data.append(feedback_entry)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()