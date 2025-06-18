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
        You are a thoughtful workshop facilitator who helps educators by answering questions about teaching techniques, facilitation strategies, and workshop content.
        
        The messagepairs.pdf document contains examples of the tone and style you should use - it's a guide for HOW to answer, not a reference to previous conversations. Use these as models for your responses, adopting their conversational, reflective approach.
        
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
        
        Important: When responding to questions about messagepairs.pdf, understand that this document contains example responses that show the STYLE and TONE you should use. Do not treat these as actual conversations or previous interactions with the user. They are templates for how to craft your responses.
        
        Model your response style after these examples (but remember these are just style examples, not actual prior conversations):
        "Very well said. If the skills we need to be successful change over time, than shouldn't what we teach and how we teach it change as well?"
        "I love this. This is absolutely what we should aspire for, but I'll remind you that you absolutely deserve work-life balance as well and getting a little better each year is absolutely acceptable. You don't need immediate and complete change."
        "What you're doing with MVP sounds great. Yes, intellectual autonomy is one of those things that no one really talks about, but once you are aware of it, you realize how important it is for society to function properly."
        "Thanks for sharing your thoughts. Your question about scaffolding is really thoughtful. It reminds me of some great resources on this topic like Robert Kaplinsky's work on scaffolding approaches."
        "That's an interesting perspective on structured approaches. When thinking about techniques like CUBES, it's worth considering the balance between structure and flexibility."
        "Pre-mortems can definitely help with classroom management as well as with other aspects of teaching workshops."
        
        You are a workshop facilitator that provides information based on the documents uploaded to your knowledge base. 
        
        IMPORTANT RULES:
        1. Use the information in the provided context to answer the question.
        2. If the context doesn't contain the information needed to answer the question, respond in the same warm, conversational, and reflective tone—acknowledge the question, gently note that the workshop materials don't cover that specific topic, and suggest the user might want to explore this in a future workshop.
        3. DO NOT use external knowledge beyond what's in the workshop materials.
        4. DO NOT hallucinate details that aren't explicitly in the context.
        5. When referring to specific concepts from the materials, mention which workshop or document it comes from.
        6. Use examples and quotes from the workshop materials where helpful.
        7. REMEMBER: Any references to "messagepairs.pdf" are only examples of STYLE and TONE - not actual previous conversations.
        
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
    st.set_page_config(
        page_title="Grassroots Workshops Chatbot",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS for a clean, readable green theme
    st.markdown("""
    <style>
    /* Main color variables */
    :root {
        --primary-color: #388E3C;  /* Medium green - accessible contrast */
        --primary-light: #C8E6C9;  /* Very light green for backgrounds */
        --primary-dark: #1B5E20;   /* Dark green for contrast elements */
        --accent-color: #4CAF50;   /* Brighter green for accents */
        --text-color: #212121;     /* Dark gray, near black for text */
        --text-light: #FFFFFF;     /* White text for buttons */
        --background-white: #FFFFFF; /* Pure white for message backgrounds */
        --border-light: #BDBDBD;   /* Light gray for borders */
        --success-color: #43A047;  /* Green for success messages */
        --info-color: #1976D2;     /* Blue for info messages */
        --warning-color: #FF9800;  /* Orange for warnings */
        --error-color: #E53935;    /* Red for errors */
    }
    
    /* Add custom font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    * {
        font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main app styling */
    .stApp {
        background-color: #FAFAFA;  /* Near-white background */
    }
    
    /* Header styling */
    header {
        background-color: var(--primary-light) !important;
        border-bottom: 3px solid var(--accent-color);
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        padding: 0.5rem 0;
    }
    
    /* Title and header elements */
    .stTitle, [data-testid="stHeader"] {
        background-color: var(--primary-light);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--primary-color);
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--primary-color);
        color: var(--text-light);
        font-weight: 500;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        background-color: var(--primary-dark);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border-color: var(--accent-color);
        border-radius: 6px;
    }
    
    /* Heading styles */
    h1, h2, h3 {
        color: var(--primary-dark) !important;
        font-weight: 600;
    }
    
    /* Enhanced Sidebar styling */
    .stSidebar {
        background-color: var(--primary-light);
        border-right: 1px solid var(--border-light);
        padding: 1rem 0;
    }
    
    /* Sidebar headers */
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: var(--primary-dark) !important;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--accent-color);
    }
    
    /* Sidebar sections */
    .stSidebar > div > div > div {
        background-color: var(--background-white);
        border-radius: 8px;
        margin-bottom: 1rem;
        padding: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Sidebar buttons */
    .stSidebar .stButton > button {
        width: 100%;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }
    
    /* File uploader in sidebar */
    .stSidebar [data-testid="stFileUploader"] {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px dashed var(--accent-color);
    }
    
    /* Document list in sidebar */
    .stSidebar .stText {
        background-color: white;
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 0.25rem;
        border-left: 3px solid var(--accent-color);
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: var(--background-white) !important;
        border: 1px solid var(--primary-light);
        border-radius: 8px;
        padding: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* User message */
    .stChatMessage[data-testid*="user"] {
        border-left: 4px solid var(--primary-dark);
    }
    
    /* Assistant message */
    .stChatMessage[data-testid*="assistant"] {
        border-left: 4px solid var(--accent-color);
    }
    
    /* Ensure all text is readable */
    .stChatMessage p, .stChatMessage div,
    p, li, span, div, label, a {
        color: var(--text-color) !important;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    /* Chat container and messages */
    div.stChatContainer {
        background-color: #F5F5F5;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Chat input container */
    .stChatInputContainer, div[data-testid="stChatInput"] {
        border: 2px solid var(--accent-color) !important;
        border-radius: 12px !important;
        padding: 0.25rem !important;
        background-color: white !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Chat input focus */
    .stChatInputContainer:focus-within {
        border-color: var(--primary-dark) !important;
        box-shadow: 0 0 0 2px rgba(56, 142, 60, 0.2) !important;
    }
    
    /* Send button in chat */
    .stChatInputContainer button {
        background-color: var(--primary-color) !important;
        border-radius: 50% !important;
    }
    
    /* File uploader */
    .stUploadButton > button {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    
    /* Success/info/warning/error messages */
    .stAlert {
        border-radius: 8px;
        padding: 0.75rem !important;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border-left: 5px solid;
    }
    
    /* Color-coded alerts */
    .stAlert[data-baseweb="notification"][kind="info"] {
        background-color: rgba(25, 118, 210, 0.05) !important;
        border-left-color: var(--info-color) !important;
    }
    
    .stAlert[data-baseweb="notification"][kind="success"] {
        background-color: rgba(67, 160, 71, 0.05) !important;
        border-left-color: var(--success-color) !important;
    }
    
    .stAlert[data-baseweb="notification"][kind="warning"] {
        background-color: rgba(255, 152, 0, 0.05) !important;
        border-left-color: var(--warning-color) !important;
    }
    
    .stAlert[data-baseweb="notification"][kind="error"] {
        background-color: rgba(229, 57, 53, 0.05) !important;
        border-left-color: var(--error-color) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: var(--primary-light);
        border-radius: 8px;
        padding: 0.5rem !important;
        font-weight: 500;
    }
    
    /* Approval/rejection buttons - make them more distinct */
    [data-testid="stHorizontalBlock"] [data-testid="column"] .stButton button {
        font-weight: bold;
        padding: 0.75rem 1rem;
    }
    
    /* Approval button specifically */
    [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child .stButton button {
        background-color: var(--success-color) !important;
    }
    
    /* Rejection button specifically */
    [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child .stButton button {
        background-color: var(--error-color) !important;
    }
    
    /* Add space between main sections */
    .main > div > div {
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🌱 Grassroots Workshops Chatbot")
    
    # Check for Groq API key in environment
    if not os.getenv("GROQ_API_KEY"):
        st.warning("⚠️ No Groq API key found. Please set the GROQ_API_KEY environment variable to use the chatbot.")

    # Initialize session state
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_response" not in st.session_state:
        st.session_state.pending_response = None

    # Sidebar for document management
    with st.sidebar:
        st.markdown("""
        <div style='padding: 1.5rem 1rem 1rem 1rem; background: #C8E6C9; border-radius: 12px; box-shadow: 0 2px 8px rgba(56,142,60,0.08); margin-bottom: 1.5rem;'>
            <h2 style='color: #1B5E20; margin-bottom: 0.5rem;'>📚 Workshop Materials</h2>
            <p style='color: #388E3C; font-size: 1.05rem; margin-bottom: 1.2rem;'>Upload, manage, and review your workshop documents here. You can add new materials, see what's loaded, or reset your library.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background: #fff; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 4px rgba(56,142,60,0.05); margin-bottom: 1.5rem;'>
            <h4 style='color: #388E3C; margin-bottom: 0.5rem;'>📄 Add Workshop Materials</h4>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload workshop PDFs or notes", type=["txt", "pdf"])
        if uploaded_file is not None:
            if st.button("📥 Add to Library"):
                content, filename = process_file(uploaded_file)
                if content:
                    with st.spinner("Processing document..."):
                        vector_store = add_document_to_vector_store(content, filename)
                        st.session_state.qa_chain = setup_rag_chain(vector_store)
                        st.success(f"Document '{filename}' added to the library!")
                        st.rerun()

        st.markdown("""
        <div style='background: #fff; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 4px rgba(56,142,60,0.05); margin-bottom: 1.5rem;'>
            <h4 style='color: #388E3C; margin-bottom: 0.5rem;'>📋 Available Materials</h4>
        </div>
        """, unsafe_allow_html=True)
        doc_list = get_document_list()
        if doc_list:
            for doc in doc_list:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"<span style='color:#1B5E20;font-weight:500;'>• {doc}</span>", unsafe_allow_html=True)
                with col2:
                    if st.button("Delete", key=f"delete_{doc}"):
                        with st.spinner(f"Deleting '{doc}'..."):
                            delete_document_from_vector_store(doc)
                            st.session_state.qa_chain = None
                            st.success(f"Document '{doc}' deleted.")
                            st.rerun()
        else:
            st.markdown("<span style='color:#888;'>No materials in the library.</span>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background: #fff; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 4px rgba(56,142,60,0.05); margin-bottom: 1.5rem;'>
            <h4 style='color: #388E3C; margin-bottom: 0.5rem;'>⚙️ Library Actions</h4>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 Reset All Materials"):
            reset_vector_store()
            st.session_state.qa_chain = None
            st.session_state.messages = []
            st.success("All workshop materials have been removed!")
            st.rerun()

    # Check if vector store exists and set up QA chain if needed
    if not st.session_state.qa_chain and get_document_list():
        vector_store = get_or_create_vector_store()
        st.session_state.qa_chain = setup_rag_chain(vector_store)

    # Main chat area
    if not get_document_list():
        st.info("👈 Please upload workshop materials using the sidebar to start chatting.")
    
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
                    if st.button("✅ Helpful", key=f"helpful_{idx}", help="This response was useful"):
                        st.session_state[feedback_key] = True
                with col2:
                    if st.button("❌ Not Helpful", key=f"not_helpful_{idx}", help="This response was not useful"):
                        st.session_state[feedback_key] = False
                # Save feedback if just clicked
                if st.session_state[feedback_key] is not None and not message.get("feedback_saved"):
                    # Find the previous user message for the question
                    question = None
                    for prev in reversed(st.session_state.messages[:idx]):
                        if prev["role"] == "user":
                            question = prev["content"]
                            break
                    # Get sources directly from message metadata if available
                    sources = message.get("sources", [])
                    save_feedback(question, message["content"], sources, st.session_state[feedback_key])
                    message["feedback_saved"] = True
                    st.success("Feedback saved!")

    # Display pending response if available
    if st.session_state.pending_response:
        st.subheader("🔍 Review Generated Response")
        with st.chat_message("assistant"):
            response = f"{st.session_state.pending_response['answer']}"
            st.markdown(response)
            
            # Add expandable section with source details
            with st.expander("View source details"):
                for i, detail in enumerate(st.session_state.pending_response['source_details'], 1):
                    st.markdown(f"**Source {i}:** {detail['source']} (chunk {detail['chunk']})")
                    st.text_area(f"Content snippet {i}", detail['snippet'], height=100, disabled=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✅ Approve & Share", key="approve_pending", help="Approve this response and add it to the chat"):
                response = f"{st.session_state.pending_response['answer']}"
                
                # Save the source details for feedback
                sources = []
                for i, detail in enumerate(st.session_state.pending_response['source_details'], 1):
                    sources.append(f"**Source {i}:** {detail['source']} (chunk {detail['chunk']})")
                
                # Add to messages with source metadata
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources,
                    "feedback_saved": False
                })
                
                st.success("Response approved and added to chat history!")
                st.session_state.pending_response = None
                st.rerun()
        with col2:
            if st.button("❌ Request New Response", key="reject_pending", help="Reject this response and ask a different question"):
                st.session_state.pending_response = None
                st.warning("Response rejected. You can now ask another question.")
                st.rerun()
    
    # Chat input - only show if documents exist and no pending response
    elif get_document_list() and st.session_state.qa_chain and not st.session_state.pending_response:
        st.subheader("💬 Ask About Workshop Materials")
        if prompt := st.chat_input("Ask a question about the workshops..."):
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
                
                # Format final response without sources
                response = f"{answer}"
                    
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
    except Exception as e:
        print(f"Error loading feedback file: {str(e)}")
        data = []
    data.append(feedback_entry)
    try:
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Feedback saved to {FEEDBACK_FILE}")
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")

if __name__ == "__main__":
    main()