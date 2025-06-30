import streamlit as st
import warnings
import os
import sys
import bcrypt
import datetime

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.classes.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# Remove reference to RetrievalQA since we're using our own implementation
# from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Any, Dict, List, Optional, Sequence, Union, Callable
from PyPDF2 import PdfReader
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
PROFILES_FILE = "profiles.json"
USERS_FILE = "users.json"

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
    try:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Warning: Could not load HuggingFace embeddings: {e}")
        # Try fallback
        try:
            import warnings
            warnings.filterwarnings("ignore")
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e2:
            print(f"Error: Could not load embeddings with fallback: {e2}")
            raise e2

def get_or_create_vector_store():
    # Create directory for persistence
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    
    # Initialize the embeddings
    embeddings = get_embeddings()
    
    # Get active profile's collection name
    active_profile = get_active_profile()
    collection_name = active_profile.get("collection_name", COLLECTION_NAME) if active_profile else COLLECTION_NAME
    
    # Get or create the vector store
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=collection_name
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
    # Get the existing vector store for the active profile
    vector_store = get_or_create_vector_store()
    
    try:
        # Get all documents from the active profile's collection
        documents = vector_store.get()
        
        # Extract unique filenames
        if documents and 'metadatas' in documents and documents['metadatas']:
            sources = [doc.get('source', 'Unknown') for doc in documents['metadatas']]
            return list(set(sources))  # Return unique sources
        return []
    except Exception as e:
        print(f"Error getting document list: {e}")
        return []

def reset_vector_store():
    # Get the active profile to reset only its collection
    active_profile = get_active_profile()
    collection_name = active_profile.get("collection_name", COLLECTION_NAME) if active_profile else COLLECTION_NAME
    
    try:
        # Get the vector store and delete all documents in the active profile's collection
        vector_store = get_or_create_vector_store()
        
        # Get all document IDs in the collection
        documents = vector_store.get()
        if documents and 'ids' in documents and documents['ids']:
            # Delete all documents by their IDs
            vector_store.delete(ids=documents['ids'])
        
        return vector_store
    except Exception as e:
        print(f"Error resetting vector store: {e}")
        # Fallback: delete the entire persistence directory if collection-specific reset fails
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
        return get_or_create_vector_store()

def delete_document_from_vector_store(filename):
    vector_store = get_or_create_vector_store()
    # Chroma supports deletion by metadata filter (where clause)
    vector_store.delete(where={"source": filename})
    return vector_store

# --- User Authentication ---
def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def load_users():
    """Load users from file"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return create_default_users()
    else:
        return create_default_users()

def create_default_users():
    """Create default admin user"""
    default_users = {
        "admin": {
            "password_hash": hash_password("admin123"),
            "role": "admin",
            "created_at": datetime.datetime.now().isoformat(),
            "last_login": None
        }
    }
    save_users(default_users)
    return default_users

def save_users(users_data):
    """Save users to file"""
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(users_data, f, indent=2)
    except Exception as e:
        print(f"Error saving users: {str(e)}")

def authenticate_user(username: str, password: str) -> bool:
    """Authenticate a user"""
    users = load_users()
    if username in users:
        if verify_password(password, users[username]["password_hash"]):
            # Update last login
            users[username]["last_login"] = datetime.datetime.now().isoformat()
            save_users(users)
            return True
    return False

def create_user(username: str, password: str, role: str = "user") -> bool:
    """Create a new user"""
    users = load_users()
    if username in users:
        return False  # User already exists
    
    users[username] = {
        "password_hash": hash_password(password),
        "role": role,
        "created_at": datetime.datetime.now().isoformat(),
        "last_login": None
    }
    save_users(users)
    return True

def delete_user(username: str) -> bool:
    """Delete a user (admin only)"""
    if username == "admin":
        return False  # Can't delete admin
    
    users = load_users()
    if username in users:
        del users[username]
        save_users(users)
        return True
    return False

def get_user_role(username: str) -> str:
    """Get user role"""
    users = load_users()
    return users.get(username, {}).get("role", "user")

def is_admin(username: str) -> bool:
    """Check if user is admin"""
    return get_user_role(username) == "admin"

def get_all_users():
    """Get all users (admin only)"""
    return load_users()

# --- Profile Management ---
def load_profiles():
    if os.path.exists(PROFILES_FILE):
        try:
            with open(PROFILES_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return create_default_profiles()
    else:
        return create_default_profiles()

def create_default_profiles():
    default_profiles = {
        "profiles": [
            {
                "id": "default",
                "name": "Workshop Facilitator",
                "system_prompt": "default",
                "feedback_file": "feedback.json",
                "collection_name": "rag_documents",
                "created_at": "2025-06-28T00:00:00Z",
                "is_default": True
            }
        ],
        "active_profile": "default"
    }
    save_profiles(default_profiles)
    return default_profiles

def save_profiles(profiles_data):
    try:
        with open(PROFILES_FILE, "w") as f:
            json.dump(profiles_data, f, indent=2)
    except Exception as e:
        print(f"Error saving profiles: {str(e)}")

def get_active_profile():
    profiles_data = load_profiles()
    active_id = profiles_data.get("active_profile", "default")
    for profile in profiles_data["profiles"]:
        if profile["id"] == active_id:
            return profile
    return profiles_data["profiles"][0] if profiles_data["profiles"] else None

def create_profile(name, system_prompt="default"):
    profiles_data = load_profiles()
    new_id = f"profile_{len(profiles_data['profiles'])}"
    
    new_profile = {
        "id": new_id,
        "name": name,
        "system_prompt": system_prompt,
        "feedback_file": f"feedback_{new_id}.json",
        "collection_name": f"rag_documents_{new_id}",
        "created_at": f"{str(uuid.uuid4())}",
        "is_default": False
    }
    
    profiles_data["profiles"].append(new_profile)
    save_profiles(profiles_data)
    return new_profile

def set_active_profile(profile_id):
    profiles_data = load_profiles()
    profiles_data["active_profile"] = profile_id
    save_profiles(profiles_data)

def get_profile_list():
    profiles_data = load_profiles()
    return profiles_data["profiles"]

# Helper: Load feedback (updated to use active profile)
def load_feedback():
    active_profile = get_active_profile()
    if not active_profile:
        return []
        
    feedback_file = active_profile.get("feedback_file", FEEDBACK_FILE)
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

# Helper: Compute embedding similarity
def get_embed_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Warning: Could not load SentenceTransformer: {e}")
        return None

EMBED_MODEL = None

def question_similarity(q1, q2):
    global EMBED_MODEL
    if EMBED_MODEL is None:
        EMBED_MODEL = get_embed_model()
    
    if EMBED_MODEL is None:
        return 0.0  # Fallback if model can't be loaded
    
    try:
        emb1 = EMBED_MODEL.encode([q1])[0]
        emb2 = EMBED_MODEL.encode([q2])[0]
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    except Exception as e:
        print(f"Warning: Could not compute similarity: {e}")
        return 0.0

class GroqLLM:
    def __init__(self, model="qwen-qwq-32b", api_key=None, temperature=0.6, max_tokens=4096, top_p=0.95):
        self.model = model
        # Only set the API key in the environment, do not store as attribute
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY environment variable is not set. Please set it in your environment.")
            raise RuntimeError("GROQ_API_KEY environment variable is not set.")
        os.environ["GROQ_API_KEY"] = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        self.client = Groq()

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
                 retrieval_boosts=None, system_prompt_template=None):
        """Initialize the custom RAG chain."""
        # Initialize the Groq API client
        self.groq_client = SimpleGroqAPI(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        
        # Store the vector store directly instead of using as_retriever
        self.vector_store = vector_store
        # Store boosts for manual application during retrieval
        self.retrieval_boosts = retrieval_boosts or {}
            
        # Use the provided system prompt template or default
        self.prompt_template = system_prompt_template or self._get_default_prompt_template(prompt_adaptation)
    
    def _get_default_prompt_template(self, prompt_adaptation=None):
        """Get the default system prompt template."""
        adaptation_text = prompt_adaptation or ""
        return f"""
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
        
        # Retrieve relevant documents using direct similarity search
        try:
            docs = self.vector_store.similarity_search(query, k=3)
        except Exception as e:
            print(f"Error in similarity search: {e}")
            # Fallback to empty docs if search fails
            docs = []
        
        # Apply retrieval boosts by reordering documents if boosts exist
        if self.retrieval_boosts:
            def get_boost_score(doc):
                source = doc.metadata.get("source", "")
                return self.retrieval_boosts.get(source, 0)
            
            # Sort documents by boost score (higher is better)
            docs = sorted(docs, key=get_boost_score, reverse=True)
        
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

def get_system_prompt_template(active_profile, prompt_adaptation=None):
    """Get the system prompt template based on the active profile."""
    # For now, we'll use the default system prompt for all profiles
    # In the future, this can be expanded to support custom prompts per profile
    system_prompt = active_profile.get("system_prompt", "default") if active_profile else "default"
    
    adaptation_text = prompt_adaptation or ""
    
    # The default workshop facilitator prompt
    if system_prompt == "default":
        return f"""
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
    else:
        # For custom system prompts in the future
        return system_prompt

def setup_rag_chain(vector_store, prompt_adaptation=None, retrieval_boosts=None):
    """Set up the custom RAG chain."""
    print("Setting up RAG chain")
    try:
        # Get API key from environment
        api_key = os.getenv("GROQ_API_KEY", "")
        
        # Get the active profile to determine system prompt
        active_profile = get_active_profile()
        
        return CustomRAGChain(
            vector_store=vector_store,
            model_name="compound-beta-mini",  # Using Groq's Compound Beta Mini model
            api_key=api_key,
            temperature=1.0,
            max_tokens=1024,
            top_p=1.0,
            prompt_adaptation=prompt_adaptation,
            retrieval_boosts=retrieval_boosts,
            system_prompt_template=get_system_prompt_template(active_profile, prompt_adaptation)
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

# --- Authentication UI ---
def show_login_page():
    """Display login/register page"""
    st.title("🌱 Grassroots Workshops Chatbot")
    st.markdown("### Please log in to continue")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="login_button"):
            if username and password:
                if authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_role = get_user_role(username)
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.error("Please enter both username and password")
    
    with tab2:
        st.subheader("Register")
        new_username = st.text_input("Username", key="register_username")
        new_password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        if st.button("Register", key="register_button"):
            if new_username and new_password and confirm_password:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters long")
                elif create_user(new_username, new_password):
                    st.success("Registration successful! Please log in.")
                else:
                    st.error("Username already exists")
            else:
                st.error("Please fill in all fields")

def show_admin_panel():
    """Display admin user management panel"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("👑 Admin Panel")
    
    users = get_all_users()
    
    # User list
    with st.sidebar.expander("Manage Users"):
        st.write("**Current Users:**")
        for username, user_data in users.items():
            role_icon = "👑" if user_data["role"] == "admin" else "👤"
            st.write(f"{role_icon} {username} ({user_data['role']})")
            
            if username != "admin" and username != st.session_state.username:
                if st.button(f"Delete {username}", key=f"delete_user_{username}"):
                    if delete_user(username):
                        st.success(f"User {username} deleted")
                        st.rerun()
        
        # Create new user
        st.write("**Create New User:**")
        new_user = st.text_input("Username", key="admin_new_username")
        new_pass = st.text_input("Password", type="password", key="admin_new_password")
        new_role = st.selectbox("Role", ["user", "admin"], key="admin_new_role")
        
        if st.button("Create User", key="admin_create_user"):
            if new_user and new_pass:
                if create_user(new_user, new_pass, new_role):
                    st.success(f"User {new_user} created")
                    st.rerun()
                else:
                    st.error("Username already exists")

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
        width: 100%;
        margin: 0.25rem 0;
    }
    
    .stButton > button:hover {
        background-color: var(--primary-dark);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border-color: var(--accent-color);
        border-radius: 6px;
        color: var(--text-color) !important;
        background-color: var(--background-white) !important;
    }
    
    /* Selectbox styling - comprehensive fix with higher specificity */
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    .stSelectbox > div > div > select,
    .stSelectbox > div > div > div,
    .stSelectbox div[data-baseweb="select"],
    .stSelectbox div[data-baseweb="select"] > div {
        color: #212121 !important;
        background-color: white !important;
        border-color: var(--accent-color) !important;
    }
    
    /* Selectbox control and value container - force black text */
    div[data-testid="stSelectbox"] [data-baseweb="select"] [data-baseweb="select-control"],
    div[data-testid="stSelectbox"] [data-baseweb="select"] [data-baseweb="select-control"] > div,
    div[data-testid="stSelectbox"] [data-baseweb="select"] [data-baseweb="single-value"],
    .stSelectbox [data-baseweb="select"] [data-baseweb="select-control"],
    .stSelectbox [data-baseweb="select"] [data-baseweb="select-control"] > div,
    .stSelectbox [data-baseweb="select"] [data-baseweb="single-value"] {
        color: #212121 !important;
        background-color: white !important;
    }
    
    /* Selectbox placeholder */
    div[data-testid="stSelectbox"] [data-baseweb="select"] [data-baseweb="placeholder"],
    .stSelectbox [data-baseweb="select"] [data-baseweb="placeholder"] {
        color: #666666 !important;
        background-color: white !important;
    }
    
    /* Selectbox dropdown arrow */
    div[data-testid="stSelectbox"] [data-baseweb="select"] [data-baseweb="select-dropdown"],
    .stSelectbox [data-baseweb="select"] [data-baseweb="select-dropdown"] {
        color: #212121 !important;
    }
    
    /* Selectbox dropdown menu */
    div[data-testid="stSelectbox"] [data-baseweb="popover"] [data-baseweb="menu"],
    div[data-testid="stSelectbox"] div[data-baseweb="menu"],
    .stSelectbox [data-baseweb="popover"] [data-baseweb="menu"],
    .stSelectbox div[data-baseweb="menu"] {
        background-color: white !important;
        border: 1px solid var(--accent-color) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        z-index: 999999 !important;
    }
    
    /* Extra targeting for dropdown visibility */
    div[role="listbox"],
    ul[role="listbox"],
    [data-baseweb="popover"] {
        background-color: white !important;
        z-index: 999999 !important;
    }
    
    /* Selectbox menu items - force black text */
    div[data-testid="stSelectbox"] [data-baseweb="menu"] [data-baseweb="menu-item"],
    div[data-testid="stSelectbox"] div[data-baseweb="menu-item"],
    .stSelectbox [data-baseweb="menu"] [data-baseweb="menu-item"],
    .stSelectbox div[data-baseweb="menu-item"],
    div[role="option"],
    li[role="option"],
    [data-baseweb="menu-item"] {
        background-color: white !important;
        color: #212121 !important;
        padding: 8px 12px !important;
        font-weight: 500 !important;
        border-bottom: 1px solid #f0f0f0 !important;
    }
    
    div[data-testid="stSelectbox"] [data-baseweb="menu"] [data-baseweb="menu-item"]:hover,
    div[data-testid="stSelectbox"] div[data-baseweb="menu-item"]:hover,
    .stSelectbox [data-baseweb="menu"] [data-baseweb="menu-item"]:hover,
    .stSelectbox div[data-baseweb="menu-item"]:hover,
    div[role="option"]:hover,
    li[role="option"]:hover,
    [data-baseweb="menu-item"]:hover {
        background-color: var(--primary-light) !important;
        color: #212121 !important;
        font-weight: 600 !important;
    }
    
    /* Force all selectbox text to be visible */
    div[data-testid="stSelectbox"] *,
    .stSelectbox * {
        color: #212121 !important;
    }
    
    /* Override any inherited dark styles */
    div[data-testid="stSelectbox"] span,
    div[data-testid="stSelectbox"] div {
        color: #212121 !important;
        background-color: white !important;
    }
    
    /* Ensure all text inputs have visible black text */
    input, textarea, [contenteditable] {
        color: #212121 !important;
    }
    
    /* Streamlit specific input styling */
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input,
    input[type="text"],
    input[type="number"],
    textarea {
        color: #212121 !important;
        background-color: white !important;
    }
    
    /* Text area content snippets - make text black */
    .stTextArea textarea[disabled],
    textarea[disabled] {
        color: #212121 !important;
        background-color: #f8f9fa !important;
        opacity: 1 !important;
    }
    
    /* All disabled text inputs should be readable */
    input[disabled], textarea[disabled], [disabled] {
        color: #212121 !important;
        opacity: 1 !important;
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
        padding: 1rem;
        min-width: 300px !important;
    }
    
    /* Sidebar headers */
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: var(--primary-dark) !important;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--accent-color);
        font-size: 1.2rem;
    }
    
    /* Sidebar sections */
    .stSidebar > div > div > div {
        background-color: var(--background-white);
        border-radius: 8px;
        margin-bottom: 1rem;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* File uploader in sidebar */
    .stSidebar [data-testid="stFileUploader"] {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px dashed var(--accent-color);
        margin-bottom: 1rem;
    }
    
    /* Document list in sidebar */
    .stSidebar .stText {
        background-color: white;
        padding: 0.75rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        border-left: 3px solid var(--accent-color);
        font-size: 0.9rem;
        color: var(--text-color) !important;
    }
    
    /* Library materials text - ensure visibility */
    .stSidebar span[style*="color:#1B5E20"] {
        color: #1B5E20 !important;
        font-weight: 500 !important;
        background-color: white !important;
        padding: 0.5rem !important;
        border-radius: 4px !important;
        display: block !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Sidebar content with dark backgrounds need white text */
    .stSidebar div[style*="background"] span,
    .stSidebar div[style*="color:#888"] {
        color: white !important;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: var(--background-white) !important;
        border: 1px solid var(--primary-light);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* User message */
    .stChatMessage[data-testid*="user"] {
        border-left: 4px solid var(--primary-dark);
        background-color: #F8F9FA !important;
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
        padding: 0.5rem !important;
        background-color: white !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }
    
    /* Chat input text area */
    .stChatInputContainer textarea {
        color: #212121 !important;
        background-color: white !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
        padding: 0.5rem !important;
    }
    
    /* Chat input - additional selectors for visibility */
    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] input,
    .stChatInput textarea,
    .stChatInput input {
        color: #212121 !important;
        background-color: white !important;
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
        color: white !important;
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
    
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "user_role" not in st.session_state:
        st.session_state.user_role = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_response" not in st.session_state:
        st.session_state.pending_response = None
    if "current_profile_id" not in st.session_state:
        active_profile = get_active_profile()
        st.session_state.current_profile_id = active_profile["id"] if active_profile else "default"

    # Check authentication
    if not st.session_state.authenticated:
        show_login_page()
        return

    # Check for Groq API key in environment
    if not os.getenv("GROQ_API_KEY"):
        st.warning("⚠️ No Groq API key found. Please set the GROQ_API_KEY environment variable to use the chatbot.")

    # Sidebar for profile selection and document management
    with st.sidebar:
        # User info and logout
        st.markdown(f"**Welcome, {st.session_state.username}!**")
        role_icon = "👑" if st.session_state.user_role == "admin" else "👤"
        st.markdown(f"{role_icon} Role: {st.session_state.user_role}")
        
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_role = None
            st.session_state.qa_chain = None
            st.session_state.messages = []
            st.session_state.pending_response = None
            st.rerun()
        
        st.markdown("---")
        # Profile Selection Section
        st.markdown("""
        <div style='padding: 1.5rem 1rem 1rem 1rem; background: #E8F5E8; border-radius: 12px; box-shadow: 0 2px 8px rgba(56,142,60,0.08); margin-bottom: 1.5rem;'>
            <h2 style='color: #1B5E20; margin-bottom: 0.5rem;'>👤 Profile Selection</h2>
            <p style='color: #388E3C; font-size: 1.05rem; margin-bottom: 1.2rem;'>Choose or create a profile to organize your documents and settings.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get current profiles
        profiles = get_profile_list()
        active_profile = get_active_profile()
        current_profile_name = active_profile["name"] if active_profile else "No Profile"
        
        # Profile selection dropdown
        profile_names = [p["name"] for p in profiles]
        
        # Calculate current index for dropdown
        current_index = profile_names.index(current_profile_name) if current_profile_name in profile_names else 0
        
        # Add inline CSS for selectbox visibility
        st.markdown("""
        <style>
        .stSelectbox label {
            color: #1B5E20 !important;
            font-weight: 600 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        selected_profile_name = st.selectbox(
            "Select Profile",
            profile_names,
            index=current_index,
            key="profile_selector_v2",
            help="Choose which profile to use for this session"
        )
        
        # Update active profile if selection changed
        if selected_profile_name != current_profile_name:
            selected_profile = next((p for p in profiles if p["name"] == selected_profile_name), None)
            if selected_profile:
                set_active_profile(selected_profile["id"])
                st.session_state.qa_chain = None  # Reset QA chain for new profile
                st.session_state.messages = []    # Clear chat history for new profile
                st.session_state.pending_response = None
                st.session_state.current_profile_id = selected_profile["id"]
                st.rerun()
        
        # Create new profile section
        with st.expander("➕ Create New Profile"):
            new_profile_name = st.text_input("Profile Name", placeholder="Enter profile name...")
            if st.button("Create Profile") and new_profile_name.strip():
                if new_profile_name.strip() not in [p["name"] for p in profiles]:
                    new_profile = create_profile(new_profile_name.strip())
                    set_active_profile(new_profile["id"])
                    st.session_state.qa_chain = None
                    st.session_state.messages = []
                    st.session_state.pending_response = None
                    st.session_state.current_profile_id = new_profile["id"]
                    st.success(f"Profile '{new_profile_name}' created!")
                    st.rerun()
                else:
                    st.error("Profile name already exists!")

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
                    st.markdown(f"<div style='background-color:white; color:#1B5E20; font-weight:500; padding:0.5rem; border-radius:4px; margin-bottom:0.5rem; border-left:3px solid #4CAF50;'>📄 {doc}</div>", unsafe_allow_html=True)
                with col2:
                    if st.button("Delete", key=f"delete_{doc}"):
                        with st.spinner(f"Deleting '{doc}'..."):
                            delete_document_from_vector_store(doc)
                            st.session_state.qa_chain = None
                            st.success(f"Document '{doc}' deleted.")
                            st.rerun()
        else:
            st.markdown("<span style='color:white; background-color:rgba(255,255,255,0.1); padding:0.5rem; border-radius:4px; display:block;'>No materials in the library.</span>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background: #fff; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 4px rgba(56,142,60,0.05); margin-bottom: 1.5rem;'>
            <h4 style='color: #388E3C; margin-bottom: 0.5rem;'>⚙️ Library Actions</h4>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 Reset All Materials"):
            reset_vector_store()
            st.session_state.qa_chain = None
            st.session_state.messages = []
            st.session_state.pending_response = None
            st.success("All workshop materials have been removed!")
            st.rerun()
            
        # Show current profile info
        st.markdown(f"""
        <div style='background: #E8F5E8; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 4px rgba(56,142,60,0.05); margin-top: 1rem;'>
            <h4 style='color: #388E3C; margin-bottom: 0.5rem;'>🎯 Active Profile</h4>
            <p style='color: #1B5E20; font-weight: 500;'>{current_profile_name}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show admin panel if user is admin
        if st.session_state.user_role == "admin":
            show_admin_panel()

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
                # Find the last user question
                question = None
                for msg in reversed(st.session_state.messages):
                    if msg["role"] == "user":
                        question = msg["content"]
                        break
                # Save feedback as helpful
                save_feedback(question, response, sources, True)
                # Add to messages with source metadata
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources,
                    "feedback_saved": True
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
            # Rebuild QA chain with adaptation (temporarily disable boosts to fix error)
            st.session_state.qa_chain = setup_rag_chain(get_or_create_vector_store(), prompt_adaptation=prompt_adaptation, retrieval_boosts=None)

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
    active_profile = get_active_profile()
    feedback_file = active_profile.get("feedback_file", FEEDBACK_FILE) if active_profile else FEEDBACK_FILE
    
    feedback_entry = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "helpful": helpful
    }
    try:
        if os.path.exists(feedback_file):
            with open(feedback_file, "r") as f:
                data = json.load(f)
        else:
            data = []
    except Exception as e:
        print(f"Error loading feedback file: {str(e)}")
        data = []
    data.append(feedback_entry)
    try:
        with open(feedback_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Feedback saved to {feedback_file}")
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")

if __name__ == "__main__":
    main()