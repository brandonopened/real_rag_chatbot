#!/usr/bin/env python3
"""
FastAPI service for the RAG Chatbot
Provides REST API endpoints for querying the chatbot programmatically
"""

import os
import sys
import warnings
import bcrypt
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.classes.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import existing chatbot functionality
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import requests

# Constants
PERSIST_DIRECTORY = "./data/chroma_db"
COLLECTION_NAME = "rag_documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
FEEDBACK_FILE = "feedback.json"
PROFILES_FILE = "profiles.json"
USERS_FILE = "users.json"
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-this")
JWT_ALGORITHM = "HS256"

app = FastAPI(title="RAG Chatbot API", description="API for the RAG Chatbot system")
security = HTTPBearer()

# Pydantic models
class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    query: str
    profile_name: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    profile_used: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

# Copy necessary functions from ragchatbot.py
def load_users():
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def load_profiles():
    try:
        with open(PROFILES_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def authenticate_user(username: str, password: str) -> bool:
    users = load_users()
    if username not in users:
        return False
    
    stored_password = users[username]['password']
    if isinstance(stored_password, str):
        stored_password = stored_password.encode('utf-8')
    
    return bcrypt.checkpw(password.encode('utf-8'), stored_password)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vector_store(collection_name: str = COLLECTION_NAME):
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=collection_name
    )

def make_groq_request(messages: List[Dict]) -> str:
    """Make request to Groq API"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "max_tokens": 4000,
        "temperature": 0.7
    }
    
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Error from Groq API")
    
    return response.json()["choices"][0]["message"]["content"]

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/auth/login", response_model=TokenResponse)
async def login(login_request: LoginRequest):
    """Authenticate user and return JWT token"""
    if not authenticate_user(login_request.username, login_request.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    access_token = create_access_token(data={"sub": login_request.username})
    return TokenResponse(access_token=access_token, token_type="bearer")

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest, username: str = Depends(verify_token)):
    """Query the RAG chatbot"""
    profiles = load_profiles()
    
    # Get profile
    if chat_request.profile_name:
        profile = next((p for p in profiles if p["name"] == chat_request.profile_name), None)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
    else:
        profile = profiles[0] if profiles else {
            "name": "default",
            "system_prompt": "You are a helpful assistant.",
            "collection_name": COLLECTION_NAME
        }
    
    # Get vector store for profile
    collection_name = profile.get("collection_name", COLLECTION_NAME)
    vector_store = get_vector_store(collection_name)
    
    # Retrieve relevant documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(chat_request.query)
    
    # Build context
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = [doc.metadata.get("source", "Unknown") for doc in docs]
    
    # Create messages for Groq
    system_message = profile["system_prompt"]
    if context:
        system_message += f"\n\nUse the following context to answer the question:\n{context}"
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": chat_request.query}
    ]
    
    # Get response from Groq
    response = make_groq_request(messages)
    
    return ChatResponse(
        response=response,
        sources=list(set(sources)),
        profile_used=profile["name"]
    )

@app.get("/profiles")
async def list_profiles(username: str = Depends(verify_token)):
    """List available chatbot profiles"""
    profiles = load_profiles()
    return [{"name": p["name"], "description": p.get("description", "")} for p in profiles]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)