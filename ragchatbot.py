import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
import os
import uuid
import shutil

# Constants
PERSIST_DIRECTORY = "./data/chroma_db"
COLLECTION_NAME = "rag_documents"

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
    
    # Add document to vector store
    vector_store.add_texts(
        texts=[content],
        metadatas=[{"source": filename}]
    )
    
    # Persistence is now handled automatically by Chroma
    
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

# --- RAG Setup ---
def setup_rag_chain(vector_store):
    llm = OllamaLLM(model="llama3.1:latest")
    
    prompt_template = """
    Answer the question based on the provided context. If asked, indicate the document source.
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # Retrieve from top 3 documents
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

# --- Streamlit App ---
def main():
    st.title("Multi-Document RAG Chatbot")

    # Initialize session state
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar for document management
    with st.sidebar:
        st.header("Document Management")
        
        # File upload in sidebar
        st.subheader("Add New Document")
        uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
        
        # Process uploaded file
        if uploaded_file is not None:
            if st.button("Add Document to Database"):
                content, filename = process_file(uploaded_file)
                if content:
                    with st.spinner("Processing document..."):
                        vector_store = add_document_to_vector_store(content, filename)
                        st.session_state.qa_chain = setup_rag_chain(vector_store)
                        st.success(f"Document '{filename}' added to the database!")
                        st.rerun()  # Refresh to update document list
        
        # Display current documents
        st.subheader("Current Documents")
        doc_list = get_document_list()
        if doc_list:
            for doc in doc_list:
                st.text(f"• {doc}")
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
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input - only show if documents exist
    if get_document_list() and st.session_state.qa_chain:
        st.subheader("Ask questions about your documents")
        if prompt := st.chat_input("Ask me anything!"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"query": prompt})  # Use invoke instead of call
                answer = result["result"]
                
                # Get unique sources from retrieved documents
                sources = []
                for doc in result["source_documents"]:
                    source = doc.metadata.get("source", "Unknown")
                    if source not in sources:
                        sources.append(source)

                # Format source information
                source_info = ", ".join(sources)

                # Handle source request
                if "which document" in prompt.lower() or "source" in prompt.lower():
                    response = f"{answer}\n\n**Sources:** {source_info}"
                else:
                    response = f"{answer}\n\n*(Sources: {source_info})*"

                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)

if __name__ == "__main__":
    main()