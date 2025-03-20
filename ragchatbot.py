import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
import os

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
def create_vector_store(content, filename):
    # Create directory for persistence
    os.makedirs("./data", exist_ok=True)
    
    # Initialize the embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create the vector store
    texts = [content]
    metadatas = [{"source": filename}]
    
    vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local("./data/faiss_index")
    
    return vector_store

# --- RAG Setup ---
def setup_rag_chain(vector_store):
    llm = Ollama(model="llama3.1:latest")
    
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
        retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

# --- Streamlit App ---
def main():
    st.title("Simple RAG Chatbot")

    # Initialize session state
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False

    # File upload
    uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

    # Process file when uploaded
    if uploaded_file is not None:
        if st.button("Process File") or st.session_state.file_processed:
            if not st.session_state.file_processed:
                content, filename = process_file(uploaded_file)
                if content:
                    with st.spinner("Processing file..."):
                        vector_store = create_vector_store(content, filename)
                        st.session_state.qa_chain = setup_rag_chain(vector_store)
                        st.session_state.file_processed = True
                        st.success(f"File '{filename}' uploaded and indexed successfully!")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input - only show if a file has been processed
    if st.session_state.file_processed and st.session_state.qa_chain:
        st.subheader("Ask questions about your document")
        if prompt := st.chat_input("Ask me anything!"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain({"query": prompt})
                answer = result["result"]
                source = result["source_documents"][0].metadata["source"]

                # Handle source request
                if "which document" in prompt.lower() or "where" in prompt.lower():
                    response = f"{answer}\n\n**Source:** {source}"
                else:
                    response = f"{answer}\n\n*(Ask 'which document' to see the source.)*"

                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
    elif not st.session_state.file_processed:
        st.warning("Please upload a file and click 'Process File' to start chatting.")

if __name__ == "__main__":
    main()