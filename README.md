# Multi-Document RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows you to upload and query multiple documents using a local LLM through Ollama.

## Features

- Upload and process multiple PDF and text documents
- Persistent document storage between sessions
- Query across all uploaded documents simultaneously
- Chat interface with conversation history
- Document source tracking for answers
- Local processing using Ollama (no data sent to external services)

## Prerequisites

1. **Python 3.9+** installed on your system
2. **Ollama** installed with Llama 3.1 model (see [Ollama installation instructions](#installing-ollama))

## Installation

1. Clone this repository or download the source code:
   ```bash
   git clone https://github.com/yourusername/real_rag_chatbot.git
   cd real_rag_chatbot
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Make sure Ollama is running with the Llama 3.1 model:
   ```bash
   ollama run llama3.1
   ```
   You can keep this running in a separate terminal.

2. Start the Streamlit application:
   ```bash
   streamlit run ragchatbot.py
   ```

3. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

## Using the Chatbot

1. **Adding Documents**:
   - In the left sidebar, under "Add New Document", click "Browse files" to select a PDF or text file
   - Click "Add Document to Database" to process and index the document
   - The document will appear in the "Current Documents" list

2. **Querying Documents**:
   - Once documents are added, the chat interface will be available in the main area
   - Type your questions in the "Ask me anything!" input box at the bottom
   - The response will include information from all relevant documents
   - The source documents will be listed below each answer

3. **Managing Documents**:
   - To clear all documents and chat history, click the "Reset Database" button in the sidebar

## How It Works

This application uses a RAG (Retrieval-Augmented Generation) architecture:

1. **Indexing**:
   - Documents are processed and split into chunks
   - Each chunk is converted into vector embeddings using HuggingFace's all-MiniLM-L6-v2 model
   - These vectors are stored in a ChromaDB vector database

2. **Retrieval**:
   - When you ask a question, it's converted to the same vector space
   - The system finds the most similar document chunks to your question
   - Top matching chunks are retrieved as context

3. **Generation**:
   - Retrieved context is sent to the Llama 3.1 LLM along with your question
   - The LLM generates a response based on the retrieved context
   - Source documents are tracked and displayed with the answer

## Installing Ollama

1. Visit [ollama.com](https://ollama.com/) and download the installer for your platform
2. Install Ollama following the platform-specific instructions
3. Pull the Llama 3.1 model:
   ```bash
   ollama pull llama3.1
   ```

## Customization

You can modify the code to:
- Use different LLM models by changing the model name in the `setup_rag_chain()` function
- Adjust the number of retrieved documents by modifying the `k` parameter in the retriever
- Change the prompt template to customize how the LLM responds to questions

## Troubleshooting

- **Ollama Connection Issues**: Ensure Ollama is running on the default port (localhost:11434)
- **Missing Models**: Run `ollama list` to check installed models, and `ollama pull llama3.1` if needed
- **Memory Issues**: For large documents, you may need to increase your system's RAM or reduce the chunk size

## License

This project is licensed under the MIT License - see the LICENSE file for details.
