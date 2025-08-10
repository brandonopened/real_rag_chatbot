  python -c "
  import os
  import json
  import requests
  from langchain_huggingface import HuggingFaceEmbeddings
  from langchain_chroma import Chroma

  # Configuration
  PERSIST_DIRECTORY = './data/chroma_db'
  PROFILES_FILE = 'profiles.json'
  GROQ_API_KEY = os.getenv('GROQ_API_KEY')

  def query_rag_chatbot(query, profile_name='Workshop Facilitator'):
      # Load profiles
      with open(PROFILES_FILE, 'r') as f:
          data = json.load(f)
      
      # Get profile
      profile = next((p for p in data['profiles'] if p['name'] == profile_name), data['profiles'][0])
      
      # Initialize vector store
      embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
      vector_store = Chroma(
          persist_directory=PERSIST_DIRECTORY,
          embedding_function=embeddings,
          collection_name=profile['collection_name']
      )
      
      # Retrieve relevant documents
      retriever = vector_store.as_retriever(search_kwargs={'k': 3})
      docs = retriever.get_relevant_documents(query)
      context = '\n\n'.join([doc.page_content for doc in docs])
      
      # Prepare Groq API request
      system_prompt = profile['system_prompt']
      if context:
          system_prompt += f'\n\nContext:\n{context}'
      
      messages = [
          {'role': 'system', 'content': system_prompt},
          {'role': 'user', 'content': query}
      ]
      
      # Make Groq API call
      response = requests.post(
          'https://api.groq.com/openai/v1/chat/completions',
          headers={'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'},
          json={
              'model': 'llama-3.3-70b-versatile',
              'messages': messages,
              'max_tokens': 4000,
              'temperature': 0.7
          }
      )
      
      return response.json()['choices'][0]['message']['content']

  # Usage
  result = query_rag_chatbot('YOUR_QUESTION_HERE', 'Workshop Facilitator')
  print(result)
  "