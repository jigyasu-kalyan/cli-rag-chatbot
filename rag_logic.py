import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

def get_query_embedding(text):
    try:
        result = genai.embed_content(
            model='models/text-embedding-004',
            content=text,
            task_type='RETRIEVAL_QUERY'
        )
        return result['embedding']
    except Exception as e:
        print(f"Error embedding query: {e}")
        return None

def find_relevant_chunks(question, collection):

    query_embedding = get_query_embedding(question)

    if query_embedding is None:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    return results['documents'][0]

def get_rag_answer(question, relevant_chunks):
    print("Generating answer...")

    prompt_template = f"""
    You are a helpful assistant. Answer the user's question based ONLY on the following context.
    If the context doesn't contain the answer, say "I'm sorry, I don't have information on that from the provided documents."
    
    CONTEXT:
    -----
    {relevant_chunks[0]}
    -----
    {relevant_chunks[1]}
    -----
    {relevant_chunks[2]}
    -----

    QUESTION:
    {question}

    ANSWER:
    """

    try:
        model = genai.GenerativeModel('models/gemini-2.5-pro')
        response = model.generate_content(prompt_template)
        return response.text
    except Exception as e:
        print(f"Sorry, I had trouble generating an answer. {e}")
