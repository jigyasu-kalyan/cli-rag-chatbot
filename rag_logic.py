import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from sentence_transformers import CrossEncoder

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


reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
print("Reranker model loaded.")

def find_relevant_chunks_with_rerank(question, collection, top_k_retrieval=10, top_k_rerank=3):

    query_embedding = get_query_embedding(question)

    if query_embedding is None:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    # return results['documents'][0]
    initial_chunks = results['documents'][0]

    if not initial_chunks:
        return []

    rerank_pairs = [[question, chunk] for chunk in initial_chunks]
    print(f"Reranking {len(rerank_pairs)} chunks...")
    scores = reranker_model.predict(rerank_pairs)
    scored_chunks = list(zip(scores, initial_chunks))
    scored_chunks.sort(reverse=True)

    reranked_chunks = [chunk for score, chunk in scored_chunks]
    print("Reranking complete.")
    return reranked_chunks


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
