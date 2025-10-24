import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from rag_logic import get_query_embedding, find_relevant_chunks_with_rerank, get_rag_answer

if __name__ == "__main__":

    try:
        client = chromadb.PersistentClient(path='./chroma_db')
        collection = client.get_collection(name='my_doc_collection')
        print("Connected to ChromaDB collection.")
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        sys.exit(1)

    print("--- Document chatbot ---")
    print("Ask about questions about your documents. Type 'q' to quit.")

    while True:
        my_question = input("Ask a question: ")

        if my_question.lower() == 'q':
            print("Goodbye!")
            break

        print("Thinking...")

        chunks = find_relevant_chunks_with_rerank(my_question, collection)

        if not chunks:
            print("No relevant information found.")
            continue
        
        answer = get_rag_answer(my_question, chunks)

        print("\n--- Answer ---")
        print(answer)