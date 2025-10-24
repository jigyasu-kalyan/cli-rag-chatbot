# It is the workflow for automated evaluations. But it is currently non-functional because
# Ragas and LangChain dependency issues

import os
from datasets import Dataset
from ragas import evaluate
# from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.metrics import faithfulness, answer_relevancy
import chromadb
from dotenv import load_dotenv
from rag_logic import get_query_embedding, find_relevant_chunks, get_rag_answer
# from openai import OpenAI
import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("Error: Can't find Google API key in .env file.")
    exit(1)

try:
    google_client = genai.GenerativeModel(
        model_name="gemini-pro", # Still specify model here
        # Pass client options directly to the underlying client setup
        client_options={"api_endpoint": "generativelanguage.googleapis.com"}
    )._client # Access the underlying client object if needed by Langchain
    # Configure the library globally (might still be needed for other parts)
    genai.configure(api_key=API_KEY, client_options={"api_endpoint": "generativelanguage.googleapis.com"})
    print("Google AI client configured with V1 endpoint.")
except Exception as e:
    print(f"Error configuring Google AI client: {e}")
    exit(1)

V1_ENDPOINT = "generativelanguage.googleapis.com"
gemini_llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0, client_options={"api_endpoint": "generativelanguage.googleapis.com"})
# ragas_gemini_llm = LangchainLLMWrapper(llm=gemini_llm)


evaluation_questions = [
    "What is Dijsktra's algorithm?",
    # "How to implement dfs?",
    # "Which data structure we use in bfs?",
    # "Teach me Floyd-Warshall Algorithm.",
    # "Summarise Bellman-Ford Algorithm."
]

print("Generating answers and contexts for evaluation questions...")
questions = []
answers = []
contexts = []

try:
    client = chromadb.PersistentClient(path='./chroma_db')
    collection = client.get_collection(name='my_doc_collection')
    print("Successfully connected to ChromaDB.")
except Exception as e:
    print(f"Error occured while connecting to ChromaDB: {e}")
    exit(1)


for q in evaluation_questions:
    retrieved_chunks = find_relevant_chunks(q, collection)

    if not retrieved_chunks:
        print(f"No relevant context found for the question: {q}")
        continue
    
    genrated_answer = get_rag_answer(q, retrieved_chunks)
    if not genrated_answer or "Sorry, I had trouble generating an answer." in genrated_answer:
        print(f"Error occured while generating an answer for question: {q}")
        continue

    questions.append(q)
    answers.append(genrated_answer)
    contexts.append(retrieved_chunks)

if not questions:
    print("No valid question-answer pair found.")
    exit(1)

response_dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts
})

print(f"Dataset created with {len(response_dataset)} entries.")
print(f"Running Ragas evaluations for faithfulness and answer-relevancy...")

metrics_to_run = [
    faithfulness,
    answer_relevancy
]

try:
    results = evaluate(
        dataset=response_dataset,
        metrics=metrics_to_run,
        llm=gemini_llm
    )
    print("Evaluation complete.")
    print("\n-- Evaluation Scores --")
    print(results)
except Exception as e:
    print(f"Error occured while evaluating: {e}")
    exit(1)
    import traceback
    traceback.print_exc()