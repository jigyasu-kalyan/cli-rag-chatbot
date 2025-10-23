import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_logic import find_relevant_chunks, get_rag_answer

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("Google API Key not found in env file.")
    exit(1)
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"Error occured while configuring Google AI: {e}")
    exit(1)

app = FastAPI(
    title="RAG Chatbot API",
    description="An API that uses Retrieval-Augmented generation to answer questions based on documents.",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

try:
    client = chromadb.PersistentClient(path='./chroma_db')
    collection = client.get_collection(name='my_doc_collection')
    print("ChromaDB connection loaded successfully.")
except Exception as e:
    print("Error occured while loading ChromaDB: {e}")
    collection = None

@app.post('/ask',
          response_model=AnswerResponse,
          summary="Asks question to RAG model",
          description="Asks question and returns answer generated based on relevant documents.")
async def ask_question(request: QuestionRequest):

    if collection is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    question = request.question
    print(f"Recieved question: {question}")

    try:
        chunks = find_relevant_chunks(question, collection)
        if not chunks:
            print("No relevant context found.")
            raise HTTPException(status_code=404, detail="No relevant context found in yoour documents for the recieved question.")
    except Exception as e:
        print(f"Error during retrieval: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving relevant documents.")
    
    try:
        answer_text = get_rag_answer(question, chunks)
        print(f"Generated answer: {answer_text[:100]}")
    except Exception as e:
        print("Error generating an answer: {e}")
        raise HTTPException(status_code=500, detail="Error generating an answer.")

    return AnswerResponse(answer=answer_text)


@app.get('/',
         summary="Root endpoint",
         description="Returns a simple hello message.")
async def root():
    return {"message": "Welcome to the RAG API!"}