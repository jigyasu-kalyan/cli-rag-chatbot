import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
import chromadb

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

print("Connecting to Google AI...")

DATA_DIR = "data"

def load_and_chunk_docs(directory=DATA_DIR):

    print(f"Loading documents from {directory}")

    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    documents.append(f.read())
                print(f" - Loaded {filename}")
            except Exception as e:
                print(f" - Error loading {filename}: {e}")
    
    print(f"Loaded {len(documents)} documents total.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunked_docs = text_splitter.create_documents(documents)
    print(f"Total chunks created: {len(chunked_docs)}")
    return chunked_docs

def get_embedding(chunk_text):

    try:
        result = genai.embed_content(
            model='models/text-embedding-004',
            content=chunk_text,
            task_type='RETRIEVAL_DOCUMENT'
        )
        return result['embedding']
    except Exception as e:
        print(f"Error embedding text: {e}")
        return None

def build_and_save_db(chunks):

    print("Building vector database...")
    client = chromadb.PersistentClient(path='./chroma_db')
    collection = client.get_or_create_collection(name='my_doc_collection')

    print("Embedding chunks and adding to database. This may take a while...")

    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        ids = []
        embeddings = []
        documents = []

        for j, chunk in enumerate(batch):
            chunk_text = chunk.page_content
            chunk_embedding = get_embedding(chunk_text)

            if chunk_embedding:
                ids.append(f"chunk_{i+j}")
                embeddings.append(chunk_embedding)
                documents.append(chunk_text)
        
        if ids:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents
            )
        
        print(f" - Added batch: {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
    print("-- Database build complete! --")
    print(f"Total documents in collection: {collection.count()}")

if __name__ == "__main__":
    all_chunks = load_and_chunk_docs()

    if all_chunks:
        build_and_save_db(all_chunks)
    else:
        print("No chunks were created. Check your 'data' folder.")