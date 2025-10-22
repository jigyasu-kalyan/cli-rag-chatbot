# Command-Line RAG Chatbot

A Python-based, command-line chatbot that uses a RAG (Retrieval-Augmented Generation) pipeline to answer questions about a custom set of documents.

This project was built in a one-week sprint to master the core components of a modern LLM application, as listed in [Job Description/Internship Goal].

## Core Technologies

* **Python 3.10+**
* **Google Gemini API** (for `gemini-pro` generation and `text-embedding-004` embeddings)
* **ChromaDB** (as a local, persistent vector store)
* **LangChain** (used *only* for its `RecursiveCharacterTextSplitter`)

---

## How It Works: The RAG Pipeline

This project is broken into two main parts:

**1. Ingestion (`load_db.py`):**
* Reads all `.txt` files from the `/data` directory.
* Uses `RecursiveCharacterTextSplitter` to "chunk" the text into 1000-character pieces.
* Calls the Gemini embedding API for *every single chunk* to create a vector.
* Stores these vectors and the corresponding text in a persistent `ChromaDB` collection.

**2. Retrieval & Generation (`chat.py`):**
* Waits for a user's question.
* Calls the Gemini embedding API to turn the *question* into a vector.
* Queries the `ChromaDB` to find the top 3 most "similar" text chunks.
* Injects those chunks (the "context") and the user's question into a prompt.
* Calls the `gemini-pro` generative model to get a final,-grounded answer based *only* on the provided context.

---

## How to Run It

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/jigyasu-kalyan/cli-rag-chatbot.git](https://github.com/jigyasu-kalyan/cli-rag-chatbot.git)
    cd cli-rag-chatbot
    ```

2.  **Set up the environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # (or .\venv\Scripts\activate on Windows)
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Add your documents:**
    Place your own `.txt` files inside the `/data` folder.

5.  **Set your API Key:**
    Open `load_db.py` and `chat.py` and paste your Google AI API key into the `API_KEY = "..."` variable.

6.  **Build the database (Run once):**
    ```bash
    python load_db.py
    ```

7.  **Run the chatbot:**
    ```bash
    python chat.py
    ```