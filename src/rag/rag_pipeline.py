import ollama
import chromadb
from fastapi.concurrency import run_in_threadpool

# --- Configuration ---
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "llama3:8b"
CHROMA_DB_PATH = "src/data/chroma_db"
CHROMA_COLLECTION_NAME = "pu_admissions_ollama"
TOP_K = 5

# --- ChromaDB Client ---
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

async def get_embedding(text: str):
    """Gets an embedding for a single piece of text from the local Ollama server."""
    try:
        response = await ollama.AsyncClient().embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"\nError getting embedding for text chunk: {e}")
        print(f"Text was: {text[:100]}...")
        return None

async def retrieve(query: str, k: int = TOP_K):
    """
    Retrieves the top k most relevant chunks from ChromaDB based on the user's query.
    """
    print(f"   - Retrieving top {k} chunks for query: '{query}'")
    query_embedding = await get_embedding(query)

    if query_embedding:
        results = await run_in_threadpool(
            collection.query,
            query_embeddings=[query_embedding],
            n_results=k
        )
        return results['documents'][0]
    return []

async def generate(query: str, context: list[str]):
    """
    Generates an answer to the user's query based on the retrieved context.
    """
    print("   - Generating answer...")
    
    context_str = "\n---\n".join(context)
    prompt = f"""
You are an expert assistant for Panjab University admissions. Your task is to answer the user's question based *only* on the provided context. If the context does not contain the answer, say "I do not have enough information to answer that question."

Context:
---
{context_str}
---

Question: {query}

Answer:
"""

    response = await ollama.AsyncClient().generate(model=OLLAMA_LLM_MODEL, prompt=prompt)
    return response['response']

# The test block below is for synchronous testing and will not work with the async functions.
# To test the async pipeline, we would need to use an async framework like asyncio.
if __name__ == "__main__":
    print("This script now contains async functions and cannot be run directly.")
    print("Please run the API with 'uvicorn src.api.main:app --reload' to test the pipeline.")