import re
import json
import requests
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"

def clean_text(text):
    """
    Cleans the raw text by removing headers, footers, page numbers,
    and other noise.
    """
    text = re.sub(r"PANJAB UNIVERSITY, CHANDIGARH", "", text)
    text = re.sub(r"HANDBOOK OF INFORMATION \d{4}", "", text)
    text = re.sub(r"Page No. \d+", "", text)
    text = re.sub(r"^\d+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"Sr. No..*Page No.", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text.strip()

def get_embedding_from_ollama(text):
    """Gets an embedding for a single piece of text from the local Ollama server."""
    try:
        response = requests.post(
            OLLAMA_EMBED_URL,
            data=json.dumps({"model": OLLAMA_MODEL, "prompt": text}),
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"\nError getting embedding for text chunk: {e}")
        print(f"Text was: {text[:100]}...") # Print first 100 chars of failing text
        return None

def main():
    """
    The main function to run the data ingestion pipeline.
    """
    print("Starting data ingestion pipeline using local Ollama...")

    # 1. Load Data
    print("   - Loading raw data from src/pu_admissions_combined.txt")
    with open("src/pu_admissions_combined.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 2. Clean Data
    print("   - Cleaning raw text...")
    cleaned_text = clean_text(raw_text)

    # 3. Chunk Data
    print("   - Chunking cleaned text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )
    chunks = splitter.create_documents([cleaned_text])
    print(f"   - Created {len(chunks)} chunks.")

    chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_metadata = [chunk.metadata for chunk in chunks]

    # 4. Embed Chunks via Ollama
    print(f"   - Embedding chunks using '{OLLAMA_MODEL}' via Ollama...")
    embeddings = []
    for text in tqdm(chunk_texts, desc="Embedding Chunks"):
        embedding = get_embedding_from_ollama(text)
        if embedding:
            embeddings.append(embedding)
    
    if len(embeddings) != len(chunk_texts):
        print("\nWarning: Some chunks failed to embed. The process will stop.")
        print(f"Successfully embedded {len(embeddings)} out of {len(chunk_texts)} chunks.")
        return

    print(f"   - Generated {len(embeddings)} embeddings.")

    # 5. Store in ChromaDB
    print("   - Storing chunks and embeddings in ChromaDB...")
    client = chromadb.PersistentClient(path="src/data/chroma_db")
    collection = client.get_or_create_collection(name="pu_admissions_ollama")

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Batch the data to avoid exceeding ChromaDB's batch size limit
    batch_size = 5000
    for i in tqdm(range(0, len(ids), batch_size), desc="Adding to ChromaDB"):
        batch_ids = ids[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        batch_documents = chunk_texts[i:i + batch_size]
        batch_metadatas = chunk_metadata[i:i + batch_size]

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )

    print("\n✅ Ingestion complete!")
    print(f"Data is stored in the 'pu_admissions_ollama' collection in the 'src/data/chroma_db' directory.")

if __name__ == "__main__":
    main()
