import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from config.settings import (
    CHROMA_HOST, CHROMA_PORT, COLLECTION_NAME,
    EMBEDDING_MODEL, BATCH_SIZE, MAX_CHUNKS
)


def get_embedding_function():
    """Get the embedding function"""
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)


def get_chroma_client():
    """Get Chroma HTTP client"""
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def create_vector_store(chunks):
    """Create and populate vector store"""
    print("\n" + "="*60)
    print("STEP 5: Creating vector store...")
    print("="*60)
    
    embedding_model = get_embedding_function()
    chroma_client = get_chroma_client()
    
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        client=chroma_client
    )
    
    # Add documents in batches
    total_chunks = min(len(chunks), MAX_CHUNKS)
    
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  📦 Adding batch {batch_num}/{total_batches}...")
        vector_store.add_documents(batch)
    
    print(f"✅ Vector store created and populated with {total_chunks} chunks!")
    return vector_store


def get_vector_store():
    """Connect to existing vector store"""
    embedding_model = get_embedding_function()
    chroma_client = get_chroma_client()
    
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        client=chroma_client
    )