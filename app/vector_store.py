import hashlib
import time

from pinecone import Pinecone, ServerlessSpec
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_DIMENSION,
    PINECONE_INDEX_NAME,
)
from app.ingestion import load_and_split_pdf


# Initialize local Ollama embeddings. Both indexing and search use this factory.
def get_embeddings():
    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


# Create or connect to the Pinecone index.
def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = pc.list_indexes().names()

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_INDEX_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        # Wait until the index is ready.
        while not pc.describe_index(
            PINECONE_INDEX_NAME
        ).status["ready"]:
            time.sleep(2)
    else:
        existing_dimension = pc.describe_index(PINECONE_INDEX_NAME).dimension
        if existing_dimension != PINECONE_INDEX_DIMENSION:
            raise ValueError(
                f"Pinecone index '{PINECONE_INDEX_NAME}' has dimension "
                f"{existing_dimension}; Ollama model '{OLLAMA_EMBEDDING_MODEL}' "
                f"requires {PINECONE_INDEX_DIMENSION}. Configure a new index "
                "name with PINECONE_OLLAMA_INDEX_NAME."
            )

    return pc.Index(PINECONE_INDEX_NAME)


# Connect LangChain to the Pinecone index.
def get_vector_store():
    index = get_pinecone_index()

    return PineconeVectorStore(
        index=index,
        embedding=get_embeddings(),
    )


# Ingest PDF chunks and store them in Pinecone.
def index_documents():
    chunks = load_and_split_pdf()

    vector_store = get_vector_store()

    ids = [
        hashlib.sha256(
            f"{doc.metadata['source']}:{doc.metadata['page']}:{i}:{doc.page_content}".encode(
                "utf-8"
            )
        ).hexdigest()
        for i, doc in enumerate(chunks)
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    return len(chunks)
