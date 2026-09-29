import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Keep the Ollama vectors in a separate index. In particular, do not reuse the
# old PINECONE_INDEX_NAME setting, which may point to the OpenAI-backed index.
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_OLLAMA_INDEX_NAME", "agentic-ai-rag-ollama"
)
PINECONE_INDEX_DIMENSION = 768

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "4"))


def validate_config():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing in .env")

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"
        )
