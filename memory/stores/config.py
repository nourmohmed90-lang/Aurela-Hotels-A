import os
from pathlib import Path

# Base Paths relative to root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", ROOT_DIR / "documents"))
VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", ROOT_DIR / "vector_db"))

VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# ChromaDB Configuration
COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "knowledge_base")

# Embedding Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Chunking Options
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retrieval Options
TOP_K = int(os.getenv("TOP_K", "3"))

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}