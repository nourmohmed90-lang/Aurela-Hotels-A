from pathlib import Path

# ==========================================
# Project Paths
# ==========================================

RAG_DIR = Path(__file__).resolve().parent

DOCUMENTS_DIR = RAG_DIR / "documents"

VECTOR_DB_DIR = RAG_DIR / "vector_db"

VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# ChromaDB
# ==========================================

COLLECTION_NAME = "aurelia_hotels_knowledge"


# ==========================================
# Embedding Model
# ==========================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ==========================================
# Chunking
# ==========================================

CHUNK_SIZE = 500

CHUNK_OVERLAP = 100


# ==========================================
# Retrieval
# ==========================================

TOP_K = 3


# ==========================================
# Supported Documents
# ==========================================

SUPPORTED_EXTENSIONS = {
    ".md",
    ".txt",
    ".pdf"
}