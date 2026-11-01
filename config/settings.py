"""Environment-overridable paths and model settings."""
import os
from pathlib import Path
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env.local", override=False)
DATA_RAW_DIR = Path(os.getenv("RAG_UPLOAD_DIR", str(BASE_DIR / "data" / "uploads")))
VECTORSTORE_DIR = Path(os.getenv("RAG_VECTORSTORE_DIR", str(BASE_DIR / "data" / "vectorstore")))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_K = 4
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
