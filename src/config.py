"""Central config for model names, Ollama host, and paths."""
from pathlib import Path

OLLAMA_HOST = "http://localhost:11434"
PRIMARY_MODEL = "qwen2.5:7b-instruct"
CODER_MODEL = "qwen2.5-coder:7b-instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
EVAL_DIR = DATA_DIR / "eval"
CHROMA_DIR = DATA_DIR / "chroma"
LOGS_DIR = ROOT_DIR / "logs"
