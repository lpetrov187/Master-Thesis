"""Central config for model names, Ollama host, and paths."""
import os
from pathlib import Path

OLLAMA_HOST = "http://localhost:11434"
PRIMARY_MODEL = "qwen2.5:7b-instruct"
CODER_MODEL = "qwen2.5-coder:7b-instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# sentence-transformers checks the HF Hub on every load, even when the
# model is already cached, which prints a noisy "unauthenticated requests"
# warning from a native extension (hf_xet) - it bypasses Python's print(),
# so it can't be silenced with warnings/logging config. If the model is
# already cached locally, skip that network check entirely instead. Only
# a genuinely fresh machine (no local cache yet) falls through to the
# normal - slower, warning-printing - first-time download.
#
# This must be set *before* anything imports huggingface_hub - its
# HF_HUB_OFFLINE flag is frozen as a module constant at import time, so
# setting the env var any later has no effect on it.
_hf_hub_cache = os.environ.get(
    "HF_HUB_CACHE", os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")) + "/hub"
)
_embedding_model_cached = (Path(_hf_hub_cache) / ("models--" + EMBEDDING_MODEL.replace("/", "--"))).is_dir()
if _embedding_model_cached:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Day 7's Claim Verifier is the pipeline's biggest latency cost by far (see
# PLAN.md's timing analysis - ~2x an LLM call per verify(), often doubled
# again by the retry it can trigger). Disabled for now to speed up demo/dev
# iteration; the module and its tests are untouched - flip back to True to
# re-enable.
ENABLE_CLAIM_VERIFIER = False

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
EVAL_DIR = DATA_DIR / "eval"
CHROMA_DIR = DATA_DIR / "chroma"
LOGS_DIR = ROOT_DIR / "logs"
