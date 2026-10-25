"""Central settings, loaded from environment variables and an optional .env file."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


class Settings:
    provider = os.environ.get("LLM_PROVIDER", "openai_compatible").strip().lower()
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    base_url = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile").strip()
    max_steps = int(os.environ.get("MAX_AGENT_STEPS", "6"))
    max_total_tool_calls = int(os.environ.get("MAX_TOTAL_TOOL_CALLS", "8"))
    enable_web = _bool("ENABLE_WEB_SEARCH", True)
    port = int(os.environ.get("PORT", "8000"))
    data_dir = ROOT / "data"
    kb_dir = ROOT / "app" / "rag" / "knowledge_base"
    max_upload_bytes = 10 * 1024 * 1024
    # --- RAG: semantic chunking with overlap ---
    rag_chunking = os.environ.get("RAG_CHUNKING", "semantic").strip().lower()      # semantic | heading
    rag_embedder = os.environ.get("RAG_EMBEDDER", "tfidf").strip().lower()         # tfidf | sbert
    rag_embed_model = os.environ.get("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
    rag_percentile = float(os.environ.get("RAG_BREAKPOINT_PERCENTILE", "70"))
    rag_window = int(os.environ.get("RAG_WINDOW", "1"))
    rag_min_sentences = int(os.environ.get("RAG_MIN_SENTENCES", "2"))
    rag_max_chars = int(os.environ.get("RAG_MAX_CHARS", "900"))
    rag_overlap = int(os.environ.get("RAG_OVERLAP_SENTENCES", "1"))
    rag_heading_bonus = float(os.environ.get("RAG_HEADING_BONUS", "0.3"))
    max_message_chars = 2000

    @property
    def llm_ready(self) -> bool:
        if self.provider == "mock":
            return True
        return self.api_key not in ("", "paste_your_key_here")


settings = Settings()
