"""RAG package. Use get_retriever() to obtain the shared, lazily-built retriever."""
from functools import lru_cache

from ..config import settings
from .chunker import ChunkConfig
from .embedders import make_embedder
from .retriever import Retriever


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    cfg = ChunkConfig(window=settings.rag_window, percentile=settings.rag_percentile,
                      min_sentences=settings.rag_min_sentences, max_chars=settings.rag_max_chars,
                      overlap_sentences=settings.rag_overlap, heading_bonus=settings.rag_heading_bonus)
    embedder, note = make_embedder(settings.rag_embedder, settings.rag_embed_model)
    return Retriever(settings.kb_dir, strategy=settings.rag_chunking, cfg=cfg, embedder=embedder, embed_note=note)
