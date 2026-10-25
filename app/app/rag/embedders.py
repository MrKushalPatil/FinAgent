"""
Embedders turn text into vectors so we can measure how similar two pieces of text are.

  tfidf  (default)  pure Python, no download, works offline. Vectors are TF-IDF weighted word vectors
                    (stemmed, stop-words removed). It measures *lexical topic similarity*: good enough to find where
                    one topic ends and the next begins in a focused finance corpus, but it does NOT understand
                    synonyms ("car" vs "vehicle").
  sbert  (optional) neural sentence embeddings (sentence-transformers). Understands meaning and synonyms.
                    Enable with:  pip install sentence-transformers   and   RAG_EMBEDDER=sbert in .env

Both expose the same tiny interface, so the chunker and retriever do not care which one is used.
"""
from __future__ import annotations

import math
import re
from collections import Counter

STOP = set("a an the of for to in on and or is are was be with as at by from that this it what how do does i my me can "
           "should which when who why about there their if".split())


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class TfidfEmbedder:
    name = "tfidf"
    dense = False

    def __init__(self):
        self.idf: dict[str, float] = {}

    def fit(self, texts: list[str]) -> None:
        df: Counter = Counter()
        for t in texts:
            df.update(set(tokenize(t)))
        n = max(len(texts), 1)
        self.idf = {w: math.log(1 + n / c) for w, c in df.items()}

    def embed(self, texts: list[str]) -> list[dict]:
        out = []
        for t in texts:
            tf = Counter(tokenize(t))
            vec = {w: (1 + math.log(c)) * self.idf.get(w, math.log(1 + 1)) for w, c in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            out.append({w: v / norm for w, v in vec.items()})
        return out

    @staticmethod
    def sim(a: dict, b: dict) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(w, 0.0) for w, v in a.items())


class SentenceTransformerEmbedder:
    name = "sbert"
    dense = True

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.model_name = model
        self._m = SentenceTransformer(model)

    def fit(self, texts: list[str]) -> None:  # pretrained: nothing to fit
        return None

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self._m.encode(texts, normalize_embeddings=True)]

    @staticmethod
    def sim(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


def make_embedder(kind: str, model: str):
    """Returns (embedder, note). Falls back to TF-IDF with an explanatory note if sbert is unavailable."""
    if kind == "sbert":
        try:
            return SentenceTransformerEmbedder(model), f"neural embeddings ({model})"
        except Exception as e:  # not installed / offline / model missing
            return TfidfEmbedder(), f"sbert unavailable ({type(e).__name__}); fell back to TF-IDF"
    return TfidfEmbedder(), "TF-IDF lexical vectors (offline)"
