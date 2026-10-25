"""
RAG retriever: knowledge-base markdown -> semantic chunks with overlap -> BM25 (+ optional dense) ranking.

Pipeline
  1. LOAD      read app/rag/knowledge_base/*.md (front-matter: title, source URL; '## ' headings = sections)
  2. CHUNK     strategy "semantic" (default): sentence splitting + embedding-similarity breakpoints + overlap
               strategy "heading"           : one chunk per '## ' section (the original baseline, kept for comparison)
  3. INDEX     BM25 statistics over each chunk (title and section words are weighted double)
  4. RETRIEVE  BM25 ranking; if a neural embedder is enabled, fuse it with dense similarity (Reciprocal Rank Fusion)
  5. STITCH    neighbouring hits from the same document are merged so the overlap is not repeated to the LLM
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import ChunkConfig, semantic_chunks, split_sentences, window_texts
from .embedders import TfidfEmbedder, tokenize


@dataclass
class Chunk:
    id: str
    doc_title: str
    doc: str
    index: int
    sections: list[str]
    core_text: str
    overlap_text: str
    source_url: str
    distance_before: float = 0.0
    tokens: list[str] = field(default_factory=list, repr=False)
    vector: object = field(default=None, repr=False)

    @property
    def section(self) -> str:
        return " / ".join(self.sections)

    @property
    def text(self) -> str:
        return (self.overlap_text + " " + self.core_text).strip()


def _parse(path: Path):
    raw = path.read_text(encoding="utf-8")
    meta, body = {}, raw
    if raw.startswith("---"):
        _, fm, body = raw.split("---", 2)
        for line in fm.strip().splitlines():
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    title = meta.get("title", path.stem.replace("_", " ").title())
    sections = []
    for block in re.split(r"\n(?=## )", body.strip()):
        head, _, text = block.strip().partition("\n")
        if text.strip():
            sections.append((head.lstrip("# ").strip(), text.strip()))
    return title, meta.get("source", ""), sections


class Retriever:
    def __init__(self, kb_dir: Path, strategy: str = "semantic", cfg: ChunkConfig | None = None,
                 embedder=None, embed_note: str = "", k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.strategy = strategy
        self.cfg = cfg or ChunkConfig()
        self.embedder = embedder or TfidfEmbedder()
        self.embed_note = embed_note or "TF-IDF lexical vectors (offline)"
        self.chunks: list[Chunk] = []
        self._build(Path(kb_dir))
        self.df: Counter = Counter()
        for c in self.chunks:
            self.df.update(set(c.tokens))
        self.avg_len = (sum(len(c.tokens) for c in self.chunks) / len(self.chunks)) if self.chunks else 1.0
        if getattr(self.embedder, "dense", False) and self.chunks:
            vecs = self.embedder.embed([f"{c.doc_title}. {c.section}. {c.text}" for c in self.chunks])
            for c, v in zip(self.chunks, vecs):
                c.vector = v

    # ------------------------------------------------------------------ build
    def _build(self, kb_dir: Path) -> None:
        docs = [(f, *_parse(f)) for f in sorted(kb_dir.glob("*.md"))]
        if self.strategy == "heading":
            for f, title, src, sections in docs:
                for i, (sec, text) in enumerate(sections):
                    self._add(f, title, src, i, [sec], text, "", 0.0)
            return

        per_doc = []
        for f, title, src, sections in docs:
            sents, secs = [], []
            for sec, text in sections:
                for s in split_sentences(text):
                    sents.append(s)
                    secs.append(sec)
            per_doc.append((f, title, src, sents, secs, window_texts(sents, secs, self.cfg.window)))
        self.embedder.fit([w for *_, ws in per_doc for w in ws])           # corpus-wide IDF for TF-IDF
        for f, title, src, sents, secs, ws in per_doc:
            vectors = self.embedder.embed(ws)
            for i, rc in enumerate(semantic_chunks(sents, secs, self.embedder, self.cfg, vectors)):
                self._add(f, title, src, i, rc.sections, rc.core_text, " ".join(rc.overlap), rc.distance_before)

    def _add(self, f: Path, title: str, src: str, idx: int, sections: list[str], core: str, overlap: str, dist: float):
        ch = Chunk(id=f"{f.stem}#{idx}", doc_title=title, doc=f.stem, index=idx, sections=sections,
                   core_text=core, overlap_text=overlap, source_url=src, distance_before=dist)
        head = f"{title} {' '.join(sections)}"
        ch.tokens = tokenize(f"{head} {head} {ch.text}")      # title + section words count double
        self.chunks.append(ch)

    # ------------------------------------------------------------------ search
    def _bm25(self, q: list[str]) -> list[tuple[float, Chunk]]:
        n = len(self.chunks)
        scored = []
        for c in self.chunks:
            tf = Counter(c.tokens)
            score = 0.0
            for term in set(q):
                if term not in tf:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                f = tf[term]
                score += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(c.tokens) / self.avg_len))
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        return scored

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = tokenize(query)
        if not q or not self.chunks:
            return []
        ranked = self._bm25(q)
        if not ranked:
            return []
        scores = {c.id: s for s, c in ranked}

        if getattr(self.embedder, "dense", False):                      # hybrid: BM25 + dense via RRF
            qv = self.embedder.embed([query])[0]
            dense = sorted(((self.embedder.sim(qv, c.vector), c) for c in self.chunks), key=lambda x: -x[0])
            fused: dict[str, float] = {}
            for rank, (_, c) in enumerate(ranked):
                fused[c.id] = fused.get(c.id, 0) + 1 / (60 + rank + 1)
            for rank, (_, c) in enumerate(dense[:20]):
                fused[c.id] = fused.get(c.id, 0) + 1 / (60 + rank + 1)
            by_id = {c.id: c for c in self.chunks}
            top = sorted(fused, key=lambda i: -fused[i])[:k]
            hits = [(scores.get(i, 0.0), by_id[i]) for i in top]
        else:
            hits = ranked[:k]
        return self._stitch(hits)

    @staticmethod
    def _stitch(hits: list[tuple[float, Chunk]]) -> list[dict]:
        """Merge hits that are neighbours in the same document so overlapped text is shown only once."""
        hits = sorted(hits, key=lambda h: (h[1].doc, h[1].index))
        groups: list[list[tuple[float, Chunk]]] = []
        for h in hits:
            if groups and groups[-1][-1][1].doc == h[1].doc and groups[-1][-1][1].index + 1 == h[1].index:
                groups[-1].append(h)
            else:
                groups.append([h])
        out = []
        for g in groups:
            first = g[0][1]
            text = first.text + "".join(" " + c.core_text for _, c in g[1:])
            sections = []
            for _, c in g:
                sections += [s for s in c.sections if s not in sections]
            out.append({"score": round(max(s for s, _ in g), 2), "title": first.doc_title,
                        "section": " / ".join(sections), "text": text, "source_url": first.source_url,
                        "chunk_ids": [c.id for _, c in g]})
        out.sort(key=lambda r: -r["score"])
        return out

    # ------------------------------------------------------------------ introspection
    def stats(self) -> dict:
        lens = [len(c.text) for c in self.chunks]
        return {"strategy": self.strategy, "embedder": self.embedder.name, "embedder_note": self.embed_note,
                "documents": len({c.doc for c in self.chunks}), "chunks": len(self.chunks),
                "avg_chunk_chars": round(sum(lens) / len(lens)) if lens else 0,
                "min_chunk_chars": min(lens, default=0), "max_chunk_chars": max(lens, default=0),
                "config": {"window": self.cfg.window, "percentile": self.cfg.percentile,
                           "min_sentences": self.cfg.min_sentences, "max_chars": self.cfg.max_chars,
                           "overlap_sentences": self.cfg.overlap_sentences}}
