"""
Semantic chunking with overlap.

Instead of cutting text at fixed sizes or only at headings, we cut where the *meaning changes*:

  1. Split each document into sentences.
  2. Describe every sentence by a small window of its neighbours (smoothing) and embed it.
     The section heading is prepended as context so a sentence like "The rate is 8.2 percent" still knows which
     scheme it belongs to.
  3. Measure the distance between each window and the next one (1 - cosine similarity).
     A big jump means the topic changed -> that is a candidate breakpoint.
  4. Keep breakpoints whose distance is above a percentile threshold, then enforce a minimum and maximum chunk size.
  5. OVERLAP: each chunk is prefixed with the last N sentences of the previous chunk, so a fact that sits on a
     boundary (an answer split across two chunks) is still fully present in at least one chunk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ChunkConfig:
    window: int = 1               # neighbours on each side used to smooth a sentence's embedding
    percentile: float = 70.0      # a distance above this percentile of all distances becomes a breakpoint
    min_sentences: int = 2        # never create a chunk shorter than this (unless the doc is that short)
    max_chars: int = 900          # hard ceiling for a chunk's core text
    overlap_sentences: int = 1    # sentences copied from the end of the previous chunk
    heading_bonus: float = 0.3    # extra distance added where the document's own heading changes (structure hint)


@dataclass
class RawChunk:
    sentences: list[str]          # core sentences owned by this chunk
    overlap: list[str]            # sentences borrowed from the previous chunk
    sections: list[str]           # headings covered by the core sentences
    distance_before: float        # semantic distance at the boundary that started this chunk (0 for the first)

    @property
    def core_text(self) -> str:
        return " ".join(self.sentences)

    @property
    def text(self) -> str:
        return " ".join(self.overlap + self.sentences)


_ABBR = re.compile(r"(?:\be\.g|\bi\.e|\bvs|\bapprox|\bRs|\bNo|\bSec|\betc)\.$", re.I)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\u20b9\"(])", text)
    out: list[str] = []
    for p in parts:
        if out and _ABBR.search(out[-1]):
            out[-1] += " " + p
        else:
            out.append(p)
    return out


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def window_texts(sentences: list[str], sections: list[str], w: int) -> list[str]:
    """Context-enriched text for each sentence: heading + the sentence and its neighbours."""
    out = []
    for i in range(len(sentences)):
        lo, hi = max(0, i - w), min(len(sentences), i + w + 1)
        out.append(f"{sections[i]}. " + " ".join(sentences[lo:hi]))
    return out


def semantic_chunks(sentences: list[str], sections: list[str], embedder, cfg: ChunkConfig,
                    vectors: list | None = None) -> list[RawChunk]:
    """Chunk one document. `vectors` can be passed in when the caller already embedded the windows."""
    n = len(sentences)
    if n == 0:
        return []
    if vectors is None:
        vectors = embedder.embed(window_texts(sentences, sections, cfg.window))
    dist = [1.0 - embedder.sim(vectors[i], vectors[i + 1]) for i in range(n - 1)]  # dist[i]: boundary after sentence i
    if cfg.heading_bonus:
        dist = [d + (cfg.heading_bonus if sections[i] != sections[i + 1] else 0.0) for i, d in enumerate(dist)]
    threshold = _percentile(dist, cfg.percentile)
    is_break = [d >= threshold and d > 0.0 for d in dist]

    spans: list[tuple[int, int]] = []
    start = 0
    for i in range(n):
        size = sum(len(s) + 1 for s in sentences[start:i + 1])
        count = i - start + 1
        last = i == n - 1
        cut = False
        if not last:
            if is_break[i] and count >= cfg.min_sentences:
                cut = True
            elif size + len(sentences[i + 1]) > cfg.max_chars and count >= cfg.min_sentences:
                cut = True  # too big: split here (a later pass could pick the best valley; this is the safe fallback)
        if cut or last:
            spans.append((start, i + 1))
            start = i + 1
    # merge a too-small tail into the previous chunk
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < cfg.min_sentences:
        a, _ = spans[-2]
        spans[-2:] = [(a, spans[-1][1])]

    chunks: list[RawChunk] = []
    for k, (a, b) in enumerate(spans):
        ov_start = max(0, a - cfg.overlap_sentences) if k > 0 else a
        secs: list[str] = []
        for s in sections[a:b]:
            if s not in secs:
                secs.append(s)
        chunks.append(RawChunk(sentences=sentences[a:b], overlap=sentences[ov_start:a], sections=secs,
                               distance_before=round(dist[a - 1], 3) if a > 0 else 0.0))
    return chunks
