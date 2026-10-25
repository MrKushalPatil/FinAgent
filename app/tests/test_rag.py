"""Tests for semantic chunking with overlap and the retriever built on it."""
import unittest

from app.config import settings
from app.rag.chunker import ChunkConfig, semantic_chunks, split_sentences, window_texts
from app.rag.embedders import TfidfEmbedder, make_embedder
from app.rag.retriever import Retriever, _parse
from tests.rag_eval_set import EVAL


def doc_sentences():
    out = {}
    for f in sorted(settings.kb_dir.glob("*.md")):
        title, src, secs = _parse(f)
        sents, sections = [], []
        for sec, text in secs:
            for s in split_sentences(text):
                sents.append(s); sections.append(sec)
        out[f.stem] = (sents, sections)
    return out


class SentenceSplitTests(unittest.TestCase):
    def test_basic_and_abbreviations(self):
        s = split_sentences("PPF pays about 7.1 percent. It has a 15-year lock-in, e.g. partial withdrawals are limited. Check Section 24(b). Done!")
        self.assertEqual(len(s), 4)
        self.assertTrue(s[1].startswith("It has") and "e.g. partial" in s[1])

    def test_empty(self):
        self.assertEqual(split_sentences("   "), [])


class ChunkerGuaranteeTests(unittest.TestCase):
    def setUp(self):
        self.docs = doc_sentences()
        emb = TfidfEmbedder()
        emb.fit([w for s, sec in self.docs.values() for w in window_texts(s, sec, 1)])
        self.emb = emb

    def chunks(self, name, **kw):
        sents, secs = self.docs[name]
        return sents, semantic_chunks(sents, secs, self.emb, ChunkConfig(**kw))

    def test_no_sentence_lost_or_duplicated_in_core(self):
        for name in self.docs:
            sents, ch = self.chunks(name)
            self.assertEqual([s for c in ch for s in c.sentences], sents, name)

    def test_overlap_is_tail_of_previous_chunk(self):
        for ov in (1, 2):
            for name in self.docs:
                _, ch = self.chunks(name, overlap_sentences=ov)
                self.assertEqual(ch[0].overlap, [])
                for prev, cur in zip(ch, ch[1:]):
                    expect = prev.sentences[-ov:] if len(prev.sentences) >= ov else prev.sentences
                    self.assertEqual(cur.overlap[-len(expect):], expect, f"{name} ov={ov}")
                    self.assertLessEqual(len(cur.overlap), ov)

    def test_zero_overlap(self):
        _, ch = self.chunks("savings_schemes", overlap_sentences=0)
        self.assertTrue(all(c.overlap == [] for c in ch))

    def test_size_limits(self):
        for name in self.docs:
            _, ch = self.chunks(name, max_chars=900, min_sentences=2)
            for c in ch:
                self.assertLessEqual(len(c.core_text), 900 + 400, name)  # one-sentence slack at the ceiling
                self.assertGreaterEqual(len(c.sentences), 2, name)

    def test_smaller_max_gives_more_chunks(self):
        _, big = self.chunks("savings_schemes", max_chars=2000, percentile=99)
        _, small = self.chunks("savings_schemes", max_chars=300, percentile=99)
        self.assertGreater(len(small), len(big))

    def test_single_sentence_document(self):
        ch = semantic_chunks(["Only one sentence."], ["Sec"], self.emb, ChunkConfig())
        self.assertEqual(len(ch), 1)
        self.assertEqual(semantic_chunks([], [], self.emb, ChunkConfig()), [])

    def test_breakpoints_prefer_topic_changes(self):
        """Two clearly different topics glued together should be split between them."""
        a = ["PPF is a fifteen year scheme.", "PPF deposits earn tax free interest.", "PPF has a yearly deposit limit."]
        b = ["Cyber fraud must be reported quickly.", "Never share your OTP with callers.", "Call the fraud helpline immediately."]
        sents = a + b
        secs = ["x"] * 6
        emb = TfidfEmbedder(); emb.fit(window_texts(sents, secs, 1))
        ch = semantic_chunks(sents, secs, emb, ChunkConfig(percentile=60, min_sentences=2))
        self.assertEqual(ch[0].sentences, a)


class RetrieverTests(unittest.TestCase):
    def test_both_strategies_build(self):
        sem = Retriever(settings.kb_dir, "semantic"); head = Retriever(settings.kb_dir, "heading")
        self.assertGreater(len(sem.chunks), 10); self.assertGreater(len(head.chunks), 10)
        self.assertTrue(all(c.overlap_text == "" for c in head.chunks))
        self.assertTrue(any(c.overlap_text for c in sem.chunks))

    def test_stitching_merges_neighbours_without_duplicate_text(self):
        r = Retriever(settings.kb_dir, "semantic", ChunkConfig(max_chars=250, percentile=99, overlap_sentences=1))
        res = r.search("PPF lock-in period yearly deposit limit tax free interest", k=5)
        for x in res:
            sents = split_sentences(x["text"])
            self.assertEqual(len(sents), len(set(sents)), "a sentence was repeated after stitching")
        self.assertTrue(any(len(x["chunk_ids"]) > 1 for x in res))

    def test_quality_regression(self):
        """Semantic+overlap must stay at least as good as simple heading chunks on the evaluation set."""
        def score(r):
            top1 = comp = 0
            for q, need in EVAL:
                res = r.search(q, 3)
                top1 += bool(res) and all(p in res[0]["text"].lower() for p in need)
                comp += all(p in " ".join(x["text"] for x in res).lower() for p in need)
            return top1 / len(EVAL), comp / len(EVAL)
        sem = score(Retriever(settings.kb_dir, "semantic", ChunkConfig(heading_bonus=0.3)))
        self.assertGreaterEqual(sem[0], 0.90); self.assertGreaterEqual(sem[1], 0.95)

    def test_irrelevant_query_returns_nothing(self):
        self.assertEqual(Retriever(settings.kb_dir).search("bitcoin pizza"), [])

    def test_stats(self):
        st = Retriever(settings.kb_dir).stats()
        self.assertEqual(st["strategy"], "semantic"); self.assertEqual(st["documents"], 6)


class FakeDense:
    """Deterministic stand-in for a neural embedder, to test the hybrid (BM25 + dense) path without a model download."""
    name, dense = "fake", True
    def fit(self, texts): pass
    def embed(self, texts):
        keys = ["ppf", "tax", "fraud", "loan", "sip", "pension"]
        return [[float(t.lower().count(k)) for k in keys] for t in texts]
    @staticmethod
    def sim(a, b):
        na = sum(x * x for x in a) ** .5 or 1; nb = sum(x * x for x in b) ** .5 or 1
        return sum(x * y for x, y in zip(a, b)) / (na * nb)


class HybridAndFallbackTests(unittest.TestCase):
    def test_hybrid_retrieval_runs(self):
        r = Retriever(settings.kb_dir, "semantic", embedder=FakeDense())
        self.assertTrue(all(c.vector is not None for c in r.chunks))
        res = r.search("PPF lock-in period", k=3)
        self.assertTrue(res and "15-year" in " ".join(x["text"] for x in res))

    def test_sbert_missing_falls_back_to_tfidf(self):
        emb, note = make_embedder("sbert", "does-not-exist")
        self.assertEqual(emb.name, "tfidf"); self.assertIn("fell back", note)


if __name__ == "__main__":
    unittest.main()
