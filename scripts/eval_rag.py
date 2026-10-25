"""
Compare chunking strategies on a 35-question retrieval set.  Run:  python scripts/eval_rag.py

top1%        : the best passage alone contains every required fact
complete@3%  : the top-3 passages together contain every required fact
avg chars    : context sent to the LLM per question (smaller = cheaper)
boundary F1  : how well the semantic breakpoints line up with the documents' own headings
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings                      # noqa: E402
from app.rag.chunker import ChunkConfig, split_sentences   # noqa: E402
from app.rag.retriever import Retriever, _parse      # noqa: E402
from tests.rag_eval_set import EVAL                  # noqa: E402


def evaluate(r, k=3):
    h1 = c3 = chars = 0
    for q, need in EVAL:
        res = r.search(q, k)
        top1 = res[0]["text"].lower() if res else ""
        union = " ".join(x["text"] for x in res).lower()
        h1 += all(p in top1 for p in need)
        c3 += all(p in union for p in need)
        chars += sum(len(x["text"]) for x in res)
    n = len(EVAL)
    return f"{h1 / n * 100:5.1f}   {c3 / n * 100:5.1f}   {chars / n:6.0f}"


def boundary_f1(r):
    tp = fp = fn = 0
    for f in sorted(Path(settings.kb_dir).glob("*.md")):
        _, _, secs = _parse(f)
        n, truth = 0, set()
        for _, text in secs:
            if n:
                truth.add(n)
            n += len(split_sentences(text))
        pos, pred = 0, set()
        for c in [c for c in r.chunks if c.doc == f.stem][:-1]:
            pos += len(split_sentences(c.core_text)); pred.add(pos)
        tp += len(pred & truth); fp += len(pred - truth); fn += len(truth - pred)
    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    return f"{2 * P * R / (P + R):.2f}" if P + R else "0.00"


rows = [("heading chunks (baseline, no overlap)", Retriever(settings.kb_dir, "heading")),
        ("semantic, overlap=0, no heading hint", Retriever(settings.kb_dir, "semantic", ChunkConfig(overlap_sentences=0, heading_bonus=0))),
        ("semantic, overlap=1, no heading hint", Retriever(settings.kb_dir, "semantic", ChunkConfig(heading_bonus=0))),
        ("semantic, overlap=1, heading hint (DEFAULT)", Retriever(settings.kb_dir, "semantic", ChunkConfig())),
        ("semantic, overlap=2, heading hint", Retriever(settings.kb_dir, "semantic", ChunkConfig(overlap_sentences=2)))]
print(f"{'strategy':46} chunks  top1%  compl@3%  avg chars  boundary F1")
for name, r in rows:
    print(f"{name:46} {len(r.chunks):5}   {evaluate(r)}      {boundary_f1(r) if r.strategy == 'semantic' else '  -'}")
