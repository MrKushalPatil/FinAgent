# FinAgent — An LLM Agent with Tool Use (Topic 184)

A financial assistant where an LLM **decides when to call tools**, chains them across multiple steps, and is
kept safe by a guardrail layer. Built to match the FINAGENT jury presentation.

> "A chatbot generates an answer. An agent decides and acts using external tools before generating the answer."

## What it does

| Tool (what the model can request) | Input | What it returns |
|---|---|---|
| `analyze_bank_statement` | `file_id` of an uploaded PDF/CSV | Avg income, expenses, surplus, savings rate, existing EMIs, category split, largest expenses |
| `calculate_emi` | principal, rate %, tenure (+ optional income & existing EMIs) | Exact EMI, total interest, yearly amortisation, affordability (FOIR) verdict |
| `search_financial_info` | query | **RAG** passages (semantic chunks with overlap, BM25-ranked) with source URLs; trusted-domain web search for time-sensitive questions |
| `calculator` | arithmetic expression | Safe AST-evaluated result (percentages, SIP maths…) |

Example multi-step task: *"Based on my statement, can I afford a ₹40 lakh loan at 9% for 20 years?"*
→ model calls `analyze_bank_statement` → feeds income + existing EMIs into `calculate_emi` → explains the verdict.

## Quick start (5 minutes)

```bash
pip install -r requirements.txt            # pdfplumber (+ ddgs for optional live web search)
cp .env.example .env                       # then edit .env and paste your LLM key
python run.py                              # open http://localhost:8000
```

**Getting a free LLM key:** the default is Groq (https://console.groq.com/keys, free tier, supports function calling).
Any OpenAI-compatible endpoint works — OpenAI, Gemini, or **Ollama for fully offline** use. See `.env.example`.
Model names change often; if a model is rejected, pick a current tool-calling model from your provider's list.

No key yet? `LLM_PROVIDER=mock python run.py` runs a **rule-based stand-in (not an LLM)** so you can click through the UI.
Never present mock mode as the real agent.

Other commands:
```bash
python cli.py --file samples/sample_statement.pdf   # terminal agent that prints every tool call
python scripts/demo_failure.py                      # Step 5: misuse failures vs guardrails (offline)
python -m unittest discover -s tests -v             # 46 tests
python scripts/make_sample_statement.py             # regenerate the simple synthetic statement
python scripts/make_demo_statement.py               # regenerate the richer demo statement (samples/demo_statement.pdf)
python scripts/eval_rag.py                          # compare RAG chunking strategies
```

## Mapping to the brief's 6 steps

1. **Frame it** — `app/agent.py` (`SYSTEM_PROMPT`): an assistant that decides when to call tools.
2. **Define tools with clear schemas** — `app/tools/*.py` (`NAME`, `DESCRIPTION`, `SCHEMA`); see them live via the UI button *"Tool definitions"*.
3. **The loop** — `FinAgent.run()` in `app/agent.py`: model → tool_calls → guarded execution → result back to model → repeat → final answer.
4. **Test on calculation / lookup tasks** — `tests/` (46 tests) + the "Try asking" prompts in the UI.
5. **Show a failure and constrain it** — `app/demo_cases.py`, `scripts/demo_failure.py`, UI button *"Safeguard demonstration"*.
6. **Present the working agent** — the web UI streams every step (tool, arguments, guardrail verdict, result) above the answer.

## Architecture

```
 Browser (app/static/index.html)  ──SSE──►  server.py ──►  FinAgent loop (agent.py)
                                                              │  ▲
                                          tool_calls (JSON)   ▼  │ tool results
                                                  registry.execute()  ← THE single choke-point
                                  whitelist → JSON parse → schema check → loop guard → semantic check → run → sanitise
                                                              │
                      ┌──────────────────┬───────────────────┼───────────────────┐
                      ▼                  ▼                   ▼                   ▼
             analyze_bank_statement  calculate_emi   search_financial_info    calculator
              (PDF/CSV parser)      (deterministic)  (BM25 RAG + web)        (AST eval)
```

## RAG pipeline: semantic chunking with overlap

```
 knowledge_base/*.md ─► split into sentences ─► embed each sentence (+ heading + neighbours)
        ─► distance between neighbouring windows ─► breakpoints where the topic shifts
        ─► min/max size rules ─► OVERLAP (last N sentences of the previous chunk are repeated)
        ─► BM25 index (optional neural embeddings fused via RRF) ─► top-3 ─► stitch neighbours ─► LLM with sources
```

* **Semantic chunking** (`app/rag/chunker.py`): chunks end where the meaning changes, not at fixed sizes. The default
  embedder is TF-IDF (offline, no download). Set `RAG_EMBEDDER=sbert` after `pip install sentence-transformers` for
  neural embeddings (untested here; falls back to TF-IDF automatically if unavailable).
* **Overlap**: each chunk starts with the last `RAG_OVERLAP_SENTENCES` (default 1) sentence(s) of the previous chunk,
  so a fact that straddles a boundary is complete in at least one chunk. Hits on neighbouring chunks are stitched so the
  LLM never receives the same sentence twice.
* **Heading hint**: the document's own headings add a small boost to breakpoints (`RAG_HEADING_BONUS`, default 0.3).
* **Explore it live**: UI button *Knowledge base explorer* (chunks, overlap highlighted, live retrieval).
* **Measure it**: `python scripts/eval_rag.py` compares strategies. Default result on the 35-question set: retrieval
  accuracy equal to the heading baseline (94.3% top-1, 97.1% complete@3), boundary quality 0.68 vs 0.35 without the
  heading hint, at about 21% more context. The set is small (one question = 2.9 points), so treat the gap as parity,
  not a win. Switch back with `RAG_CHUNKING=heading`.

## Guardrails (slide 8 → code)

| Control | Where | What it does |
|---|---|---|
| Tool whitelisting | `registry.execute` | Unknown tool names are refused; all tools are read-only |
| Input validation | `guardrails.validate_schema` + per-tool `validate` | Types, ranges, enums, unknown args; error text includes a hint so the model can retry |
| Execution loop limits | `LoopGuard`, `FinAgent.max_steps` | Max steps, total + per-tool call budgets, identical-call blocking, forced final answer |
| Data protection | `guardrails.redact`, server upload checks | Masks account numbers/PAN/e-mail/UPI IDs; the model only sees server-issued `file_id`s, never paths; 10 MB, PDF/CSV only |
| Untrusted content | `sanitize_external_text` | Strips prompt-injection phrases from web/search text; system prompt says tool output is data |
| Domain preference | `is_trusted_domain` | Web results kept only from official domains (rbi.org.in, sebi.gov.in, *.gov.in …) |
| Disclaimer | `agent._final` | Appended whenever tools were used |

## Project layout

```
app/agent.py          the agent loop + system prompt          app/registry.py     tool registry + guarded dispatcher
app/guardrails.py     all guardrails                          app/llm.py          OpenAI-compatible client (stdlib)
app/tools/            emi, analyzer, search, calculator       app/rag/            BM25 retriever + knowledge_base/*.md
app/server.py         HTTP + SSE server                       app/static/         single-file UI
app/demo_cases.py     failure/guardrail scenarios             app/mock_llm.py     dev-only rule-based stand-in
samples/              synthetic statements: sample_* (simple) and demo_* (richer, 110 txns)         tests/              unittest suite
```

## Known limitations (say these before the jury asks)

- **Statement parsing** handles the common layout "Date, Description, Amount, Balance" on one line (and CSVs with
  debit/credit columns). Multi-line narrations, scanned PDFs and bank-specific formats are not supported; the tool
  says so instead of guessing. Test with your own bank's format before demoing it.
- **Knowledge base** is a small hand-written corpus. Rates and tax slabs change; the agent is told to mark them as
  "verify on the official site", and the web fallback exists for time-sensitive questions. Review the KB files
  before presenting figures as current.
- Live web search (`ddgs`) depends on network access and on search-engine availability; the agent works without it.
- Not a licensed financial adviser; the UI and every tool-based answer carry a disclaimer.
- Sessions live in memory; restarting the server clears them. Uploaded files are stored under `data/uploads/`.

## Extending (matches the deck's "future expansion")

Add a tool = one new file in `app/tools/` exposing `NAME, DESCRIPTION, SCHEMA, MAX_CALLS, validate(), run()`, then
register it in `app/registry.py`. Ideas from the deck: SIP / tax / credit-score / loan-comparison tools,
budgets and spending trends, more bank formats, Indian-language support.
