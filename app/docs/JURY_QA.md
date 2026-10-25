# Jury Q&A cheat sheet

## Q1. How does the model decide to call a tool?

**One-line answer:** We give the model a menu of tools (name + description + parameter schema). It is trained for
*function calling*, so instead of plain text it can reply with a structured request like
`calculate_emi({"principal": 5000000, ...})`. It chooses by matching the user's intent to the tool descriptions.
Our code then validates and runs the call and returns the result to the model.

**Walk-through (use the UI):**
1. Click **"Tool definitions"** — show that the model gets only names, descriptions and schemas, never code.
2. Ask: *"Based on my statement, can I afford a ₹40 lakh loan at 9% for 20 years?"* (attach the sample statement).
3. Point at the ledger: Step 1 `analyze_bank_statement` → Step 2 `calculate_emi` using the income and EMIs from step 1
   → final answer. "Nobody clicked a tool button. The model decided WHEN, WHICH, and WHETHER another is needed."
4. Ask *"What is the PPF lock-in period?"* → it picks `search_financial_info` instead. Ask *"What is 18% of 85000?"* → `calculator`.
   Ask *"Hi, what can you do?"* → **no tool call**. That is the "when" decision.

**What influences the decision:** (a) tool descriptions say when to use / not use each tool,
(b) the system prompt rules ("never compute EMI yourself", "ask if inputs are missing"),
(c) the files list in the system message, (d) previous tool results in the conversation.
`temperature=0.1` keeps routing stable.

**Where in code:** `app/agent.py` (docstring + loop), tool descriptions in `app/tools/*.py`, schemas via `GET /api/tools`.

## Q2. What guardrails prevent tool misuse?

**One-line answer:** The model only *requests* actions; every request passes one choke-point
(`registry.execute`) that whitelists the tool, validates arguments, enforces budgets, and sanitises output.

| Risk | Control | Proof in the demo |
|---|---|---|
| Hallucinated / dangerous tool | Whitelist, read-only tools | Safeguard demonstration #3 |
| Wrong units (0.085 vs 8.5, 50 vs 50 lakh) | Schema ranges + hint in the error → model self-corrects | Lab #1, #2; `test_self_correction_after_guardrail_block` |
| File/path abuse | Model sees server-issued `file_id`s only | Lab #4 |
| Infinite / wasteful loops | Max steps, call budgets, duplicate-call blocking, forced final answer | Lab #5 |
| Prompt injection via web/PDF text | Sanitiser + "tool output is data" rule + trusted domains only | Lab #6 |
| Privacy | Redaction of account no./PAN/e-mail/UPI IDs; upload limits | `test_redaction` |
| Arithmetic hallucination | Math is done by deterministic tools, not the LLM | `calculate_emi`, `calculator` |
| Over-trust | Disclaimer on tool-based answers; no stock/fund picks | System prompt rule 7 |

## Other likely questions

**Why not just let the LLM do the maths?** LLMs make arithmetic mistakes. EMI is a formula, so a deterministic tool
gives exact, testable results (`test_known_value`: ₹50L @ 8.5% / 20y = ₹43,391).

**What is RAG here and why BM25?** Retrieval-Augmented Generation: the agent retrieves passages from a curated
knowledge base (`app/rag/knowledge_base/`) and answers from them with source URLs. BM25 needs no embedding model
or GPU, is explainable (term overlap weighted by rarity), and works well on a small keyword-rich corpus.
Upgrade path: swap `Retriever.search` for embeddings + a vector store.

**What if the knowledge base doesn't have the answer?** `auto` mode falls back to a web search restricted to official
domains; if that also fails, the tool returns "no reliable results" and the model is told to say so.

**How do you stop the model from making up a loan rate?** Rule 2 in the system prompt + schema `required` fields:
missing values are rejected and the model is told to ask the user.

**What if the model calls the wrong tool?** The call is still validated, so it cannot do harm; wrong-tool errors
(e.g. unknown `file_id`) come back as readable messages and the model retries or asks the user.

**What if the LLM provider is down?** The client retries transient errors (429/5xx and malformed tool calls), then
shows a clear error. Ollama can run the same agent fully offline.

**How would you evaluate it?** Unit tests for tools/guardrails (30 tests), scripted-agent tests for the loop, and a set of
prompt → expected-tool-sequence cases (e.g. affordability question must call analyzer then EMI). Next step: log
real conversations and measure tool-selection accuracy and block rate.

**Is it secure to upload bank statements?** Demo uses synthetic data. Files stay on the local server under
`data/uploads/`, are never sent to the LLM raw (only the computed summary, with identifiers masked), and the model
cannot choose paths. For production add encryption at rest, retention limits, and auth.
