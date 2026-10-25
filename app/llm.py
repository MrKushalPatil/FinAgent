"""
Minimal client for any OpenAI-compatible chat-completions API with function calling
(Groq, OpenAI, Gemini's OpenAI endpoint, Ollama, vLLM, ...). Standard library only.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .config import settings


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 60):
        self.api_key, self.base_url, self.model, self.timeout = api_key, base_url.rstrip("/"), model, timeout

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Returns the assistant message dict: {'role','content','tool_calls'?}."""
        payload: dict = {"model": self.model, "messages": messages, "temperature": 0.1}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        body = json.dumps(payload).encode()
        last_err = "unknown error"
        for attempt in range(3):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=body, method="POST",
                headers={"Content-Type": "application/json", "Accept": "application/json",
                         "Authorization": f"Bearer {self.api_key}",
                         # Cloudflare-fronted APIs (e.g. Groq) reject Python's default urllib user-agent with a 403.
                         "User-Agent": "FinAgent/1.0 (+python)"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                return data["choices"][0]["message"]
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:400]
                last_err = f"HTTP {e.code}: {detail}"
                # 429/5xx are transient; 400 'tool_use_failed' means the model emitted a malformed call -> retry
                transient = e.code in (429, 500, 502, 503, 504) or (e.code == 400 and "tool_use_failed" in detail)
                if e.code == 401:
                    raise LLMError("The LLM provider rejected the API key (401). Check LLM_API_KEY in .env: "
                                   "no quotes or spaces, and the key must belong to the provider in LLM_BASE_URL.")
                if e.code == 403:
                    raise LLMError(f"The LLM provider refused the request (403). Check that your key is active and "
                                   f"that LLM_MODEL is available to your account. Provider said: {detail[:160]}")
                if not transient:
                    raise LLMError(last_err)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"LLM request failed after retries ({last_err})")


def make_llm():
    if settings.provider == "mock":
        from .mock_llm import MockLLM
        return MockLLM()
    return LLMClient(settings.api_key, settings.base_url, settings.model)
