"""Tests the agent loop with (a) a scripted fake model and (b) the real HTTP client against a local fake API."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from app.agent import FinAgent
from app.guardrails import DISCLAIMER
from app.llm import LLMClient, LLMError
from app.registry import ToolContext

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "sample_statement.csv"


def call(i, name, args):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


class Scripted:
    """Plays back a fixed list of assistant messages and records what it was sent."""
    def __init__(self, script): self.script, self.seen = list(script), []
    def chat(self, messages, tools=None):
        self.seen.append((json.loads(json.dumps(messages)), tools))
        return self.script.pop(0)


class AgentLoopTests(unittest.TestCase):
    def ctx(self):
        return ToolContext(files={"f_0000abcd": {"path": SAMPLE, "name": "s.csv"}})

    def test_multi_step_chain(self):
        llm = Scripted([
            call(1, "analyze_bank_statement", {"file_id": "f_0000abcd"}),
            call(2, "calculate_emi", {"principal": 4000000, "annual_rate_percent": 9, "tenure_value": 20,
                                      "tenure_unit": "years", "monthly_income": 85000, "existing_monthly_obligations": 8500}),
            {"role": "assistant", "content": "Your EMI would be ₹35,989."}])
        events = list(FinAgent(llm).run("Can I afford a 40 lakh loan?", [], self.ctx()))
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds, ["tool_call", "tool_result", "tool_call", "tool_result", "final"])
        self.assertTrue(all(e["ok"] for e in events if e["type"] == "tool_result"))
        self.assertIn(DISCLAIMER, events[-1]["content"])
        # the 2nd model call must have received the tool result of the 1st
        second_msgs = llm.seen[1][0]
        self.assertEqual(second_msgs[-1]["role"], "tool")
        self.assertIn("averages_per_month", second_msgs[-1]["content"])
        # system prompt exposes file_id but not the server path
        self.assertIn("f_0000abcd", llm.seen[0][0][0]["content"]); self.assertNotIn(str(SAMPLE), llm.seen[0][0][0]["content"])

    def test_self_correction_after_guardrail_block(self):
        """The failure from Step 5: model passes 0.085, is blocked with a hint, retries correctly."""
        bad = {"principal": 5000000, "annual_rate_percent": 0.085, "tenure_value": 20, "tenure_unit": "years"}
        good = {**bad, "annual_rate_percent": 8.5}
        llm = Scripted([call(1, "calculate_emi", bad), call(2, "calculate_emi", good),
                        {"role": "assistant", "content": "EMI is ₹43,391."}])
        events = list(FinAgent(llm).run("EMI?", [], self.ctx()))
        results = [e for e in events if e["type"] == "tool_result"]
        self.assertFalse(results[0]["ok"]); self.assertEqual(results[0]["guardrail"], "input_validation")
        self.assertTrue(results[1]["ok"])
        self.assertIn("NOT 0.085", llm.seen[1][0][-1]["content"])  # the hint reached the model

    def test_step_limit_forces_final_answer(self):
        endless = [call(i, "calculator", {"expression": f"{i}+1"}) for i in range(1, 10)]
        llm = Scripted(endless[:3] + [{"role": "assistant", "content": "Here is what I have."}])
        events = list(FinAgent(llm, max_steps=3).run("loop", [], self.ctx()))
        self.assertEqual(events[-2]["type"], "guardrail")
        self.assertEqual(events[-1]["type"], "final")
        self.assertIsNone(llm.seen[-1][1])  # final call made WITHOUT tools

    def test_fake_tool_is_blocked_and_loop_continues(self):
        llm = Scripted([call(1, "transfer_money", {"amount": 5}), {"role": "assistant", "content": "I can't do that."}])
        events = list(FinAgent(llm).run("send money", [], self.ctx()))
        self.assertEqual(events[1]["guardrail"], "tool_whitelist")
        self.assertEqual(events[-1]["type"], "final")
        self.assertNotIn(DISCLAIMER, events[-1]["content"])  # no successful tool use -> no disclaimer needed


class FakeOpenAI(BaseHTTPRequestHandler):
    """Local stand-in for an OpenAI-compatible /chat/completions endpoint."""
    mode = "ok"
    hits = 0

    def do_POST(self):
        FakeOpenAI.hits += 1
        assert "urllib" not in self.headers.get("User-Agent", "").lower()
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.headers["Authorization"] == "Bearer test-key" or FakeOpenAI.mode == "auth"
        if FakeOpenAI.mode == "auth":
            self.send_response(401); self.end_headers(); self.wfile.write(b'{"error":"bad key"}'); return
        if FakeOpenAI.mode == "flaky" and FakeOpenAI.hits == 1:
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":{"code":"tool_use_failed"}}'); return
        has_tool_result = any(m["role"] == "tool" for m in body["messages"])
        if not has_tool_result:
            msg = call(1, "calculator", {"expression": "2+2"})
        else:
            msg = {"role": "assistant", "content": "The answer is 4."}
        out = json.dumps({"choices": [{"message": msg}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(out)

    def log_message(self, *a): pass


class RealClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), FakeOpenAI)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls): cls.srv.shutdown()

    def setUp(self): FakeOpenAI.mode, FakeOpenAI.hits = "ok", 0

    def test_http_function_calling_roundtrip(self):
        agent = FinAgent(LLMClient("test-key", self.url, "m"))
        events = list(agent.run("what is 2+2", [], ToolContext()))
        self.assertEqual([e["type"] for e in events], ["tool_call", "tool_result", "final"])
        self.assertEqual(events[1]["result"]["result"], 4)
        self.assertIn("The answer is 4.", events[-1]["content"])

    def test_bad_key_gives_friendly_error(self):
        FakeOpenAI.mode = "auth"
        events = list(FinAgent(LLMClient("x", self.url, "m")).run("hi", [], ToolContext()))
        self.assertEqual(events[0]["type"], "error"); self.assertIn("API key", events[0]["message"])

    def test_retries_malformed_tool_call(self):
        FakeOpenAI.mode = "flaky"
        events = list(FinAgent(LLMClient("test-key", self.url, "m")).run("what is 2+2", [], ToolContext()))
        self.assertEqual(events[-1]["type"], "final")


if __name__ == "__main__":
    unittest.main()
