"""serve/test_mcp.py - tools from MCP servers (serve/mcp.py) and the web app's tool loop, against the mock engine and
the fake MCP server in serve/mcp_fake_server.py (no GPU, no pack, no MCP SDK).

    python -m unittest serve.test_mcp -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import mcp_fake_server as fake  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.mcp import McpCancelled, McpHub, hub_from_config  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAKE = str(ROOT / "serve" / "mcp_fake_server.py")
CTX = 8192


def stdio(*args, **extra):
    return {"command": sys.executable, "args": [FAKE, *args], **extra}


def call_script(name, **params):
    """The model's text for one tool call, as Qwen's template asks for it."""
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"</think>\n\nLet me check.\n\n<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


class ScriptedEngine(MockEngine):
    """A mock engine with one script per call (the last one repeats): a tool call first, then the answer."""

    def __init__(self, tok, scripts, max_context=CTX, delay_s=0.0):
        super().__init__(tok, list(scripts), max_context=max_context, delay_s=delay_s)
        self.prompts = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)

    def prompt_text(self, i):
        return bytes(t for t in self.prompts[i] if t < 256).decode("utf-8", "replace")


# ------------------------------------------------------------------------------------------------ the client
class Client(unittest.TestCase):
    """The stdio client against the fake server: start, pagination, calls, errors, crashes, timeouts, the cap."""

    @classmethod
    def setUpClass(cls):
        cls.log = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
        cls.log.close()
        os.environ["FAKE_MCP_LOG"] = cls.log.name
        cls.hub = McpHub({"fake": stdio(), "paged": stdio("--page", "4"), "broken": stdio("--crash-at-start"),
                          "missing": {"command": "strata-no-such-program"}},
                         {"timeout_s": 1.5, "max_result_chars": 1000})
        cls.hub.start(wait=True)

    @classmethod
    def tearDownClass(cls):
        cls.hub.close()
        del os.environ["FAKE_MCP_LOG"]
        os.unlink(cls.log.name)

    def test_initialize_and_list_every_page(self):
        s = self.hub.servers["fake"]
        self.assertEqual(s.status, "ready")
        self.assertEqual(s.info["name"], "fake")
        self.assertEqual(s.info["protocol"], "2025-06-18")
        want = [t["name"] for t in fake.TOOLS]
        self.assertEqual([t["name"] for t in s.tools], want)         # 3 pages of 2
        self.assertEqual([t["name"] for t in self.hub.servers["paged"].tools], want)   # 2 pages of 4

    def test_namespaced_openai_tools(self):
        tools = self.hub.openai_tools()
        names = [t["function"]["name"] for t in tools]
        self.assertIn("fake__echo", names)
        self.assertIn("paged__echo", names)
        echo = next(t for t in tools if t["function"]["name"] == "fake__echo")
        self.assertEqual(echo["type"], "function")
        self.assertEqual(echo["function"]["parameters"]["properties"]["text"]["type"], "string")

    def test_a_server_that_fails_is_left_out(self):
        for name, why in (("broken", "missing configuration"), ("missing", "could not start")):
            with self.subTest(name=name):
                s = self.hub.servers[name]
                self.assertEqual(s.status, "failed")
                self.assertIn(why, s.error)
                self.assertFalse([t for t in self.hub.template_tools() if t["name"].startswith(name + "__")])
        st = {s["name"]: s for s in self.hub.status()["servers"]}
        self.assertEqual(st["broken"]["status"], "failed")
        self.assertEqual(st["fake"]["status"], "ready")

    def test_call(self):
        self.hub.routes()
        r = self.hub.call("fake__echo", {"text": "hello wörld"})
        self.assertEqual((r["ok"], r["text"], r["server"], r["tool"]), (True, "hello wörld", "fake", "echo"))
        self.assertEqual(self.hub.call("fake__add", {"a": 2, "b": 40})["text"], "42")

    def test_errors_become_text(self):
        self.hub.routes()
        r = self.hub.call("fake__fail", {})                         # the tool's own error (isError)
        self.assertFalse(r["ok"])
        self.assertEqual(r["text"], "error: it failed on purpose")
        r = self.hub.call("nobody__nothing", {})                    # not a tool at all
        self.assertTrue(r["text"].startswith("error: there is no tool"))
        # a JSON-RPC error from the server
        s = self.hub.servers["fake"]
        with self.assertRaises(Exception) as ctx:
            s.call("no-such-tool", {}, 5)
        self.assertIn("unknown tool", str(ctx.exception))

    def test_timeout(self):
        self.hub.routes()
        t0 = time.monotonic()
        r = self.hub.call("fake__sleep", {"seconds": 5})
        self.assertLess(time.monotonic() - t0, 4)
        self.assertFalse(r["ok"])
        self.assertIn("timed out after 1.5 s", r["text"])
        self.assertEqual(self.hub.call("fake__echo", {"text": "still here"})["text"], "still here")

    def test_truncation(self):
        self.hub.routes()
        r = self.hub.call("fake__big", {"n": 50000})
        self.assertTrue(r["ok"])
        self.assertTrue(r["truncated"])
        self.assertEqual(r["chars"], 50000)
        self.assertTrue(r["text"].startswith("y" * 1000 + "\n\n[... truncated"))
        self.assertIn("50,000 characters", r["text"])
        self.assertLess(len(r["text"]), 1200)

    def test_crash_mid_call_then_restart(self):
        hub = McpHub({"solo": stdio()}, {"timeout_s": 5})
        hub.start(wait=True)
        try:
            hub.routes()
            r = hub.call("solo__die", {})
            self.assertFalse(r["ok"])
            self.assertIn("the server stopped", r["text"])
            self.assertIn("dying on purpose", r["text"])
            self.assertEqual(hub.servers["solo"].status, "stopped")
            self.assertIn("solo__echo", hub.routes())                # its tools stay: the next call restarts it
            self.assertEqual(hub.call("solo__echo", {"text": "back"})["text"], "back")
            self.assertEqual(hub.servers["solo"].status, "ready")
        finally:
            hub.close()

    def test_cancel(self):
        self.hub.routes()
        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        t0 = time.monotonic()
        with self.assertRaises(McpCancelled):
            self.hub.call("fake__sleep", {"seconds": 1.2}, cancel)
        self.assertLess(time.monotonic() - t0, 1.0)
        time.sleep(0.3)
        seen = [json.loads(line) for line in Path(self.log.name).read_text().splitlines()]
        self.assertTrue(any("cancelled" in e for e in seen), seen)   # notifications/cancelled reached the server


class HttpClient(unittest.TestCase):
    """Streamable HTTP: JSON answers, an event-stream answer, the session id."""

    @classmethod
    def setUpClass(cls):
        cls.httpd = fake.http_server(0)
        url = f"http://127.0.0.1:{cls.httpd.server_address[1]}/mcp"
        cls.hub = McpHub({"web": {"url": url}, "down": {"url": "http://127.0.0.1:9/mcp"}}, {"timeout_s": 1.5})
        cls.hub.start(wait=True)

    @classmethod
    def tearDownClass(cls):
        cls.hub.close()
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_list_and_call(self):
        s = self.hub.servers["web"]
        self.assertEqual(s.status, "ready", s.error)
        self.assertEqual(len(s.tools), len(fake.TOOLS))             # pages, with the session id on each request
        self.assertEqual(s.transport.session, "s-1")
        self.hub.routes()
        self.assertEqual(self.hub.call("web__echo", {"text": "over http"})["text"], "over http")   # event stream
        self.assertEqual(self.hub.call("web__fail", {})["text"], "error: it failed on purpose")

    def test_timeout_and_unreachable(self):
        self.hub.routes()
        r = self.hub.call("web__sleep", {"seconds": 4})
        self.assertIn("timed out", r["text"])
        self.assertEqual(self.hub.servers["down"].status, "failed")
        self.assertIn("could not reach", self.hub.servers["down"].error)


class Config(unittest.TestCase):
    def test_both_spellings_and_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "claude_desktop_config.json")
            Path(path).write_text(json.dumps({"mcpServers": {"b": {"url": "http://x/mcp"}, "a": stdio()}}),
                                  encoding="utf-8")
            hub = hub_from_config({"mcp_servers": {"a": {"command": "old"}},
                                   "mcpServers": {"c": stdio(), "off": {"command": "x", "disabled": True}},
                                   "mcp": {"timeout_s": 5, "max_result_chars": 100, "max_rounds": 3}}, path)
            self.assertEqual(sorted(hub.servers), ["a", "b", "c"])
            self.assertEqual(hub.servers["a"].cfg["command"], sys.executable)   # the file wins a name clash
            self.assertEqual(hub.servers["b"].kind, "http")
            self.assertEqual((hub.settings["timeout_s"], hub.settings["max_result_chars"], hub.settings["max_rounds"]),
                             (5.0, 100, 3))
        self.assertIsNone(hub_from_config({}))
        self.assertEqual(McpHub({}).settings["max_result_chars"], 20000)

    def test_bad_entries_stop_the_start(self):
        for cfg in ({"mcp_servers": {"x": {}}}, {"mcp_servers": {"x": {"command": "a", "args": "b"}}},
                    {"mcp_servers": ["x"]}, {"mcpServers": {"x": {"url": "http://a", "type": "sse"}}},
                    {"mcp_servers": {"x": {"command": "a"}}, "mcp": {"timeout_s": 0}},
                    {"mcp_servers": {"x": {"command": "a"}}, "mcp": {"max_rounds": 1.5}}):
            with self.subTest(cfg=cfg), self.assertRaises(SystemExit):
                hub_from_config(cfg)
        with self.assertRaises(SystemExit):
            hub_from_config({}, os.path.join(tempfile.gettempdir(), "strata-no-such-mcp.json"))


# ------------------------------------------------------------------------------------------------ the tool loop
class ToolLoop(unittest.TestCase):
    """The web app's chat (`"strata_mcp": true`): the model calls an MCP tool, the server runs it and the model
    answers with its result; plain API requests never see the MCP tools."""

    def setUp(self):
        self.log = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
        self.log.close()
        os.environ["FAKE_MCP_LOG"] = self.log.name
        self.hub = McpHub({"fake": stdio(), "broken": stdio("--crash-at-start")},
                          {"timeout_s": 10, "max_result_chars": 500, "max_rounds": 2})
        self.hub.start(wait=True)
        self.httpd = None

    def tearDown(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        self.hub.close()
        del os.environ["FAKE_MCP_LOG"]
        os.unlink(self.log.name)

    def start(self, *scripts, delay_s=0.0):
        tok = ByteTokenizer()
        self.engine = ScriptedEngine(tok, list(scripts), delay_s=delay_s)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.mcp = self.hub
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def post(self, body, headers=None, stream=True):
        body = {"model": "m", "messages": [{"role": "user", "content": "check it"}], "stream": stream, **body}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                text = r.read().decode()
                return r.status, text
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read().decode()

    @staticmethod
    def chunks(text):
        return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]

    def test_tool_call_then_answer(self):
        self.start(call_script("fake__echo", text="hello from the tool"), "</think>\n\nThe tool said hello.")
        code, text = self.post({"strata_mcp": True})
        self.assertEqual(code, 200, text)
        cs = self.chunks(text)
        mcp = [c["strata_mcp"] for c in cs if "strata_mcp" in c]
        self.assertEqual([m["event"] for m in mcp], ["start", "call", "result"])
        self.assertEqual(mcp[1]["arguments"], {"text": "hello from the tool"})
        self.assertEqual((mcp[1]["server"], mcp[1]["tool"]), ("fake", "echo"))
        self.assertEqual((mcp[2]["ok"], mcp[2]["text"]), (True, "hello from the tool"))
        self.assertEqual(len({m["id"] for m in mcp}), 1)
        content = "".join((c["choices"][0]["delta"].get("content") or "") for c in cs)
        self.assertEqual(content, "Let me check.The tool said hello.")
        self.assertFalse([c for c in cs if c["choices"][0]["delta"].get("tool_calls")])   # nothing for the client to run
        self.assertEqual(cs[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(len(self.engine.prompts), 2)
        first, second = self.engine.prompt_text(0), self.engine.prompt_text(1)
        self.assertIn('"name": "fake__echo"', first)                 # the MCP tools are in the prompt
        self.assertNotIn("broken__", first)                          # the server that failed is left out
        self.assertIn("<function=fake__echo>\n<parameter=text>\nhello from the tool\n</parameter>", second)
        self.assertIn("<tool_response>\nhello from the tool\n</tool_response>", second)
        self.assertEqual(cs[-1]["usage"]["prompt_tokens"], len(self.engine.prompts[1]))

    def test_a_non_default_effort_round_renders_its_effort_turn_back(self):
        """#458 follow-up ("effort_position": "end"): every round's assistant message carries the request's effort, so
        the next round renders that effort turn back where the engine holds it - round 2's prompt starts with round
        1's prompt exactly, and the round's answer is not read again."""
        self.start(call_script("fake__echo", text="hello from the tool"), "</think>\n\nThe tool said hello.")
        self.svc.effort_end = True
        code, text = self.post({"strata_mcp": True, "reasoning_effort": "low"})
        self.assertEqual(code, 200, text)
        self.assertEqual(len(self.engine.prompts), 2, text)
        first, second = self.engine.prompts[0], self.engine.prompts[1]
        self.assertEqual(second[:len(first)], first, "round 2 did not start with round 1's prompt (the effort turn)")
        self.assertIn("Reasoning effort is set to low", self.engine.prompt_text(1))

    def test_a_call_the_output_ends_inside_is_not_run(self):
        """#211: the model's turn ends inside an MCP call: nothing runs (it used to run with no arguments)."""
        script = call_script("fake__echo", text="hello from the tool")
        self.start(script[:script.index("from the tool")], "</think>\n\nnever")
        code, text = self.post({"strata_mcp": True})
        self.assertEqual(code, 200, text)
        cs = self.chunks(text)
        mcp = [c["strata_mcp"] for c in cs if "strata_mcp" in c]
        self.assertEqual([m["event"] for m in mcp], ["start"])        # the web app shows it as "Not run" at the end
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertNotIn('"call"', Path(self.log.name).read_text())    # nothing ran
        self.assertEqual(cs[-1]["choices"][0]["finish_reason"], "stop")

    def test_a_rounds_calls_run_at_the_same_time(self):
        """Two calls of 1 s each in one round: the round takes ~1 s, and the results go back in the order written."""
        two = ("</think>\n\nBoth.\n\n<tool_call>\n<function=fake__sleep>\n<parameter=seconds>\n1\n</parameter>\n"
               "</function>\n</tool_call>\n<tool_call>\n<function=fake__echo>\n<parameter=text>\nsecond\n"
               "</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=fake__sleep>\n<parameter=seconds>\n1\n"
               "</parameter>\n</function>\n</tool_call>")
        self.start(two, "</think>\n\nDone.")
        t0 = time.monotonic()
        code, text = self.post({"strata_mcp": True})
        took = time.monotonic() - t0
        self.assertEqual(code, 200, text)
        mcp = [c["strata_mcp"] for c in self.chunks(text) if "strata_mcp" in c]
        self.assertEqual([m["event"] for m in mcp if m["event"] != "start"], ["call"] * 3 + ["result"] * 3)
        self.assertLess(took, 1.9)
        second = self.engine.prompt_text(1)
        self.assertLess(second.index("<tool_response>\nsecond"), second.rindex("<tool_response>"))   # order kept

    def test_a_tool_that_crashes_does_not_end_the_answer(self):
        """An unexpected error in a tool call (not McpError) used to end the answer silently, as if it was stopped -
        inside a helper, the helper and every call after it showed "Not run".  Now it is the tool's error."""
        from unittest import mock
        self.hub.settings["subagents"] = True
        self.start(call_script("subagent", task="Look it up."), call_script("fake__echo", text="x"),
                   "</think>\n\nThe tool broke; reporting what I have.", "</think>\n\nDone anyway.")
        with mock.patch.object(self.hub, "call", side_effect=RuntimeError("boom")):
            code, text = self.post({"strata_mcp": True})
        self.assertEqual(code, 200, text)
        cs = self.chunks(text)
        result = [c["strata_mcp"] for c in cs if c.get("strata_mcp", {}).get("event") == "result"]
        self.assertTrue(result[0]["ok"])                                  # the helper reported
        self.assertIn("The tool broke", result[0]["text"])
        self.assertIn("error: the tool failed (RuntimeError: boom)", self.engine.prompt_text(2))
        content = "".join((c["choices"][0]["delta"].get("content") or "") for c in cs)
        self.assertTrue(content.endswith("Done anyway."))

    def test_run_a_helper_from_the_page(self):
        """POST /v1/subagent: the page's Run on a helper block - the helper runs with the task as given, and the
        result comes back as the block's events."""
        self.hub.settings["subagents"] = True
        self.start(call_script("fake__echo", text="found"), "</think>\n\nReport: found it.")
        req = urllib.request.Request(self.base + "/v1/subagent", data=json.dumps(
            {"task": "Find it.", "label": "finder"}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read().decode()
        events = [c["strata_mcp"] for c in self.chunks(text)]
        self.assertIn("progress", [e["event"] for e in events])
        self.assertEqual(events[-1]["event"], "result")
        self.assertTrue(events[-1]["text"].startswith("Report: found it."))
        self.assertIn("Find it.", self.engine.prompt_text(0))
        self.assertNotIn('"name": "subagent"', self.engine.prompt_text(0))     # still no helper inside a helper

    def test_run_a_helper_needs_helpers_on(self):
        self.start("x")
        req = urllib.request.Request(self.base + "/v1/subagent", data=b'{"task": "x"}',
                                     headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=30)
        self.assertEqual(e.exception.code, 404)

    def test_subagent_off_by_default(self):
        self.start("</think>\n\nNo tools needed.")
        self.post({"strata_mcp": True})
        self.assertNotIn('"name": "subagent"', self.engine.prompt_text(0))

    def test_subagent_runs_in_a_fresh_context(self):
        """mcp.subagents: the helper gets only the task, the MCP tools but not itself, and its final answer is the
        call's result; its own tool call and result never enter the main conversation."""
        self.hub.settings["subagents"] = True
        self.start(call_script("subagent", task="Find the magic word.", label="finder"),   # main: hands it off
                   call_script("fake__echo", text="xyzzy-from-the-tool"),                  # helper: uses a tool
                   "</think>\n\nThe magic word is xyzzy.",                                 # helper: its report
                   "</think>\n\nThe helper says: xyzzy.")                                  # main: answers
        code, text = self.post({"strata_mcp": True})
        self.assertEqual(code, 200, text)
        cs = self.chunks(text)
        mcp = [c["strata_mcp"] for c in cs if "strata_mcp" in c]
        events = [m["event"] for m in mcp]
        self.assertEqual(events[:2], ["start", "call"])
        self.assertIn("progress", events)
        self.assertTrue(any("echo" in m.get("text", "") for m in mcp if m["event"] == "progress"))
        result = [m for m in mcp if m["event"] == "result"]
        self.assertEqual(len(result), 1)                               # the helper's tool result is not an event here
        self.assertTrue(result[0]["ok"])
        self.assertEqual(result[0]["text"].split("\n\n[helper id:")[0], "The magic word is xyzzy.")
        self.assertEqual(len(self.engine.prompts), 4)
        main0, helper0, helper1, main1 = (self.engine.prompt_text(i) for i in range(4))
        self.assertIn('"name": "subagent"', main0)
        self.assertNotIn('"name": "subagent"', helper0)                # no helper starts another
        self.assertIn('"name": "fake__echo"', helper0)
        self.assertIn("You are a helper agent", helper0)
        self.assertIn("Find the magic word.", helper0)
        self.assertNotIn("check it", helper0)                          # the main chat is not shown to it
        self.assertIn("<tool_response>\nxyzzy-from-the-tool\n</tool_response>", helper1)
        self.assertTrue(main1.startswith(main0))
        self.assertIn("<tool_response>\nThe magic word is xyzzy.\n\n[helper id: h-", main1)
        self.assertNotIn("xyzzy-from-the-tool", main1)
        content = "".join((c["choices"][0]["delta"].get("content") or "") for c in cs)
        self.assertEqual(content, "Let me check.The helper says: xyzzy.")   # the helper's words are not streamed

    def test_subagent_at_its_round_limit_writes_its_report(self):
        """max_rounds is 2 here: the helper's third call is answered "not run" and it gets one more turn for its
        report, instead of returning the notes it wrote before that call."""
        self.hub.settings["subagents"] = True
        echo = call_script("fake__echo", text="fact")
        self.start(call_script("subagent", task="Research it."),
                   echo, echo, echo,                                              # helper: 2 rounds run, 1 not
                   "</think>\n\nReport: fact, twice.",                            # helper: the wrap-up turn
                   "</think>\n\nDone.")                                           # main
        code, text = self.post({"strata_mcp": True})
        self.assertEqual(code, 200, text)
        mcp = [c["strata_mcp"] for c in self.chunks(text) if "strata_mcp" in c]
        result = [m for m in mcp if m["event"] == "result"]
        self.assertTrue(result[-1]["ok"])
        self.assertTrue(result[-1]["text"].startswith("Report: fact, twice."))
        self.assertIn("limit of 2 tool rounds", result[-1]["text"])
        self.assertIn("used all your tool calls", self.engine.prompt_text(4))
        self.assertEqual(len(self.engine.prompts), 6)

    def test_subagent_has_its_own_round_limit(self):
        self.hub.settings.update(subagents=True, subagent_max_rounds=3)
        echo = call_script("fake__echo", text="fact")
        self.start(call_script("subagent", task="Research it."), echo, echo, echo,   # 3 rounds: more than max_rounds 2
                   "</think>\n\nReport: three facts.", "</think>\n\nDone.")
        code, text = self.post({"strata_mcp": True})
        result = [c["strata_mcp"] for c in self.chunks(text) if c.get("strata_mcp", {}).get("event") == "result"]
        self.assertTrue(result[-1]["text"].startswith("Report: three facts.\n\n[helper id:"))
        self.assertEqual(len(self.engine.prompts), 6)

    def test_subagent_whose_context_fills_up(self):
        """A helper's prompt grows past the context: the main answer goes on with what the helper wrote."""
        self.hub.settings["subagents"] = True
        big = call_script("fake__big")
        self.start(call_script("subagent", task="Read it all."), big, big, big, "</think>\n\nDone.")
        self.engine.max_context = 3600        # the main prompt ~2,800 tokens; each capped result ~560 more
        code, text = self.post({"strata_mcp": True})
        self.assertEqual(code, 200, text)
        result = [c["strata_mcp"] for c in self.chunks(text) if c.get("strata_mcp", {}).get("event") == "result"]
        self.assertIn("stopped before its report", result[0]["text"])     # the helper's (then main reads the rest)

    def _leave_early(self, body):
        """Send a streamed request, read its first event, then hang up (a page reload)."""
        import socket
        payload = json.dumps({"model": "m", "messages": [{"role": "user", "content": "check it"}], "stream": True,
                              **body}).encode()
        sock = socket.create_connection(("127.0.0.1", self.httpd.server_address[1]), timeout=30)
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
        got = b""
        while b"data: {" not in got:
            got += sock.recv(4096)
        self.assertIn(b" 200 ", got.split(b"\r\n", 1)[0])
        sock.close()

    def _get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=60) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read().decode()

    def test_detached_run_survives_the_page(self):
        """strata_run: the page hangs up mid-answer; the run goes on, and GET /v1/runs/<id> streams all of it."""
        self.start(call_script("fake__echo", text="hello from the tool"), "</think>\n\nThe tool said hello.", delay_s=0.01)
        self._leave_early({"strata_mcp": True, "strata_run": "run-abc12345"})
        code, text = self._get("/v1/runs/run-abc12345")
        self.assertEqual(code, 200, text)
        cs = self.chunks(text)
        content = "".join((c["choices"][0]["delta"].get("content") or "") for c in cs)
        self.assertEqual(content, "Let me check.The tool said hello.")
        self.assertTrue(text.rstrip().endswith("data: [DONE]"))
        self.assertEqual(len(self.engine.prompts), 2)                  # it was not cancelled by the hang-up
        code, again = self._get("/v1/runs/run-abc12345")               # a finished run streams again, whole
        self.assertEqual(again, text)
        self.assertEqual(self._get("/v1/runs/run-nosuchrun")[0], 404)

    def _post_json(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_a_message_queued_during_the_run_reaches_the_agent(self):
        """POST /v1/runs/<id>/message while a tool runs: the next round reads it with the tool's result."""
        self.start(call_script("fake__sleep", seconds="1.5"), "</think>\n\nNoted.")
        self._leave_early({"strata_mcp": True, "strata_run": "run-inbox001"})
        code, body = self._post_json("/v1/runs/run-inbox001/message", {"id": "n1", "text": "Share the results."})
        self.assertEqual((code, body["status"]), (200, "queued"))
        code, text = self._get("/v1/runs/run-inbox001")
        notes = [c["strata_mcp"] for c in self.chunks(text) if c.get("strata_mcp", {}).get("event") == "user_note"]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["notes"], ["n1"])
        second = self.engine.prompt_text(1)
        self.assertIn("A message from the user, sent while you were working", second)
        self.assertIn("Share the results.\n</tool_response>", second)
        code, body = self._post_json("/v1/runs/run-inbox001/message", {"id": "n2", "text": "Too late."})
        self.assertEqual((code, body["status"]), (409, "ended"))

    def test_detached_run_cancel(self):
        self.start("</think>\n\n" + "word " * 2000, delay_s=0.01)
        self._leave_early({"strata_mcp": True, "strata_run": "run-cancel01"})
        req = urllib.request.Request(self.base + "/v1/runs/run-cancel01/cancel", data=b"{}",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            self.assertEqual(r.status, 200)
        code, text = self._get("/v1/runs/run-cancel01")
        content = "".join((c["choices"][0]["delta"].get("content") or "") for c in self.chunks(text))
        self.assertLess(content.count("word"), 2000)

    def test_without_strata_run_a_hang_up_still_cancels(self):
        self.start("</think>\n\n" + "word " * 2000, delay_s=0.01)
        self._leave_early({"strata_mcp": True})
        time.sleep(1.5)
        self.assertEqual(self._get("/v1/runs/anything1")[0], 404)

    def test_subagent_continue(self):
        """A report ends with the helper's id; `continue` with that id gives the helper its whole conversation back
        plus the new task."""
        import uuid
        from unittest import mock
        self.hub.settings["subagents"] = True
        self.start(call_script("subagent", task="Find the first fact."),
                   "</think>\n\nFirst fact: alpha.",                                       # helper's report
                   call_script("subagent", task="Now find the second fact.", **{"continue": "h-abcdef"}),
                   "</think>\n\nSecond fact: beta.",                                       # the same helper, again
                   "</think>\n\nBoth found.")
        with mock.patch.object(uuid, "uuid4", return_value=uuid.UUID("abcdef00" * 4)):
            code, text = self.post({"strata_mcp": True})
        result = [c["strata_mcp"] for c in self.chunks(text) if c.get("strata_mcp", {}).get("event") == "result"]
        self.assertIn("helper id: h-abcdef", result[0]["text"])
        self.assertTrue(result[1]["text"].startswith("Second fact: beta."))
        again = self.engine.prompt_text(3)                                # the helper's second turn
        self.assertIn("Find the first fact.", again)
        self.assertIn("First fact: alpha.", again)
        self.assertIn("Now find the second fact.", again)
        self.assertTrue(again.startswith(self.engine.prompt_text(1)[:200]))

    def test_subagent_continue_unknown(self):
        self.hub.settings["subagents"] = True
        self.start(call_script("subagent", task="More.", **{"continue": "h-000000"}), "</think>\n\nOk.")
        code, text = self.post({"strata_mcp": True})
        result = [c["strata_mcp"] for c in self.chunks(text) if c.get("strata_mcp", {}).get("event") == "result"]
        self.assertFalse(result[0]["ok"])
        self.assertIn("no helper", result[0]["text"])

    def test_subagent_without_a_task(self):
        self.hub.settings["subagents"] = True
        self.start(call_script("subagent", label="nobody"), "</think>\n\nOk.")
        code, text = self.post({"strata_mcp": True})
        result = [c["strata_mcp"] for c in self.chunks(text) if c.get("strata_mcp", {}).get("event") == "result"]
        self.assertFalse(result[0]["ok"])
        self.assertIn("task", result[0]["text"])
        self.assertEqual(len(self.engine.prompts), 2)

    def test_non_stream(self):
        self.start(call_script("fake__add", a=2, b=3), "</think>\n\n5.")
        code, text = self.post({"strata_mcp": True}, stream=False)
        self.assertEqual(code, 200, text)
        msg = json.loads(text)["choices"][0]["message"]
        self.assertEqual(msg["content"], "Let me check.5.")
        self.assertEqual(msg["strata_mcp"][-1]["text"], "5")

    def test_plain_requests_get_no_mcp_tools(self):
        self.start(call_script("fake__echo", text="x"), "</think>\n\nnever")
        code, text = self.post({})
        self.assertEqual(code, 200, text)
        self.assertNotIn("fake__echo", self.engine.prompt_text(0))
        cs = self.chunks(text)
        self.assertFalse([c for c in cs if "strata_mcp" in c])
        self.assertEqual(cs[-1]["choices"][0]["finish_reason"], "tool_calls")   # returned to the client as always
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertNotIn('"call"', Path(self.log.name).read_text())    # nothing ran
        for req in ({"strata_mcp": "yes"}, {"strata_mcp": 1}):         # only a real true opts in
            self.post(req)
            self.assertNotIn("fake__echo", self.engine.prompt_text(-1))

    def test_own_tools_still_go_to_the_client(self):
        own = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {
            "q": {"type": "string"}}}}}]
        self.start(call_script("lookup", q="x"), "</think>\n\nnever")
        code, text = self.post({"strata_mcp": True, "tools": own})
        cs = self.chunks(text)
        calls = [tc for c in cs for tc in c["choices"][0]["delta"].get("tool_calls") or []]
        self.assertEqual(calls[0]["function"]["name"], "lookup")
        self.assertEqual(cs[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertIn('"name": "lookup"', self.engine.prompt_text(0))
        self.assertIn('"name": "fake__echo"', self.engine.prompt_text(0))
        self.assertEqual(len(self.engine.prompts), 1)

    def test_tool_error_is_a_result(self):
        self.start(call_script("fake__fail"), "</think>\n\nIt failed.")
        code, text = self.post({"strata_mcp": True})
        mcp = [c["strata_mcp"] for c in self.chunks(text) if "strata_mcp" in c]
        self.assertEqual((mcp[-1]["ok"], mcp[-1]["text"]), (False, "error: it failed on purpose"))
        self.assertIn("<tool_response>\nerror: it failed on purpose\n</tool_response>", self.engine.prompt_text(1))
        self.assertEqual(self.chunks(text)[-1]["choices"][0]["finish_reason"], "stop")

    def test_truncated_for_the_model(self):
        self.start(call_script("fake__big", n=3000), "</think>\n\nLong.")
        self.post({"strata_mcp": True})
        second = self.engine.prompt_text(1)
        self.assertIn("y" * 500 + "\n\n[... truncated: the tool returned 3,000 characters", second)
        self.assertNotIn("y" * 501, second)

    def test_max_rounds(self):
        self.start(call_script("fake__echo", text="again"))          # the model never stops calling
        code, text = self.post({"strata_mcp": True})
        mcp = [c["strata_mcp"] for c in self.chunks(text) if "strata_mcp" in c]
        self.assertEqual(len(self.engine.prompts), 3)                # 2 rounds of tools, then the limit
        self.assertEqual(sum(m["event"] == "call" for m in mcp), 2)
        self.assertIn({"event": "limit", "max_rounds": 2}, mcp)
        self.assertTrue(mcp[-1].get("skipped"))
        self.assertEqual(self.chunks(text)[-1]["choices"][0]["finish_reason"], "stop")

    def test_stop_during_a_tool(self):
        """Closing the connection while a slow tool runs stops the tool (notifications/cancelled) and the loop."""
        self.start(call_script("fake__sleep", seconds=8), "</think>\n\nnever")
        body = {"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": True, "strata_mcp": True}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=30)
        for raw in r:
            if b'"event": "call"' in raw:
                break
        r.close()                                                    # the web app's Stop aborts the fetch
        t0 = time.monotonic()
        while time.monotonic() - t0 < 6 and "cancelled" not in Path(self.log.name).read_text():
            time.sleep(0.1)
        self.assertIn("cancelled", Path(self.log.name).read_text())
        self.assertLess(time.monotonic() - t0, 6)
        time.sleep(0.5)
        self.assertEqual(len(self.engine.prompts), 1)                # no second round
        self.assertFalse(self.svc.status["busy"])

    def test_only_from_the_app_s_own_page(self):
        self.start("</think>\n\nhi")
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(
            {"model": "m", "messages": [{"role": "user", "content": "x"}], "strata_mcp": True}).encode(),
            headers={"Content-Type": "text/plain"})                  # a cross-site "simple" request
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(ctx.exception.code, 415)
        code, _ = self.post({"strata_mcp": True}, {"Origin": "http://evil.example"})
        self.assertEqual(code, 403)
        self.assertEqual(self.engine.prompts, [])
        host = self.base.split("://", 1)[1]
        code, _ = self.post({"strata_mcp": True}, {"Origin": "http://" + host})
        self.assertEqual(code, 200)

    def test_get_mcp(self):
        self.start("</think>\n\nhi")
        with urllib.request.urlopen(self.base + "/mcp", timeout=10) as r:
            st = json.loads(r.read())
        servers = {s["name"]: s for s in st["servers"]}
        self.assertEqual(servers["fake"]["status"], "ready")
        self.assertEqual(servers["fake"]["transport"], "stdio")
        self.assertIn("fake__echo", [t["name"] for t in servers["fake"]["tools"]])
        self.assertEqual(servers["broken"]["status"], "failed")
        self.assertIn("missing configuration", servers["broken"]["error"])
        self.assertEqual(st["tools"], len(fake.TOOLS))
        self.assertEqual(st["settings"]["max_rounds"], 2)
        self.svc.api_key = "k"
        with self.assertRaises(urllib.error.HTTPError):
            urllib.request.urlopen(self.base + "/mcp", timeout=10)
        self.svc.api_key = ""


class NoServers(unittest.TestCase):
    def test_opt_in_without_servers_is_a_plain_chat(self):
        tok = ByteTokenizer()
        svc = Service(ScriptedEngine(tok, ["</think>\n\nhi"]), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            with urllib.request.urlopen(base + "/mcp", timeout=10) as r:
                self.assertEqual(json.loads(r.read()), {"servers": [], "tools": 0})
            req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(
                {"model": "m", "messages": [{"role": "user", "content": "x"}], "strata_mcp": True}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                self.assertEqual(json.loads(r.read())["choices"][0]["message"]["content"], "hi")
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
