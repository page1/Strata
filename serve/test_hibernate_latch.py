"""serve/test_hibernate_latch.py - hibernate through #1937's persistent disk tier: POST /unload {"save": true} ends the
engine (QUIT, which writes its conversations to --conversation-disk-dir) and a latch file holds the model unloaded -
requests answer 503 - until POST /load starts it again (mock engine: no GPU, no pack).

    python -m unittest serve.test_hibernate_latch -v
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import HIBERNATE_LATCH, ByteTokenizer, MockEngine, Service, hibernate_latch, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class StoppableEngine(MockEngine):
    """Stopped (QUIT) and started like StrataEngine."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.running, self.quits, self.starts = True, 0, 0

    def alive(self):
        return self.running

    def unload(self):
        self.running = False
        self.quits += 1

    def restart(self):
        self.running = True
        self.starts += 1


class LatchPath(unittest.TestCase):
    def test_only_with_the_disk_tier_in_persist_mode(self):
        d = ["--conversation-disk-dir", "/tmp/conv"]
        self.assertEqual(hibernate_latch(d + ["--conversation-disk-persist", "on"]), Path("/tmp/conv") / HIBERNATE_LATCH)
        self.assertEqual(hibernate_latch(["--conversation-disk-persist", "exit-only"] + d),
                         Path("/tmp/conv") / HIBERNATE_LATCH)
        self.assertIsNone(hibernate_latch(d + ["--conversation-disk-persist", "off"]))
        self.assertIsNone(hibernate_latch(d))
        self.assertIsNone(hibernate_latch(["--conversation-disk-persist", "on"]))

    def test_a_relative_folder_is_the_configs_cwd(self):
        args = ["--conversation-disk-dir", "conv", "--conversation-disk-persist", "on"]
        self.assertEqual(hibernate_latch(args, cwd="/srv/strata"), Path("/srv/strata/conv") / HIBERNATE_LATCH)

    def test_the_disk_tier_leaves_the_latch_alone(self):
        # conversation_disk.cpp removes only its own files (strata-conv-*) at start and at the end
        self.assertFalse(HIBERNATE_LATCH.startswith("strata-conv-"))


class Latch(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = StoppableEngine(tok, "</think>\n\nok", max_context=4096)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.tmp = tempfile.TemporaryDirectory()
        self.latch = Path(self.tmp.name) / "conv" / HIBERNATE_LATCH
        self.svc.hibernate_latch = self.latch
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def req(self, path, body=None, method="POST"):
        data = None if body is None else json.dumps(body).encode()
        r = urllib.request.Request(self.base + path, data=data, method=method,
                                   headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def chat(self):
        return self.req("/v1/chat/completions", {"model": "x", "max_tokens": 4,
                                                 "messages": [{"role": "user", "content": "hi"}]})

    def health(self):
        return self.req("/health", method="GET")[1]

    def test_save_quits_latches_and_answers_503_until_load(self):
        s, b = self.req("/unload", {"save": True})
        self.assertEqual((s, b.get("status"), b.get("hibernated")), (200, "unloaded", True))
        self.assertEqual(self.engine.quits, 1)                      # the QUIT that writes the conversations
        self.assertTrue(self.latch.exists())
        h = self.health()
        self.assertEqual((h["hibernated"], h["loaded"]), (True, False))
        s, b = self.chat()
        self.assertEqual(s, 503)
        self.assertIn("POST /load", b["error"]["message"])
        self.assertEqual(self.engine.starts, 0)                     # nothing loaded it back
        s, b = self.req("/load", {})
        self.assertEqual((s, b.get("status")), (200, "loaded"))
        self.assertFalse(self.latch.exists())
        self.assertEqual(self.engine.starts, 1)
        self.assertEqual(self.chat()[0], 200)
        self.assertFalse(self.health()["hibernated"])

    def test_a_plain_unload_does_not_latch(self):
        s, b = self.req("/unload", {})
        self.assertEqual((s, b.get("status")), (200, "unloaded"))
        self.assertFalse(self.latch.exists())
        self.assertEqual(self.chat()[0], 200)                       # the next request loads it, as before
        self.assertEqual(self.engine.starts, 1)

    def test_save_while_unloaded_latches_without_a_second_quit(self):
        self.req("/unload", {})
        s, b = self.req("/unload", {"save": True})
        self.assertEqual((s, b.get("status"), b.get("hibernated")), (200, "not loaded", True))
        self.assertEqual(self.engine.quits, 1)
        self.assertEqual(self.chat()[0], 503)

    def test_without_the_disk_tier_save_is_refused(self):
        self.svc.hibernate_latch = None
        s, b = self.req("/unload", {"save": True})
        self.assertEqual(s, 400)
        self.assertIn("--conversation-disk-persist", b["error"]["message"])
        self.assertTrue(self.engine.alive())                        # nothing was stopped

    def test_v1_unload_save_and_v1_load(self):
        s, b = self.req("/v1/unload", {"save": True})
        self.assertEqual((s, b.get("status")), (200, "unloaded"))
        self.assertTrue(self.latch.exists())
        self.assertEqual(self.chat()[0], 503)
        s, b = self.req("/v1/load", {})
        self.assertEqual((s, b.get("status")), (200, "loaded"))
        self.assertFalse(self.latch.exists())
        self.assertEqual(self.chat()[0], 200)

    def test_the_latch_survives_a_new_service(self):
        self.req("/unload", {"save": True})
        tok = ByteTokenizer()
        other = Service(StoppableEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        other.hibernate_latch = self.latch                         # as main() sets it from the config's args
        self.assertTrue(other.hibernated())


if __name__ == "__main__":
    unittest.main()
