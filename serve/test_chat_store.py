"""local: the web chat kept on the server (chat_dir): versions, a save from an older version refused with the current
chat, New chat's archive, a restart keeps it, and the routes (own page only, 404 without chat_dir)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.chat_store import ChatStore  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Store(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "chats"

    def tearDown(self):
        self.tmp.cleanup()

    def test_versions_conflicts_and_restart(self):
        s = ChatStore(self.dir)
        self.assertEqual(s.state(), {"version": 0, "updated": 0.0, "messages": []})
        ok, st = s.save(0, [{"role": "user", "text": "hi"}])
        self.assertTrue(ok)
        self.assertEqual(st["version"], 1)
        ok, st = s.save(0, [{"role": "user", "text": "from another device"}])   # made from an older version
        self.assertFalse(ok)
        self.assertEqual((st["version"], st["messages"][0]["text"]), (1, "hi"))
        again = ChatStore(self.dir)                                          # a restart reads it back
        self.assertEqual((again.version, again.messages[0]["text"]), (1, "hi"))
        self.assertFalse(list(self.dir.glob("*.tmp")))

    def test_new_chat_archives_the_old_one(self):
        s = ChatStore(self.dir)
        s.save(0, [{"role": "user", "text": "old"}])
        ok, _ = s.save(1, [], archive=True)
        self.assertTrue(ok)
        saved = list((self.dir / "archive").glob("*.json"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(json.loads(saved[0].read_text(encoding="utf-8"))["messages"][0]["text"], "old")

    def test_unreadable_file_is_kept_aside(self):
        self.dir.mkdir(parents=True)
        (self.dir / "current.json").write_text("{not json", encoding="utf-8")
        s = ChatStore(self.dir)
        self.assertEqual(s.messages, [])
        self.assertTrue(list(self.dir.glob("current.unreadable-*.json")))


class Routes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, ["hi"]), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.chat_store = ChatStore(Path(self.tmp.name))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def call(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_save_load_and_conflict(self):
        self.assertEqual(self.call("/v1/chat-store")[1]["version"], 0)
        code, st = self.call("/v1/chat-store", {"version": 0, "messages": [{"role": "user", "text": "a"}]})
        self.assertEqual((code, st["version"]), (200, 1))
        code, st = self.call("/v1/chat-store")
        self.assertEqual(st["messages"][0]["text"], "a")
        code, st = self.call("/v1/chat-store?v=1")                     # unchanged: only the version
        self.assertEqual((code, "messages" in st), (200, False))
        code, st = self.call("/v1/chat-store", {"version": 0, "messages": []})
        self.assertEqual((code, st["messages"][0]["text"]), (409, "a"))
        self.assertEqual(self.call("/v1/chat-store", {"version": "1", "messages": []})[0], 400)

    def test_other_pages_are_refused(self):
        evil = {"Origin": "http://evil.example"}
        self.assertEqual(self.call("/v1/chat-store", headers=evil)[0], 403)
        self.assertEqual(self.call("/v1/chat-store", {"version": 0, "messages": []}, headers=evil)[0], 403)

    def test_without_chat_dir(self):
        self.svc.chat_store = None
        self.assertEqual(self.call("/v1/chat-store")[0], 404)


if __name__ == "__main__":
    unittest.main()
