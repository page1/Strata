"""tools/test_lease_waker.py - my own tests for job lease-4b-waker beyond the verdict: the CLI's lease folder and
clean-shutdown choice, the model-only key on the new lease routes, wake_all's skip set, and run_leased's log and
on_lease hook.  No Strata, no GPU, no real opencode or claude: fake servers and temp files.

    python tools/test_lease_waker.py
"""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import strata_jobs as J  # noqa: E402
import strata_suspend as S  # noqa: E402

PY = sys.executable
LID = "ab" * 16


class LeaseFolder(unittest.TestCase):
    def test_leases_dir_wins(self):
        self.assertEqual(S.lease_folder_from_config({"leases_dir": "D:/x/leases", "hibernate_dir": "D:/h"}),
                         "D:/x/leases")

    def test_hibernate_dir_default(self):
        self.assertEqual(S.lease_folder_from_config({"hibernate_dir": "D:/h"}), str(Path("D:/h") / "leases"))

    def test_neither(self):
        self.assertIsNone(S.lease_folder_from_config({}))


class ModelKey(unittest.TestCase):
    def setUp(self):
        from serve.server import model_key_control_ok
        self.ok = model_key_control_ok

    def test_lease_with_log(self):
        self.assertTrue(self.ok("/lease", {"cmd": "c", "cwd": ".", "mode": "full", "session": None,
                                           "log": "a.log"}))
        self.assertTrue(self.ok("/lease", {"cmd": "c", "cwd": ".", "mode": "full", "log": None}))
        self.assertFalse(self.ok("/lease", {"cmd": "c", "cwd": ".", "mode": "full", "fresh": True}))

    def test_woken_route(self):
        self.assertTrue(self.ok(f"/lease/{LID}/woken", {}))
        self.assertFalse(self.ok(f"/lease/{LID}/woken", {"x": 1}))

    def test_session_woken_route(self):
        self.assertTrue(self.ok("/lease/session-woken", {"agent": "opencode", "id": "ses_1"}))
        self.assertFalse(self.ok("/lease/session-woken", {"agent": "opencode", "id": "s", "x": 1}))
        self.assertFalse(self.ok("/lease/session-woken", {}))


class RunLeasedLog(unittest.TestCase):
    def test_log_goes_in_the_lease_and_on_lease_fires(self):
        calls = []
        fake = {}

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                calls.append((self.path, body))
                data = json.dumps({"id": LID, "status": "unloaded"} if self.path == "/lease"
                                  else {"id": LID}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        log = str(Path(tempfile.mkdtemp()) / "a.log")        # run_leased opens the job's log itself: a real path
        try:
            rc = S.run_leased(f"http://127.0.0.1:{srv.server_address[1]}", None, [PY, "-c", "pass"], env={},
                              renew_s=0.2, log=log, on_lease=lambda lid: fake.setdefault("lid", lid))
            self.assertEqual(rc, 0)
            self.assertEqual(fake["lid"], LID)
            self.assertEqual(calls[0][1]["log"], log)
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
