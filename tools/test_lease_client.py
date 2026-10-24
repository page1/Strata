"""tools/test_lease_client.py - my own tests for the lease client in tools/strata_suspend.py: session_from_env,
the CLI taking the lease path by default whenever the server answers (--lease and STRATA_SUSPEND_LEASES are accepted
and change nothing; an older server without /lease refuses the job), and Ctrl+C in run_leased.  No Strata, no GPU:
a fake lease server and short real processes.

    python tools/test_lease_client.py
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_jobs as J  # noqa: E402
import strata_suspend as S  # noqa: E402

PY = sys.executable
LID = "cd" * 16


class FakeServer:
    """A stand-in Strata: /leases answers `leases_status`; /lease, /lease/<id>/renew, /lease/<id>/end work;
    /unload answers 200 "unloaded" (the old pid-hold path)."""

    def __init__(self, leases_status=200):
        self.calls = []
        self.leases_status = leases_status
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/leases":
                    self._send(fake.leases_status, {"leases": []} if fake.leases_status == 200 else {})
                elif self.path == "/health":
                    self._send(200, {"status": "ok", "loaded": True})
                else:
                    self._send(404, {})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                fake.calls.append((self.path, body))
                if self.path == "/lease":
                    if fake.leases_status != 200:            # an older server: no lease route at all
                        self._send(404, {"error": {"message": "no such route"}})
                    else:
                        self._send(200, {"id": LID, "status": "unloaded"})
                elif self.path.endswith("/renew"):
                    self._send(200, {"id": LID})
                elif self.path.endswith("/end"):
                    self._send(200, {"id": LID, "ended": "released", "exit": body.get("exit")})
                elif self.path == "/unload":
                    self._send(200, {"status": "unloaded", "held_by": body.get("hold_pids") or []})
                elif self.path == "/hold":
                    self._send(200, {"held_by": body.get("hold_pids") or []})
                else:
                    self._send(404, {})

            def _send(self, st, obj):
                data = json.dumps(obj).encode()
                self.send_response(st)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def paths(self):
        return [p for p, _ in self.calls]

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


class Session(unittest.TestCase):
    def test_claude_wins_over_opencode(self):
        self.assertEqual(S.session_from_env({"CLAUDE_CODE_SESSION_ID": "c", "OPENCODE_SESSION_ID": "o"}),
                         {"agent": "claude", "id": "c"})

    def test_opencode(self):
        self.assertEqual(S.session_from_env({"OPENCODE_SESSION_ID": "o"}), {"agent": "opencode", "id": "o"})

    def test_none(self):
        self.assertIsNone(S.session_from_env({"OPENCODE_PID": "123"}))


class RunLeased(unittest.TestCase):
    def test_ctrl_c_kills_the_job_and_ends_the_lease(self):
        f = FakeServer()
        try:
            real_popen = subprocess.Popen

            class Boom:
                pid = 999999

                def poll(self):
                    raise KeyboardInterrupt

                def wait(self):
                    return 130

            def popen(cmd, **kw):
                return Boom() if "-c" in cmd else real_popen(cmd, **kw)

            S.subprocess.Popen = popen
            S.stop_tree = lambda pid: None
            try:
                rc = S.run_leased(f.base, None, [PY, "-c", "pass"], env={}, renew_s=0.2)
            finally:
                S.subprocess.Popen = real_popen
            self.assertEqual(rc, 130)
            self.assertEqual(f.paths()[-1], f"/lease/{LID}/end")
            self.assertEqual(f.calls[-1][1].get("exit"), 130)
        finally:
            f.close()

    def test_no_renewal_after_the_end(self):
        f = FakeServer()
        try:
            rc = S.run_leased(f.base, None, [PY, "-c", "pass"], env={}, renew_s=0.2)
            self.assertEqual(rc, 0)
            self.assertEqual(f.paths()[-1], f"/lease/{LID}/end")
            time.sleep(0.5)
            self.assertEqual(f.paths()[-1], f"/lease/{LID}/end")
        finally:
            f.close()


class Cli(unittest.TestCase):
    def setUp(self):
        self.jobs_dir = tempfile.mkdtemp()
        os.environ["STRATA_SUSPEND_JOBS_DIR"] = self.jobs_dir

    def tearDown(self):
        os.environ.pop("STRATA_SUSPEND_JOBS_DIR", None)

    def _run_cli(self, f, marker, extra=(), env_extra=None):
        env = dict(os.environ, STRATA_API_KEY="", STRATA_SUSPEND_JOBS_DIR=self.jobs_dir)
        env.pop("STRATA_SUSPEND_LEASES", None)
        env.update(env_extra or {})
        out = subprocess.run([PY, str(Path(__file__).resolve().parent / "strata_suspend.py"), "run",
                              "--port", str(f.port), *extra, "--", PY, "-c",
                              f"open(r'{marker}','w').write('x')"],
                             capture_output=True, text=True, env=env, timeout=120)
        return out

    def test_cli_default_takes_the_lease_path_when_the_server_answers(self):
        f = FakeServer(leases_status=200)
        tmp = tempfile.mkdtemp()
        marker = os.path.join(tmp, "ran.txt")
        try:
            out = self._run_cli(f, marker)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertTrue(Path(marker).exists())
            self.assertIn("/lease", f.paths())               # the lease is the only lock now
            self.assertIn(f"/lease/{LID}/end", f.paths())
            self.assertNotIn("/unload", f.paths())           # no pid-hold path
        finally:
            f.close()

    def test_cli_uses_the_lease_path_with_the_flag(self):
        f = FakeServer(leases_status=200)
        tmp = tempfile.mkdtemp()
        marker = os.path.join(tmp, "ran.txt")
        try:
            out = self._run_cli(f, marker, extra=["--lease"])
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertTrue(Path(marker).exists())
            self.assertIn("/lease", f.paths())
            self.assertIn(f"/lease/{LID}/end", f.paths())
            self.assertNotIn("/unload", f.paths())          # no pid-hold path
        finally:
            f.close()

    def test_cli_uses_the_lease_path_with_the_env_var(self):
        f = FakeServer(leases_status=200)
        tmp = tempfile.mkdtemp()
        marker = os.path.join(tmp, "ran.txt")
        try:
            out = self._run_cli(f, marker, env_extra={"STRATA_SUSPEND_LEASES": "1"})
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertTrue(Path(marker).exists())
            self.assertIn("/lease", f.paths())
            self.assertIn(f"/lease/{LID}/end", f.paths())
            self.assertNotIn("/unload", f.paths())
        finally:
            f.close()

    def test_cli_the_flag_changes_nothing_on_an_older_server(self):
        f = FakeServer(leases_status=404)
        tmp = tempfile.mkdtemp()
        marker = os.path.join(tmp, "ran.txt")
        try:
            out = self._run_cli(f, marker, extra=["--lease"])
            self.assertEqual(out.returncode, 1, out.stdout + out.stderr)   # no pid path to fall back to
            self.assertIn("lease refused", out.stdout)
            self.assertFalse(Path(marker).exists())                        # the job was not started
            self.assertIn("/lease", f.paths())
            self.assertNotIn("/unload", f.paths())
        finally:
            f.close()

    def test_cli_an_older_server_refuses_the_job(self):
        f = FakeServer(leases_status=404)
        tmp = tempfile.mkdtemp()
        marker = os.path.join(tmp, "ran.txt")
        try:
            out = self._run_cli(f, marker)
            self.assertEqual(out.returncode, 1, out.stdout + out.stderr)   # the lease is the only lock
            self.assertIn("lease refused", out.stdout)
            self.assertFalse(Path(marker).exists())                        # the job was not started
            self.assertIn("/lease", f.paths())
            self.assertNotIn("/unload", f.paths())
        finally:
            f.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
