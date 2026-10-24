"""tools/test_lease_default.py - my own tests for the lease-only lock in tools/strata_suspend.py, for what the
verdict does not cover: the stop CLI (--id / --all) posting the lease stop routes, status listing the live leases,
hold --pid keeping a lease while a process lives, release being gone, and a renewal overridden (410 without
"stopped") ending the wrapper with 75.  No Strata, no GPU: a fake lease server and short real processes.

    python tools/test_lease_default.py
"""
import contextlib
import inspect
import io
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
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_suspend as S  # noqa: E402

PY = sys.executable
LID = "cd" * 16


class FakeServer:
    """A stand-in Strata: GET /health and /leases answer; POST /lease, /lease/<id>/renew, /lease/<id>/end,
    /lease/<id>/stop and /lease/stop work on an in-memory table."""

    def __init__(self, renew_answer=None, lease_answers=None):
        self.calls = []
        self.leases = {LID: {"id": LID, "cmd": "python job.py", "cwd": "C:\\jobs", "session": None,
                             "mode": "full", "age_s": 3.0}}
        self.renew_answer = renew_answer
        self.lease_answers = list(lease_answers or [])      # POST /lease answers in order, then 200
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/health":
                    self._send(200, {"status": "ok", "loaded": False, "hibernated": True, "model": "m",
                                     "leases": fake.leases})
                elif self.path == "/leases":
                    self._send(200, {"leases": list(fake.leases.values())})
                else:
                    self._send(404, {})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                fake.calls.append((self.path, body))
                if self.path == "/lease":
                    if fake.lease_answers:
                        self._send(*fake.lease_answers.pop(0))
                    else:
                        self._send(200, {"id": LID, "status": "unloaded"})
                elif self.path.endswith("/renew"):
                    if fake.renew_answer is not None:
                        self._send(*fake.renew_answer)
                    else:
                        self._send(200, {"id": LID})
                elif self.path.endswith("/end"):
                    fake.leases.pop(self.path.split("/")[2], None)
                    self._send(200, {"id": LID, "ended": "released", "exit": body.get("exit")})
                elif self.path == "/lease/stop":
                    if body.get("all") is not True or set(body) != {"all"}:
                        self._send(400, {"error": {"message": 'stop: send {"all": true}'}})
                        return
                    stopped = list(fake.leases.values())
                    fake.leases.clear()
                    self._send(200, {"stopped": stopped})
                elif self.path.endswith("/stop"):
                    lid = self.path.split("/")[2]
                    if lid not in fake.leases:
                        self._send(404, {"error": {"message": f"no lease {lid}"}})
                        return
                    lease = fake.leases.pop(lid)
                    lease["ended"] = "stopped"
                    self._send(200, lease)
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


class Cli(unittest.TestCase):
    def setUp(self):
        self.jobs_dir = tempfile.mkdtemp()
        os.environ["STRATA_SUSPEND_JOBS_DIR"] = self.jobs_dir

    def tearDown(self):
        os.environ.pop("STRATA_SUSPEND_JOBS_DIR", None)

    def _cli(self, f, *args):
        env = dict(os.environ, STRATA_API_KEY="", STRATA_SUSPEND_JOBS_DIR=self.jobs_dir)
        return subprocess.run([PY, str(Path(__file__).resolve().parent / "strata_suspend.py"), *args,
                               "--port", str(f.port)], capture_output=True, text=True, env=env, timeout=120)

    def test_stop_one_lease_posts_the_route_and_names_the_job(self):
        f = FakeServer()
        try:
            out = self._cli(f, "stop", "--id", LID)
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            self.assertEqual(f.paths(), [f"/lease/{LID}/stop"])
            self.assertIn("python job.py", out.stdout)             # what was stopped, by its command
            self.assertIn("renewal", out.stdout)                   # the job ends at its next renewal
        finally:
            f.close()

    def test_stop_all_posts_the_route(self):
        f = FakeServer()
        try:
            out = self._cli(f, "stop", "--all")
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            self.assertEqual(f.paths(), ["/lease/stop"])
            self.assertEqual(f.calls[0][1], {"all": True})
            self.assertIn("python job.py", out.stdout)
        finally:
            f.close()

    def test_stop_needs_id_or_all(self):
        f = FakeServer()
        try:
            out = self._cli(f, "stop")
            self.assertEqual(out.returncode, 2)
            self.assertIn("--id", out.stdout)
            self.assertEqual(f.paths(), [])
        finally:
            f.close()

    def test_stop_of_an_unknown_lease_reports_the_404(self):
        f = FakeServer()
        try:
            out = self._cli(f, "stop", "--id", "ab" * 16)
            self.assertEqual(out.returncode, 1)
            self.assertIn("404", out.stdout)
        finally:
            f.close()

    def test_status_lists_the_live_leases(self):
        f = FakeServer()
        try:
            out = self._cli(f, "status")
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            self.assertIn("hibernated", out.stdout)
            self.assertIn(LID, out.stdout)
            self.assertIn("python job.py", out.stdout)
            self.assertIn("C:\\jobs", out.stdout)
        finally:
            f.close()

    def test_release_is_gone(self):
        f = FakeServer()
        try:
            out = self._cli(f, "release")
            self.assertEqual(out.returncode, 2)
            self.assertIn("gone", out.stdout)
            self.assertEqual(f.paths(), [])
        finally:
            f.close()

    def test_hold_pid_keeps_a_lease_while_the_process_lives(self):
        f = FakeServer()
        p = subprocess.Popen([PY, "-c", "import time; time.sleep(6)"])
        try:
            code = S.hold_pid(f.base, None, p.pid, renew_s=0.2)
            self.assertEqual(code, 0)
            self.assertIn("/lease", f.paths())
            self.assertIn(f"/lease/{LID}/renew", f.paths())        # renewed while the process lives
            self.assertEqual(f.paths()[-1], f"/lease/{LID}/end")   # ended when it exits
            p.wait()
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
            f.close()

    def test_a_renewal_overridden_ends_the_wrapper_with_75(self):
        f = FakeServer(renew_answer=(410, {"error": {"message": f"lease {LID} ended (overridden)"}}))
        try:
            rc = S.run_leased(f.base, None, [PY, "-c", "import time; time.sleep(30)"], env={}, renew_s=0.2)
            self.assertEqual(rc, 75)
        finally:
            f.close()

    def test_a_stopped_renewal_ends_the_job_tree_and_says_so(self):
        f = FakeServer(renew_answer=(410, {"error": {"message": f"lease {LID} ended (stopped)"}}))
        try:
            rc = S.run_leased(f.base, None, [PY, "-c", "import time; time.sleep(30)"], env={}, renew_s=0.2)
            self.assertEqual(rc, 143)
        finally:
            f.close()

    def test_launch_prints_the_log_path_the_run_writes(self):
        # `launch` prints "tail the log: <path>"; the `run` it starts writes its job log THERE (the path is passed
        # on as --log), not next to the record.  The child runs without a new console here (CREATE_NEW_CONSOLE -> 0).
        f = FakeServer()
        log = str(Path(self.jobs_dir) / "launch-run.log")
        try:
            env = dict(os.environ, STRATA_API_KEY="", STRATA_SUSPEND_JOBS_DIR=self.jobs_dir)
            with mock.patch.object(subprocess, "CREATE_NEW_CONSOLE", 0):
                out = subprocess.run([PY, str(Path(__file__).resolve().parent / "strata_suspend.py"), "launch",
                                      "--log", log, "--port", str(f.port), "--", PY, "-c", "print('job ran')"],
                                     capture_output=True, text=True, env=env, timeout=120)
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            printed = [ln.split("tail the log: ")[1].strip() for ln in out.stdout.splitlines()
                       if "tail the log: " in ln]
            self.assertEqual(printed, [log])                          # what is printed is the path given
            deadline = time.time() + 60
            while time.time() < deadline and not (Path(log).exists() and "job ran" in Path(log).read_text(
                    encoding="utf-8", errors="replace")):
                time.sleep(0.3)
            self.assertIn("job ran", Path(log).read_text(encoding="utf-8", errors="replace"))
            self.assertIn("/lease", f.paths())                        # the run made its lease as usual
        finally:
            f.close()


BUSY = (409, {"status": "busy", "error": {"message": "a request is running or queued"}})
FOUR_KB = (503, {"error": {"type": "server_error", "message": "cannot give the RAM back alone: the expert arena uses "
                                                              "4 KB pages"}})


class Wait(unittest.TestCase):
    """The --wait of the lease path: launch passes it on to its run, hold --pid waits the way run does, the
    large-pages fallback still works after a busy retry (the verdict covers the wait itself)."""

    def setUp(self):
        self.jobs_dir = tempfile.mkdtemp()
        self.addCleanup(os.environ.pop, "STRATA_SUSPEND_JOBS_DIR", None)
        os.environ["STRATA_SUSPEND_JOBS_DIR"] = self.jobs_dir

    def test_run_leased_waits_60_s_by_default(self):
        self.assertEqual(inspect.signature(S.run_leased).parameters["wait"].default, 120)

    def test_launch_passes_its_wait_on_to_the_run_it_starts(self):
        # in-process with Popen mocked: the child command of launch (its run) and of launch --detach (the launch the
        # background process runs, which starts the run) both carry --wait 7.  __file__ points into a temp folder so
        # the detached log goes there, not into the worktree's data/.
        tmp = Path(tempfile.mkdtemp())
        (tmp / "tools").mkdir()
        for extra in ([], ["--detach"]):
            argv = ["strata_suspend.py", "launch", "--wait", "7", "--port", "1", "--log", str(tmp / "job.log"),
                    *extra, "--", PY, "-c", "pass"]
            with self.subTest(extra=extra), mock.patch.object(subprocess, "Popen") as popen, \
                    mock.patch.object(S, "__file__", str(tmp / "tools" / "strata_suspend.py")), \
                    mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(S.main(), 0)
                child = popen.call_args[0][0]
                self.assertIn("--wait", child)
                self.assertEqual(child[child.index("--wait") + 1], "7", child)

    def test_hold_pid_waits_out_a_busy_answer(self):
        f = FakeServer(lease_answers=[BUSY])
        p = subprocess.Popen([PY, "-c", "import time; time.sleep(3)"])
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = S.hold_pid(f.base, None, p.pid, renew_s=0.2, wait=60)
            self.assertEqual(code, 0, out.getvalue())
            self.assertEqual(f.paths().count("/lease"), 2)          # the busy answer, then the lease
            self.assertIn("busy", out.getvalue())
            p.wait()
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
            f.close()

    def test_hold_pid_gives_up_past_its_wait_while_busy(self):
        f = FakeServer(lease_answers=[BUSY] * 50)
        p = subprocess.Popen([PY, "-c", "import time; time.sleep(6)"])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                code = S.hold_pid(f.base, None, p.pid, renew_s=0.2, wait=1)
            self.assertEqual(code, 1)
            self.assertNotIn(f"/lease/{LID}/renew", f.paths())      # no hold was made
        finally:
            p.kill()
            p.wait()
            f.close()

    def test_large_pages_fallback_still_works_after_a_busy_retry(self):
        f = FakeServer(lease_answers=[BUSY, FOUR_KB])
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                rc = S.run_leased(f.base, None, [PY, "-c", "pass"], keep_gpu=True, env={}, renew_s=0.2, wait=60)
            self.assertEqual(rc, 0, out.getvalue())
            modes = [b["mode"] for p, b in f.calls if p == "/lease"]
            self.assertEqual(modes, ["ram", "ram", "full"])          # busy, the 4 KB refusal, then full
            self.assertIn("freeing RAM and VRAM instead", out.getvalue())
            self.assertIn(f"/lease/{LID}/end", f.paths())            # the job ran and its lease ended
        finally:
            f.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
