"""tools/test_strata_jobs.py - local: the lease waker's kept helpers (tools/strata_jobs.py): the session that ran a
job, the note a session is told, the launch check, the request line, and the lease waker itself.  No GPU, no Strata,
no real opencode or claude: a fake session database, a fake opencode program and temp files.

    python tools/test_strata_jobs.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import strata_jobs as J  # noqa: E402

PY = sys.executable
LID = "ab" * 16


class Session(unittest.TestCase):
    """find_opencode_session: the opencode session whose bash call ran a wrapped job (named once, at lease creation)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.db = self.d / "opencode.db"
        con = sqlite3.connect(self.db)
        con.execute("CREATE TABLE session (id text PRIMARY KEY, directory text)")
        con.execute("CREATE TABLE part (id text PRIMARY KEY, session_id text, time_created integer, data text)")
        self.started = time.time() - 120
        rows = [("p1", "ses_right", self.started - 2, "python strata_suspend.py run -- python sim.py --days 90"),
                ("p2", "ses_other", self.started - 2, "python strata_suspend.py run -- python sim.py --days 30"),
                ("p3", "ses_old", self.started - 5000, "python strata_suspend.py run -- python sim.py --days 90")]
        for pid, sid, t, cmd in rows:
            con.execute("INSERT INTO session VALUES (?, ?)", (sid, str(self.d)))
            con.execute("INSERT INTO part VALUES (?, ?, ?, ?)", (pid, sid, int(t * 1000), json.dumps(
                {"type": "tool", "tool": "bash", "state": {"status": "running", "input": {
                    "command": cmd, "workdir": str(self.d)}}})))
        con.commit()
        con.close()

    def test_finds_the_session_that_ran_the_job(self):
        rec = {"cmd": ["python", "sim.py", "--days", "90"], "cwd": str(self.d), "started": self.started}
        self.assertEqual(J.find_opencode_session(rec, self.db), "ses_right")

    def test_another_folder_finds_nothing(self):
        rec = {"cmd": ["python", "sim.py", "--days", "90"], "cwd": str(self.d / "elsewhere"), "started": self.started}
        self.assertIsNone(J.find_opencode_session(rec, self.db))

    def test_a_missing_database_finds_nothing(self):
        rec = {"cmd": ["x"], "cwd": str(self.d), "started": self.started}
        self.assertIsNone(J.find_opencode_session(rec, self.d / "none.db"))


class Note(unittest.TestCase):
    """The note an expired lease's session is told, and the partial output it carries."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.log = self.d / "job.log"
        self.log.write_text("epoch 1 loss 0.9\nepoch 2 loss 0.7\nLAST-LINE-XYZ\n", encoding="utf-8")
        (self.d / "results-partial.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        self.rec = {"state": J.LEASE_EXPIRED, "cmd": ["python", "sim.py", "--days", "90"], "cwd": str(self.d),
                    "started": time.time() - 600, "log": str(self.log)}

    def test_the_note_names_the_command_and_the_partial_output(self):
        msg = J.wake_message(self.rec)
        self.assertTrue(msg.startswith("[Automatic note from Strata"), msg[:80])
        self.assertIn("sim.py --days 90", msg)
        self.assertIn("LAST-LINE-XYZ", msg)
        self.assertIn("results-partial.csv", msg)
        self.assertIn("lease expired", msg)

    def test_log_tail_is_the_end_of_the_log(self):
        self.assertIn("LAST-LINE-XYZ", J.log_tail(str(self.log)))
        self.assertEqual(J.log_tail(str(self.d / "none.log")), "(no output was saved)")

    def test_files_written_lists_what_the_job_left_behind(self):
        got = J.files_written(self.rec)
        self.assertTrue(any("results-partial.csv" in x for x in got), got)
        self.assertFalse(any("job.log" in x for x in got), got)         # the job's own log is not its output

    def test_register_message_says_the_request_was_in_flight(self):
        msg = J.register_message({"agent": "opencode", "id": "ses_9"})
        self.assertIn("request in flight", msg)


class Launch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)

    def test_a_program_that_fails_at_once_is_not_counted_as_told(self):
        failed = mock.Mock(**{"poll.return_value": 1})
        with mock.patch("builtins.print"):
            self.assertFalse(J._launch(lambda *a, **k: failed, ["claude.exe"], ".", self.d / "w.log"))

    def test_a_program_that_keeps_running_counts_as_told(self):
        running = mock.Mock(**{"poll.return_value": None})
        with mock.patch.object(J, "LAUNCH_CHECK_S", 0.6):
            self.assertTrue(J._launch(lambda *a, **k: running, ["claude.exe"], ".", self.d / "w.log"))

    def test_agent_env_drops_another_sessions_variables(self):
        with mock.patch.dict(os.environ, {"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "x", "CLAUDE_PID": "5",
                                          "CLAUDE_CONFIG_DIR": "keep", "PATH": os.environ.get("PATH", "")}):
            env = J.agent_env()
        self.assertNotIn("CLAUDECODE", env)
        self.assertNotIn("CLAUDE_CODE_SESSION_ID", env)
        self.assertNotIn("CLAUDE_PID", env)
        self.assertEqual(env.get("CLAUDE_CONFIG_DIR"), "keep")


class Program(unittest.TestCase):
    def test_a_bare_name_is_found_on_path(self):
        self.assertEqual(J.program([PY, "-c", "x"])[0], PY)

    def test_an_unfound_program_is_left_as_given(self):
        self.assertEqual(J.program(["no-such-program-xyz", "a"]), ["no-such-program-xyz", "a"])

    def test_empty(self):
        self.assertEqual(J.program([]), [])


class Relay(unittest.TestCase):
    def test_the_log_is_copied_to_stdout_until_the_job_ends(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log = Path(tmp.name) / "job.log"
        proc = subprocess.Popen([PY, "-c", "print('hello from the job')"], stdout=open(log, "wb"))
        import io
        buf = io.BytesIO()
        real = sys.stdout

        class Out:
            buffer = buf

        sys.stdout = Out()
        try:
            code = J.relay(proc, str(log))
        finally:
            sys.stdout = real
        proc.wait()
        self.assertEqual(code, 0)
        self.assertIn(b"hello from the job", buf.getvalue())


class KillTree(unittest.TestCase):
    def test_kill_tree_ends_a_process_and_its_children(self):
        p = subprocess.Popen([PY, "-c", "import time; time.sleep(60)"])
        J.kill_tree(p.pid)
        p.wait(timeout=10)
        self.assertIsNotNone(p.poll())


class RequestLine(unittest.TestCase):
    def test_a_request_without_a_session_names_its_client(self):
        self.assertEqual(J.describe_request({"state": "generating", "waiting_s": 0, "client": "web page"}),
                         "generating - from web page")

    def test_a_subagent_shows_its_parent_and_task(self):
        line = J.describe_request({"state": "queued", "waiting_s": 120, "session": "s", "parent": "p",
                                   "parent_title": "Nightly", "subagent": "general", "task": "Write a report"})
        self.assertIn("queued (2 min)", line)
        self.assertIn('subagent general of "Nightly"', line)
        self.assertIn("Write a report", line)


class SessionInfo(unittest.TestCase):
    def test_a_subagent_carries_its_parent_and_task(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Path(tmp.name) / "opencode.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE session (id text PRIMARY KEY, title text, parent_id text, directory text, agent text)")
        con.execute("CREATE TABLE part (id text PRIMARY KEY, session_id text, time_created integer, data text)")
        con.execute("INSERT INTO session VALUES ('parent','Nightly',NULL,'C:/w',NULL)")
        con.execute("INSERT INTO session VALUES ('child','t','parent','C:/w','general')")
        con.execute("INSERT INTO part VALUES ('p','parent',1,?)", (json.dumps(
            {"tool": "task", "state": {"input": {"description": "Write a report", "subagent_type": "general"},
             "metadata": {"sessionId": "child"}}}),))
        con.commit()
        con.close()
        info = J.session_info("child", db)
        self.assertEqual(info["parent"], "parent")
        self.assertEqual(info["parent_title"], "Nightly")
        self.assertEqual(info["task"], "Write a report")

    def test_an_unknown_session_is_empty(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.assertEqual(J.session_info("nope", Path(tmp.name) / "none.db"), {})


class Up(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


class WakeLeases(unittest.TestCase):
    """wake_leases tells exactly the sessions to_wake_from_leases lists, once, and marks through the server."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.leases = self.d / "leases"
        self.leases.mkdir()
        self.work = self.d / "work"
        self.work.mkdir()
        srv = ThreadingHTTPServer(("127.0.0.1", 0), Up)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        self.url = f"http://127.0.0.1:{srv.server_address[1]}"
        self.marked = []

    def mark(self, entry):
        self.marked.append(dict(entry))
        return True

    def write_lease(self, lid, session, woken=False, log=None, ended="expired"):
        (self.leases / f"{lid}.json").write_text(json.dumps(
            {"id": lid, "cmd": "python job.py --n 3", "cwd": str(self.work), "mode": "full", "session": session,
             "created": time.time() - 120, "renewed": time.time() - 100, "ended": ended, "woken": woken, "log": log}),
            encoding="utf-8")

    def test_expired_lease_is_told_once_with_its_log(self):
        log = self.work / "job.log"
        log.write_text("step 1\nLAST-LINE-XYZ\n", encoding="utf-8")
        self.write_lease(LID, {"agent": "opencode", "id": "ses_1"}, log=str(log))
        self.write_lease("b2" * 16, {"agent": "opencode", "id": "ses_2"}, woken=True)
        calls = []
        told = J.wake_leases(self.leases, self.mark, True, url=self.url, popen=lambda a, **k: calls.append(list(a)),
                             opencode=["oc.exe"], claude=["claude.exe"])
        self.assertEqual(len(calls), 1, calls)
        self.assertEqual(calls[0][:7], ["oc.exe", "run", "--attach", self.url, "--session", "ses_1", "--dir"])
        self.assertIn("LAST-LINE-XYZ", calls[0][8])
        self.assertEqual([m["id"] for m in self.marked], ["ses_1"])
        self.assertEqual([t["id"] for t in told], ["ses_1"])

    def test_a_failed_launch_marks_nothing(self):
        self.write_lease("e5" * 16, {"agent": "opencode", "id": "ses_5"})
        proc = mock.Mock(**{"poll.return_value": 1})
        with mock.patch("builtins.print"):
            told = J.wake_leases(self.leases, self.mark, True, url=self.url, popen=lambda *a, **k: proc,
                                 opencode=["oc.exe"])
        self.assertEqual((told, self.marked), ([], []))

    def test_register_sessions_only_after_an_unclean_shutdown(self):
        (self.leases / "checkpoint.prev.json").write_text(json.dumps(
            {"time": time.time(), "leases": [], "register": {"opencode": {"ses_9": "a request"}, "claude": {}},
             "files": []}), encoding="utf-8")
        calls = []
        popen = lambda a, **k: calls.append(list(a))  # noqa: E731
        self.assertEqual(J.wake_leases(self.leases, self.mark, True, url=self.url, popen=popen, opencode=["oc.exe"]),
                         [])
        told = J.wake_leases(self.leases, self.mark, False, url=self.url, popen=popen, opencode=["oc.exe"])
        self.assertEqual(len(calls), 1, calls)
        self.assertEqual(calls[0][:6], ["oc.exe", "run", "--attach", self.url, "--session", "ses_9"])
        self.assertEqual([t["id"] for t in told], ["ses_9"])


class Reaper(unittest.TestCase):
    """run_leased on POSIX: a reaper in its own session kills the job's process group when the wrapper's pipe
    closes, so a killed wrapper takes its job with it (Windows uses the kill-on-close job object instead)."""

    def test_the_reaper_source_compiles(self):
        import strata_suspend as S
        compile(S._REAPER_SRC, "<reaper>", "exec")

    @unittest.skipIf(os.name == "nt", "POSIX only: the kill-on-close job object covers Windows")
    def test_a_killed_wrapper_takes_its_job_with_it(self):
        import signal
        import strata_suspend as S

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self._send({"leases": []})

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self._send({"id": LID, "status": "unloaded"})

            def _send(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.addCleanup(srv.shutdown)
        self.addCleanup(srv.server_close)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        tmp = Path(tempfile.mkdtemp())
        pidf = tmp / "pids.txt"
        job = ("import os, subprocess, sys, time\n"
               "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
               f"open(r'{pidf}', 'w').write(f'{{os.getpid()}} {{c.pid}}')\n"
               "time.sleep(120)\n")
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import strata_suspend as S\n"
                f"S.run_leased('http://127.0.0.1:{srv.server_address[1]}', None, [sys.executable, '-c', {job!r}],"
                f" cwd=r'{tmp}', renew_s=0.5, env={{}})\n")
        w = subprocess.Popen([PY, "-c", code, str(HERE)], cwd=str(HERE.parent))
        self.addCleanup(lambda: [w.kill()] if w.poll() is None else None)

        def alive(pid):
            try:
                with open(f"/proc/{pid}/stat") as f:
                    return f.read().split(") ")[-1].split()[0] != "Z"
            except OSError:
                return False

        deadline = time.time() + 30
        while time.time() < deadline and not (pidf.exists() and pidf.read_text().strip()):
            time.sleep(0.1)
        pids = [int(x) for x in pidf.read_text().split()]
        self.assertTrue(all(alive(p) for p in pids))
        w.send_signal(signal.SIGKILL)
        w.wait()
        deadline = time.time() + 10
        while time.time() < deadline and any(alive(p) for p in pids):
            time.sleep(0.2)
        self.assertEqual([p for p in pids if alive(p)], [], "the job outlived its wrapper")


if __name__ == "__main__":
    unittest.main(verbosity=2)
