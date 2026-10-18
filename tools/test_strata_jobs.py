"""tools/test_strata_jobs.py - local: the record per wrapped job (tools/strata_jobs.py) and waking the opencode session
a job was cut off from.  No GPU, no Strata, no opencode: a fake session database and a fake opencode program.

    python tools/test_strata_jobs.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import strata_jobs as J  # noqa: E402

PY = sys.executable
SUSPEND = str(HERE / "strata_suspend.py")


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Records(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.jobs = Path(self.tmp.name) / "jobs"
        self.proj = Path(self.tmp.name) / "proj"
        self.proj.mkdir()
        self.env = {**os.environ, "STRATA_SUSPEND_JOBS_DIR": str(self.jobs)}
        self.env.pop("CLAUDE_CODE_SESSION_ID", None)
        patcher = mock.patch.object(J, "JOBS_DIR", self.jobs)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        for _ in range(20):
            try:
                self.tmp.cleanup()
                return
            except PermissionError:
                time.sleep(0.25)

    def run_wrapped(self, code: str, **kw):
        """strata_suspend.py run on a port where nothing answers: the job runs as is, still recorded."""
        return subprocess.Popen([PY, SUSPEND, "run", "--port", str(free_port()), "--", PY, "-c", code],
                                cwd=self.proj, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, **kw)

    def only_record(self) -> tuple[Path, dict]:
        recs = J.records()
        self.assertEqual(len(recs), 1, recs)
        return recs[0]

    def test_a_finished_job_is_recorded_and_its_output_relayed(self):
        p = self.run_wrapped("print('hello from the job'); raise SystemExit(3)")
        out = p.communicate(timeout=60)[0]
        self.assertEqual(p.returncode, 3)
        self.assertIn("hello from the job", out)                      # relayed to the agent from the log
        f, rec = self.only_record()
        self.assertEqual((rec["state"], rec["exit"], rec.get("delivered")), ("done", 3, True))
        self.assertEqual(J.state(rec), "done")
        self.assertIn("hello from the job", Path(rec["log"]).read_text(encoding="utf-8"))
        self.assertEqual(os.path.normcase(rec["cwd"]), os.path.normcase(str(self.proj)))

    def test_a_relative_program_with_forward_slashes_starts(self):
        # agents write `.venv/Scripts/python.exe ...`; Windows' CreateProcess does not find that form by itself
        rel = os.path.relpath(PY, self.proj).replace("\\", "/")
        p = subprocess.Popen([PY, SUSPEND, "run", "--port", str(free_port()), "--", rel, "-c", "print('rel ok')"],
                             cwd=self.proj, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = p.communicate(timeout=60)[0]
        self.assertEqual(p.returncode, 0, out)
        self.assertIn("rel ok", out)
        self.assertEqual(J.program(["no-such-program-xyz", "a"]), ["no-such-program-xyz", "a"])   # unfound: as given

    def test_a_killed_wrapper_leaves_the_job_running_and_its_exit_recorded(self):
        p = self.run_wrapped("import time; print('started', flush=True); time.sleep(10); print('finished')")
        deadline = time.time() + 30
        while time.time() < deadline and not (J.records() and J.records()[0][1].get("job")):
            time.sleep(0.1)
        time.sleep(1.5)
        f, rec = self.only_record()
        # TerminateProcess on the wrapper itself (the record's pid: under a venv, p is only python.exe's launcher)
        subprocess.run(["taskkill", "/PID", str(rec["wrapper"]["pid"]), "/F"], capture_output=True)
        p.kill()
        p.wait()
        time.sleep(0.5)
        rec = J.read(f)
        self.assertIn(J.state(rec), ("running (wrapper gone)", J.UNDELIVERED))
        deadline = time.time() + 30
        while time.time() < deadline and J.read(f).get("state") != "done":
            time.sleep(0.2)
        rec = J.read(f)
        self.assertEqual((rec["state"], rec["exit"]), ("done", 0))      # the shim recorded it
        self.assertEqual(J.state(rec), J.UNDELIVERED)
        self.assertIn("finished", Path(rec["log"]).read_text(encoding="utf-8"))
        self.assertIn("never reached you", J.cause(rec))

    def test_the_cause_says_when_the_whole_pc_restarted(self):
        rec = {"state": "running", "started": time.time() - 3600, "wrapper": {"pid": 999999, "started": 1.0},
               "job": [{"pid": 999998, "started": 1.0}]}
        self.assertEqual(J.state(rec), J.CUT_OFF)
        with mock.patch.object(J, "boot_time", return_value=time.time() - 60):
            self.assertIn("system-wide", J.cause(rec))
            self.assertIn("not caused by the job's code", J.cause(rec))
        with mock.patch.object(J, "boot_time", return_value=time.time() - 86400):
            self.assertIn("ended from outside", J.cause(rec))

    def test_the_next_job_in_the_folder_notes_a_cut_off_one_once(self):
        self.jobs.mkdir()
        (self.jobs / "1-1.json").write_text(json.dumps({
            "state": "running", "cmd": ["python", "train.py"], "cwd": str(self.proj), "started": time.time() - 99,
            "wrapper": {"pid": 999999, "started": 1.0}, "job": [], "log": str(self.jobs / "1-1.log")}),
            encoding="utf-8")
        first = self.run_wrapped("pass").communicate(timeout=60)[0]
        self.assertIn("did not return its result (cut off", first)
        self.assertIn("train.py", first)
        second = self.run_wrapped("pass").communicate(timeout=60)[0]
        self.assertNotIn("did not return", second)


class Wake(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.jobs = Path(self.tmp.name) / "jobs"
        self.jobs.mkdir()
        self.proj = Path(self.tmp.name) / "proj"
        self.proj.mkdir()
        patcher = mock.patch.object(J, "JOBS_DIR", self.jobs)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.started = time.time() - 120
        self.db = Path(self.tmp.name) / "opencode.db"
        con = sqlite3.connect(self.db)
        con.execute("CREATE TABLE session (id text PRIMARY KEY, directory text NOT NULL)")
        con.execute("CREATE TABLE part (id text PRIMARY KEY, message_id text, session_id text, time_created integer, "
                    "time_updated integer, data text)")
        con.execute("INSERT INTO session VALUES ('ses_right', ?), ('ses_other', ?)", (str(self.proj), str(self.proj)))
        wrapper = ('"C:/x/.venv/Scripts/python.exe" "C:/x/tools/strata_suspend.py" run -- '
                   '"C:/x/.venv/Scripts/python.exe" sim.py --days 90')
        rows = [("p1", "ses_right", self.started - 2, wrapper, str(self.proj)),               # the one
                ("p2", "ses_other", self.started - 2, wrapper.replace("90", "30"), str(self.proj)),   # other args
                ("p3", "ses_other", self.started - 5000, wrapper, str(self.proj)),            # long before
                ("p4", "ses_other", self.started - 1, "ls -la", str(self.proj))]              # not a wrapped job
        for pid, sid, t, command, wd in rows:
            con.execute("INSERT INTO part VALUES (?, 'm', ?, ?, ?, ?)", (pid, sid, int(t * 1000), int(t * 1000),
                        json.dumps({"type": "tool", "tool": "bash", "state": {"status": "running", "input": {
                            "command": command, "workdir": wd}}})))
        con.commit()
        con.close()
        self.log = self.jobs / "1-1.log"
        self.log.write_text("epoch 1 loss 0.9\nepoch 2 loss 0.7\n", encoding="utf-8")
        self.rec = {"state": "running", "cmd": ["C:/x/.venv/Scripts/python.exe", "sim.py", "--days", "90"],
                    "cwd": str(self.proj), "started": self.started, "wrapper": {"pid": 999999, "started": 1.0},
                    "job": [{"pid": 999998, "started": 1.0}], "log": str(self.log)}
        (self.jobs / "1-1.json").write_text(json.dumps(self.rec), encoding="utf-8")
        (self.proj / "results-partial.csv").write_text("a,b\n1,2\n", encoding="utf-8")   # written after the start

    def test_finds_the_session_that_ran_the_job(self):
        self.assertEqual(J.find_opencode_session(self.rec, self.db), "ses_right")
        self.assertIsNone(J.find_opencode_session({**self.rec, "cwd": str(self.jobs)}, self.db))   # other folder

    def test_the_note_says_what_happened_and_what_is_on_disk(self):
        with mock.patch.object(J, "boot_time", return_value=time.time() - 30):
            msg = J.wake_message(self.rec)
        self.assertIn("sim.py --days 90", msg)
        self.assertIn("system-wide", msg)
        self.assertIn("epoch 2 loss 0.7", msg)                         # the partial output that reached the disk
        self.assertIn("results-partial.csv", msg)                      # the files it wrote
        self.assertIn("run it again", msg)

    def test_wake_sends_each_cut_off_session_its_note_once(self):
        calls = []
        with mock.patch.object(J, "opencode_up", return_value=True), \
                mock.patch.object(J, "opencode_exe", return_value=["opencode.exe"]):
            self.assertEqual(J.wake("http://127.0.0.1:4096", self.db, popen=lambda a, **k: calls.append(a)), 1)
            self.assertEqual(J.wake("http://127.0.0.1:4096", self.db, popen=lambda a, **k: calls.append(a)), 0)
        self.assertEqual(len(calls), 1)
        args = calls[0]
        self.assertEqual(args[:8], ["opencode.exe", "run", "--attach", "http://127.0.0.1:4096", "--session",
                                    "ses_right", "--dir", str(self.proj)])
        self.assertIn("epoch 2 loss 0.7", args[8])
        rec = J.read(self.jobs / "1-1.json")
        self.assertEqual(rec["session"], "ses_right")
        self.assertTrue(rec["woken"])

    def test_a_job_run_again_later_in_its_folder_wakes_nobody(self):
        later = {**self.rec, "state": "done", "exit": 0, "delivered": True, "started": self.started + 60,
                 "wrapper": {"pid": 999997, "started": 2.0}}
        (self.jobs / "2-2.json").write_text(json.dumps(later), encoding="utf-8")
        calls = []
        with mock.patch.object(J, "opencode_up", return_value=True), \
                mock.patch.object(J, "opencode_exe", return_value=["opencode.exe"]):
            self.assertEqual(J.wake("http://127.0.0.1:4096", self.db, popen=lambda a, **k: calls.append(a)), 0)
        self.assertEqual(calls, [])
        self.assertIsNotNone(J.run_again(self.rec))
        self.assertIsNone(J.run_again({**self.rec, "cmd": ["other.exe"]}))      # another command: not a rerun

    def test_a_dismissed_job_wakes_nobody_and_says_so(self):
        self.assertEqual(J.dismiss(["1-1.json", "nope-1.json"], "the agent moved on"), 1)
        rec = J.read(self.jobs / "1-1.json")
        self.assertEqual(rec["dismissed_why"], "the agent moved on")
        self.assertIn("dismissed", J.not_needed(rec))
        calls = []
        with mock.patch.object(J, "opencode_up", return_value=True), \
                mock.patch.object(J, "opencode_exe", return_value=["opencode.exe"]):
            self.assertEqual(J.wake("http://127.0.0.1:4096", self.db, popen=lambda a, **k: calls.append(a)), 0)
        self.assertEqual(calls, [])
        done = {**self.rec, "state": "done", "exit": 0, "delivered": True}
        (self.jobs / "3-3.json").write_text(json.dumps(done), encoding="utf-8")
        self.assertEqual(J.dismiss(["3-3"]), 0)                       # not cut off: left alone
        self.assertNotIn("dismissed", J.read(self.jobs / "3-3.json"))

    def test_no_opencode_server_sends_nothing_and_tries_again_next_time(self):
        calls = []
        with mock.patch.object(J, "opencode_up", return_value=False):
            self.assertEqual(J.wake("http://127.0.0.1:4096", self.db, popen=lambda a, **k: calls.append(a)), 0)
        self.assertEqual(calls, [])
        self.assertNotIn("woken", J.read(self.jobs / "1-1.json"))

    def transcript(self, sid: str, entries: list) -> Path:
        projects = Path(self.tmp.name) / "projects"
        d = projects / "C--proj"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{sid}.jsonl").write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
        return projects

    def claude_entries(self, tool: str, text: str):
        from datetime import datetime, timezone
        ts = datetime.fromtimestamp(self.started - 3, timezone.utc).isoformat().replace("+00:00", "Z")
        return [{"type": "user", "cwd": "C:/the/claude/project", "sessionId": "cl-1", "message": {"content": "go"}},
                {"type": "assistant", "timestamp": ts, "message": {"content": [
                    {"type": "tool_use", "name": tool, "input": {"command": text} if tool != "Write" else
                     {"file_path": "x.py", "content": text}}]}}]

    def wake_with(self, projects):
        calls = []
        with mock.patch.object(J, "opencode_up", return_value=True), \
                mock.patch.object(J, "opencode_exe", return_value=["opencode.exe"]), \
                mock.patch.object(J, "claude_exe", return_value=["claude.exe"]):
            n = J.wake("http://127.0.0.1:4096", self.db, projects=projects,
                       popen=lambda a, **k: calls.append((a, k.get("cwd"))))
        return n, calls

    def test_an_opencode_match_wins_over_an_inherited_claude_id(self):
        # an opencode run started from a Claude shell inherits CLAUDE_CODE_SESSION_ID: the opencode session is told
        (self.jobs / "1-1.json").write_text(json.dumps({**self.rec, "claude_session": "cl-1"}), encoding="utf-8")
        projects = self.transcript("cl-1", self.claude_entries("Bash", "strata_suspend.py run -- python sim.py "
                                                                        "--days 90"))
        n, calls = self.wake_with(projects)
        self.assertEqual(n, 1)
        self.assertEqual(calls[0][0][0], "opencode.exe")

    def test_a_claude_code_session_that_ran_the_job_is_resumed(self):
        rec = {**self.rec, "cmd": ["python", "train.py", "--epochs", "5"], "claude_session": "cl-1"}
        (self.jobs / "1-1.json").write_text(json.dumps(rec), encoding="utf-8")
        cmd = '"C:/x/.venv/Scripts/python.exe" "C:/x/tools/strata_suspend.py" run -- python train.py --epochs 5'
        n, calls = self.wake_with(self.transcript("cl-1", self.claude_entries("Bash", cmd)))
        self.assertEqual(n, 1)
        args, cwd = calls[0]
        self.assertEqual(args[:4], ["claude.exe", "--resume", "cl-1", "-p"])
        self.assertIn("train.py --epochs 5", args[4])
        self.assertEqual(cwd, "C:/the/claude/project")              # resumed from the session's own folder
        self.assertEqual(J.read(self.jobs / "1-1.json")["agent"], "claude")

    def test_a_claude_session_that_only_wrote_about_the_job_is_not_woken(self):
        rec = {**self.rec, "cmd": ["python", "train.py"], "claude_session": "cl-1"}
        (self.jobs / "1-1.json").write_text(json.dumps(rec), encoding="utf-8")
        n, calls = self.wake_with(self.transcript("cl-1", self.claude_entries(
            "Write", "subprocess.run(['strata_suspend.py', 'run', '--', 'python', 'train.py'])")))
        self.assertEqual((n, calls), (0, []))

    def test_unknown_sessions_are_tried_a_few_times_then_left(self):
        (self.jobs / "1-1.json").write_text(json.dumps({**self.rec, "cmd": ["python", "other.py"]}), encoding="utf-8")
        for _ in range(J.WAKE_TRIES + 1):
            n, calls = self.wake_with(Path(self.tmp.name) / "none")
            self.assertEqual(calls, [])
        self.assertEqual(J.read(self.jobs / "1-1.json")["wake_tries"], J.WAKE_TRIES)

    def test_the_server_keeps_a_heartbeat_and_passes_the_last_one_on(self):
        sys.path.insert(0, str(HERE.parent))
        from serve import server
        beat = Path(self.tmp.name) / "alive.json"
        beat.write_text(json.dumps({"alive": 1234.5}), encoding="utf-8")
        with mock.patch.object(server, "HEARTBEAT", beat):
            self.assertEqual(server.start_heartbeat(0.2), 1234.5)        # the previous run's last beat
            time.sleep(0.6)
            self.assertGreater(json.loads(beat.read_text(encoding="utf-8"))["alive"], time.time() - 5)
        calls = []
        with mock.patch("builtins.print"):
            server.spawn_job_wake(0.5, popen=lambda a, **k: calls.append(a), last_alive=1234.5,
                                  log=Path(self.tmp.name) / "wake.log")   # never the real log
        self.assertEqual(calls[0][-2:], ["--strata-last-alive", "1234.5"])

    def test_the_server_starts_wake_only_when_asked(self):
        sys.path.insert(0, str(HERE.parent))
        from serve import server
        calls = []
        with mock.patch("builtins.print"):
            server.spawn_job_wake(0.5, popen=lambda a, **k: calls.append(a), log=Path(self.tmp.name) / "wake.log")
        self.assertEqual(calls[0][1:], [str(HERE / "strata_suspend.py"), "wake", "--after", "0.5"])


class Interrupted(unittest.TestCase):
    """Turns cut off without a wrapped job: mid-turn at a PC restart, or a request that failed while Strata was down."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.jobs = root / "jobs"
        patcher = mock.patch.object(J, "JOBS_DIR", self.jobs)
        patcher.start()
        self.addCleanup(patcher.stop)
        live = mock.patch.object(J, "LIVE_PREV", root / "no-register.json")   # never the real Strata's register
        live.start()
        self.addCleanup(live.stop)
        self.now = time.time()
        self.boot = self.now - 100
        self.db = root / "opencode.db"
        con = sqlite3.connect(self.db)
        con.execute("CREATE TABLE session (id text PRIMARY KEY, directory text, time_updated integer, "
                    "time_archived integer, parent_id text)")
        con.execute("CREATE TABLE message (id text PRIMARY KEY, session_id text, time_created integer, data text)")
        con.execute("CREATE TABLE part (id text PRIMARY KEY, message_id text, session_id text, time_created integer, "
                    "time_updated integer, data text)")
        b = self.boot
        unable = {"name": "APIError", "data": {"message": "Cannot connect to API: Unable to connect.",
                                               "metadata": {"url": "http://127.0.0.1:8080/v1/chat/completions"}}}
        sessions = [  # id, updated, last message, archived, parent, a part of the last message
            ("mid", b - 300, {"role": "assistant", "time": {"created": 1}}, None, None, None),
            ("finished", b - 300, {"role": "assistant", "time": {"created": 1, "completed": 2}}, None, None, None),
            ("long-before", b - 20000, {"role": "assistant", "time": {"created": 1}}, None, None, None),
            ("asked", b - 200, {"role": "user", "time": {"created": 1}}, None, None, None),
            ("tool", b - 200, {"role": "assistant", "time": {"created": 1, "completed": 2}}, None, None,
             {"type": "tool", "tool": "bash", "state": {"status": "running"}}),
            ("child", b - 300, {"role": "assistant", "time": {"created": 1}}, None, "mid", None),
            ("step-tools", b - 250, {"role": "assistant", "finish": "tool-calls", "time": {"created": 1,
                                                                                         "completed": 2}}, None, None,
             {"type": "tool", "tool": "bash", "state": {"status": "completed"}}),   # its tools ran, the next step not
            ("asking", b - 250, {"role": "assistant", "time": {"created": 1}}, None, None,
             {"type": "tool", "tool": "question", "state": {"status": "running"}}),  # waiting for the user's answer
            ("archived", b - 300, {"role": "assistant", "time": {"created": 1}}, 5, None, None),
            ("down", self.now - 600, {"role": "assistant", "error": unable,
                                      "time": {"created": 1, "completed": int((self.now - 600) * 1000)}}, None, None,
             None),
            ("aborted", self.now - 600, {"role": "assistant", "error": {"name": "MessageAbortedError"},
                                         "time": {"created": 1, "completed": int((self.now - 600) * 1000)}}, None,
             None, None),
            ("down-long-ago", self.now - 9000, {"role": "assistant", "error": unable,
                                                "time": {"created": 1, "completed": int((self.now - 9000) * 1000)}},
             None, None, None),
        ]
        for sid, updated, msg, archived, parent, part in sessions:
            con.execute("INSERT INTO session VALUES (?, ?, ?, ?, ?)", (sid, f"C:/proj/{sid}", int(updated * 1000),
                                                                       archived, parent))
            con.execute("INSERT INTO message VALUES (?, ?, ?, ?)", (f"m-{sid}", sid, 10, json.dumps(msg)))
            if part:
                con.execute("INSERT INTO part VALUES (?, ?, ?, 10, 10, ?)", (f"p-{sid}", f"m-{sid}", sid,
                                                                           json.dumps(part)))
        con.commit()
        con.close()
        self.projects = root / "projects"
        d = self.projects / "C--proj"
        d.mkdir(parents=True)
        user = {"type": "user", "cwd": "C:/claude/proj", "message": {"content": "go on"}}
        tool = {"type": "assistant", "message": {"stop_reason": "tool_use", "content": [{"type": "tool_use",
                                                                                         "name": "Bash"}]}}
        done = {"type": "assistant", "message": {"stop_reason": "end_turn", "content": [{"type": "text",
                                                                                         "text": "Done."}]}}
        ask = {"type": "assistant", "message": {"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "name": "AskUserQuestion"}]}}
        for name, entries, t in (("cl-tool", [user, tool], b - 200), ("cl-done", [user, done], b - 200),
                                 ("cl-ask", [user, ask], b - 200),
                                 ("cl-asked", [done, user], b - 200), ("cl-old", [user, tool], b - 20000),
                                 ("cl-after", [user, tool], b + 50)):
            f = d / f"{name}.jsonl"
            f.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
            os.utime(f, (t, t))

    def down(self):
        return J.went_down_at(self.boot, self.db, self.projects)

    def test_finds_exactly_the_sessions_cut_off(self):
        self.assertAlmostEqual(self.down(), self.now - 300, delta=1)   # the last write before the boot
        oc = {(x["session"], x["why"]) for x in J.opencode_interrupted(self.db, self.down(), None, self.now)}
        self.assertEqual(oc, {("mid", "restart"), ("asked", "restart"), ("tool", "restart"), ("step-tools", "restart"),
                              ("down", "strata-down")})                # not "asking": it waited for the user
        cl = {x["session"] for x in J.claude_interrupted(self.projects, self.down())}
        self.assertEqual(cl, {"cl-tool", "cl-asked"})                   # not "cl-ask" (AskUserQuestion)

    def test_the_previous_boot_bounds_it_not_a_window(self):
        # a session last active 5 h before the PC went down (a long command) was cut off by it, if that was after the
        # previous boot; one active before the previous boot was not
        down = self.down()
        found = {x["session"] for x in J.opencode_interrupted(self.db, down, None, self.now, since_boot=down - 30000)}
        self.assertIn("long-before", found)                             # 20000 s before: past any fixed window
        found = {x["session"] for x in J.opencode_interrupted(self.db, down, None, self.now, since_boot=down - 600)}
        self.assertNotIn("long-before", found)
        self.assertIn("cl-old", {x["session"] for x in J.claude_interrupted(self.projects, down, down - 30000)})
        self.assertNotIn("cl-old", {x["session"] for x in J.claude_interrupted(self.projects, down, down - 600)})

    def test_no_restart_means_only_strata_down_requests(self):
        oc = {x["session"] for x in J.opencode_interrupted(self.db, None, None, self.now)}
        self.assertEqual(oc, {"down"})
        self.assertEqual(J.claude_interrupted(self.projects, None), [])

    def test_a_power_cut_of_days_is_still_found(self):
        # Windows starts three days after the power went: the sessions mid-turn then are still told
        later = self.now + 3 * 86400
        down = J.went_down_at(later, self.db, self.projects)
        self.assertAlmostEqual(down, self.now - 50, delta=1)            # the last write (cl-after), not the boot
        oc = {x["session"] for x in J.opencode_interrupted(self.db, down, None, later) if x["why"] == "restart"}
        self.assertEqual(oc, {"mid", "asked", "tool", "step-tools"})
        self.assertEqual({x["session"] for x in J.claude_interrupted(self.projects, down)},
                         {"cl-tool", "cl-asked", "cl-after"})

    def test_a_request_that_failed_while_strata_was_down_counts_however_long_ago(self):
        later = self.now + 3 * 86400                                    # Strata started again three days later
        alive = self.now - 1000                                         # its last heartbeat before it stopped
        got = {x["session"] for x in J.opencode_interrupted(self.db, None, alive, later)}
        self.assertEqual(got, {"down"})                                 # failed after the heartbeat (now - 600)
        got = {x["session"] for x in J.opencode_interrupted(self.db, None, self.now - 400, later)}
        self.assertEqual(got, set())                                    # failed before Strata stopped: not this

    def test_each_is_told_once_with_the_right_cause(self):
        calls = []

        def run():
            with mock.patch.object(J, "opencode_up", return_value=True), \
                    mock.patch.object(J, "opencode_exe", return_value=["opencode.exe"]), \
                    mock.patch.object(J, "claude_exe", return_value=["claude.exe"]):
                return J.wake_interrupted("http://127.0.0.1:4096", self.db, self.projects, skip={"asked"},
                                          popen=lambda a, **k: calls.append((a, k.get("cwd"))), boot=self.boot,
                                          now=self.now, since_boot=self.boot - 10000)
        self.assertEqual(run(), 6)          # mid, tool, step-tools, down, cl-tool, cl-asked ("asked": skipped)
        self.assertEqual(run(), 0)                                  # once
        by = {a[a.index("--session") + 1] if "--session" in a else a[2]: (a, cwd) for a, cwd in calls}
        self.assertEqual(set(by), {"mid", "tool", "step-tools", "down", "cl-tool", "cl-asked"})
        self.assertIn("the PC was restarted or went down", by["mid"][0][-1])
        self.assertIn("not caused by anything you did", by["mid"][0][-1])
        self.assertIn("was not running when your request reached it", by["down"][0][-1])
        self.assertEqual(by["cl-tool"][0][:4], ["claude.exe", "--resume", "cl-tool", "-p"])
        self.assertEqual(by["cl-tool"][1], "C:/claude/proj")


class Live(unittest.TestCase):
    """The live register: who was working when the PC went down - recorded while it happened, not inferred."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.now = time.time()
        self.db = root / "opencode.db"
        con = sqlite3.connect(self.db)
        con.execute("CREATE TABLE part (id text PRIMARY KEY, message_id text, session_id text, time_created integer, "
                    "time_updated integer, data text)")
        for pid, sid, cmd in (("p1", "ses_cmd", 'pytest -x "tests/test_big.py"'), ("p2", "ses_dead", "make all")):
            con.execute("INSERT INTO part VALUES (?, 'm', ?, ?, ?, ?)", (pid, sid, int(self.now * 1000),
                        int(self.now * 1000), json.dumps({"type": "tool", "tool": "bash", "state": {
                            "status": "running", "input": {"command": cmd}}})))
        con.commit()
        con.close()
        self.projects = root / "projects"
        (self.projects / "C--p").mkdir(parents=True)
        fresh, stale = self.projects / "C--p" / "cl-fresh.jsonl", self.projects / "C--p" / "cl-stale.jsonl"
        fresh.write_text("{}\n", encoding="utf-8")
        stale.write_text("{}\n", encoding="utf-8")
        os.utime(stale, (self.now - 3600, self.now - 3600))

    def test_the_register_names_who_is_working_and_why(self):
        opencode_child = mock.Mock(**{"cmdline.return_value": ["bash", "-c", "pytest -x tests/test_big.py"]})
        claude_child = mock.Mock(**{"environ.return_value": {"CLAUDE_CODE_SESSION_ID": "cl-cmd"},
                                    "cmdline.return_value": ["bash.exe", "-c", "source ~/.claude/shell-snapshots/x"]})
        mcp_server = mock.Mock(**{"environ.return_value": {"CLAUDE_CODE_SESSION_ID": "cl-idle"},
                                  "cmdline.return_value": ["cmd", "/c", "npx", "-y", "some-mcp"]})
        with mock.patch.object(J, "_descendants", side_effect=lambda names: [opencode_child] if "opencode.exe" in
                               names else [claude_child, mcp_server]):
            live = J.live_register({"ses_req": {"state": "waiting out a hold"}}, self.db, self.projects, self.now)
        self.assertEqual(live["opencode"], {"ses_req": "a request in Strata (waiting out a hold)",
                                            "ses_cmd": "a command running"})   # not ses_dead: no process runs it
        self.assertEqual(live["claude"], {"cl-cmd": "a command running", "cl-fresh": "active in the last 10 minutes"})

    def test_the_heartbeat_writes_the_register_and_keeps_the_last_one_aside(self):
        sys.path.insert(0, str(HERE.parent))
        from serve import server
        beat = Path(self.tmp.name) / "strata-alive.json"
        beat.write_text(json.dumps({"alive": 1234.5, "live": {"time": 1234.0, "opencode": {"s1": "x"}}}),
                        encoding="utf-8")
        with mock.patch.object(server, "HEARTBEAT", beat):
            self.assertEqual(server.start_heartbeat(0.2, register=lambda: {"time": 1.0, "opencode": {"s2": "y"}}),
                             1234.5)
            time.sleep(0.6)
        self.assertEqual(J.read_live(beat.with_name("strata-alive.prev.json"))["opencode"], {"s1": "x"})
        self.assertEqual(json.loads(beat.read_text(encoding="utf-8"))["live"]["opencode"], {"s2": "y"})
        with mock.patch.object(server, "HEARTBEAT", beat):
            server.last_beat(shutting_down=True)                       # the close handler's final entry
        final = json.loads(beat.read_text(encoding="utf-8"))
        self.assertTrue(final["shutting_down"] and final["live"]["shutting_down"])
        # Windows ended an agent's command before the close handler ran: the last beat's sessions stay in
        beat.write_text(json.dumps({"alive": 1.0, "live": {"time": 1.0, "opencode": {"s-cmd": "a command running"},
                                                           "claude": {"c1": "a command running"}}}), encoding="utf-8")
        with mock.patch.object(server, "HEARTBEAT", beat):
            server.last_beat(shutting_down=True)
        final = json.loads(beat.read_text(encoding="utf-8"))["live"]
        self.assertEqual(final["opencode"], {"s-cmd": "a command running", "s2": "y"})
        self.assertEqual(final["claude"], {"c1": "a command running"})


class Holders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.jobs = Path(self.tmp.name) / "jobs"
        self.jobs.mkdir()
        patcher = mock.patch.object(J, "JOBS_DIR", self.jobs)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.p = subprocess.Popen([PY, "-c", "import time; time.sleep(60)"])
        self.addCleanup(lambda: (self.p.kill(), self.p.wait()))
        self.now = time.time()

    def record(self, name, started, running=True, pids=True):
        rec = {"state": "running" if running else "done", "cmd": ["python", "train.py"], "cwd": "C:/proj",
               "started": started, "wrapper": {"pid": 999999, "started": 1.0},
               "job": [{"pid": self.p.pid, "started": J.proc_started(self.p.pid)}] if pids else []}
        (self.jobs / name).write_text(json.dumps(rec), encoding="utf-8")

    def test_a_holder_is_named_and_a_restart_loop_is_seen(self):
        for i, ago in enumerate((1500, 900, 300)):                     # the same job, three times in 25 min
            self.record(f"{i}-1.json", self.now - ago, running=(i == 2), pids=(i == 2))
        with mock.patch.object(J, "find_opencode_session", return_value="ses_x"), \
                mock.patch.object(J, "_session_title", return_value="nightly training"):
            hs = J.describe_holders([self.p.pid, 424242], db=Path(self.tmp.name) / "none.db")
        job, other = hs
        self.assertEqual((job["command"], job["folder"], job["title"], job["starts"], job["loop"]),
                         ("python train.py", "C:/proj", "nightly training", 3, True))
        self.assertIn("stop that session first", J.advice(job))
        self.assertFalse(other["wrapped"])                              # held with `hold --pid`: no record

    def test_stop_ends_the_job_marks_it_and_leaves_others_alone(self):
        self.record("1-1.json", self.now - 60)
        other = subprocess.Popen([PY, "-c", "import time; time.sleep(60)"])
        self.addCleanup(lambda: (other.kill(), other.wait()))
        with mock.patch.object(J, "find_opencode_session", return_value=None), \
                mock.patch.object(J, "find_claude_session", return_value=None):
            done = J.stop_jobs(None, [self.p.pid, other.pid])
        self.assertEqual([d["stopped"] for d in done], [True, False])
        self.p.wait(timeout=10)                                          # the job was ended
        self.assertIsNone(other.poll())                                  # the other holder was not
        rec = J.read(self.jobs / "1-1.json")
        self.assertEqual(J.state(rec), J.STOPPED)
        with mock.patch.object(J, "opencode_up", return_value=True):
            self.assertEqual(J.wake("http://x", Path(self.tmp.name) / "none.db", popen=lambda *a, **k: None), 0)


class Subagents(unittest.TestCase):
    def test_a_working_subagent_puts_its_parent_in_the_register(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "opencode.db"
            con = sqlite3.connect(db)
            con.execute("CREATE TABLE session (id text PRIMARY KEY, parent_id text)")
            con.execute("CREATE TABLE part (id text PRIMARY KEY, message_id text, session_id text, time_created "
                        "integer, time_updated integer, data text)")
            con.executemany("INSERT INTO session VALUES (?, ?)", [("parent", None), ("child", "parent"),
                                                                 ("grandchild", "child"), ("other", None)])
            con.commit()
            con.close()
            with mock.patch.object(J, "_descendants", return_value=[]):
                live = J.live_register({"grandchild": {"state": "generating"}}, db, Path(tmp) / "none")
        self.assertEqual(set(live["opencode"]), {"grandchild", "child", "parent"})
        self.assertIn("a subagent working", live["opencode"]["parent"])


class Delivery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = Path(self.tmp.name) / "1-1.json"

    def write(self, wrapper_alive: bool, **kw):
        me = {"pid": os.getpid(), "started": J.proc_started(os.getpid())}
        self.f.write_text(json.dumps({"state": "done", "exit": 0, "cmd": ["x"], "cwd": ".", "started": 1,
                                      "wrapper": me if wrapper_alive else {"pid": 999999, "started": 1.0},
                                      "job": [], **kw}), encoding="utf-8")

    def test_a_job_whose_wrapper_is_gone_tells_its_session_when_it_ends(self):
        self.write(False)
        with mock.patch.object(J, "wake") as w:
            J.tell_if_undelivered(self.f)
        w.assert_called_once_with(only=self.f)

    def test_a_live_wrapper_delivers_it_itself(self):
        for alive, kw in ((True, {}), (False, {"delivered": True}), (False, {"woken": 5})):
            self.write(alive, **kw)
            with mock.patch.object(J, "wake") as w:
                J.tell_if_undelivered(self.f)
            w.assert_not_called()

    def test_an_agent_program_that_fails_at_once_does_not_count_as_told(self):
        failed = mock.Mock(**{"poll.return_value": 1})                 # e.g. claude: its sign-in has expired
        with mock.patch("builtins.print"):
            self.assertFalse(J._launch(lambda *a, **k: failed, ["claude.exe"], ".", Path(self.tmp.name) / "w.log"))
        running = mock.Mock(**{"poll.return_value": None})
        with mock.patch.object(J, "LAUNCH_CHECK_S", 0.6):
            self.assertTrue(J._launch(lambda *a, **k: running, ["claude.exe"], ".", Path(self.tmp.name) / "w.log"))

    def test_claude_is_started_without_another_sessions_variables(self):
        with mock.patch.dict(os.environ, {"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "x", "CLAUDE_PID": "5",
                                          "CLAUDE_CONFIG_DIR": "keep", "PATH": os.environ.get("PATH", "")}):
            env = J.agent_env()
        self.assertNotIn("CLAUDECODE", env)
        self.assertNotIn("CLAUDE_CODE_SESSION_ID", env)
        self.assertNotIn("CLAUDE_PID", env)
        self.assertEqual(env.get("CLAUDE_CONFIG_DIR"), "keep")


class InterruptedWithRegister(Interrupted):
    """The same sessions as Interrupted, now with Strata's last register."""

    def run_wake(self, live):
        calls = []
        with mock.patch.object(J, "opencode_up", return_value=True), \
                mock.patch.object(J, "opencode_exe", return_value=["opencode.exe"]), \
                mock.patch.object(J, "claude_exe", return_value=["claude.exe"]), mock.patch("builtins.print"):
            J.wake_interrupted("http://127.0.0.1:4096", self.db, self.projects, popen=lambda a, **k: calls.append(a),
                               boot=self.boot, now=self.now, since_boot=self.boot - 30000, live=live)
        return {a[a.index("--session") + 1] if "--session" in a else a[2]: a[-1] for a in calls}

    def test_only_sessions_working_when_the_pc_went_down_are_told(self):
        down = J.went_down_at(self.boot, self.db, self.projects)
        live = {"time": down + 30, "opencode": {"tool": "a command running"},
                "claude": {"cl-tool": "a command running"}}
        told = self.run_wake(live)
        # mid, asked, step-tools, long-before, cl-asked, cl-old look mid-turn but were not working: left alone;
        # "down" (a request that failed while Strata was down) does not depend on the register
        self.assertEqual(set(told), {"tool", "cl-tool", "down"})
        self.assertIn("(you had a command running)", told["tool"])

    def test_a_final_entry_from_shutdown_says_windows_was_restarted(self):
        down = J.went_down_at(self.boot, self.db, self.projects)
        told = self.run_wake({"time": down + 20, "shutting_down": True, "opencode": {"tool": "a command running"},
                              "claude": {}})
        self.assertIn("Windows was shut down or restarted at", told["tool"])
        self.assertIn("(you had a command running)", told["tool"])

    def test_a_register_from_long_before_the_crash_does_not_decide(self):
        down = J.went_down_at(self.boot, self.db, self.projects)
        told = self.run_wake({"time": down - 3600, "opencode": {}, "claude": {}})   # Strata stopped an hour before
        self.assertIn("mid", told)                                     # the fallback (previous boot) decides
        self.assertIn("long-before", told)


if __name__ == "__main__":
    unittest.main()
