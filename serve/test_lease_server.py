"""My own tests for job lease-2-server: the lease API of serve/server.py beyond the verdict - the checkpoint on
/unload {"save": true}, a lease on an unloaded model, a resume that cannot run, the model-only key's lease bodies
and the watcher's lease pass.  Mock engines only: no GPU, no ports beyond the test server's own."""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys_path_root = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(sys_path_root))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, serve  # noqa: E402
from serve.test_hibernate import HibernatingEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self, t=10_000.0):
        self.t = t

    def __call__(self):
        return self.t


class LeaseServer(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = HibernatingEngine(tok, "</think>\n\nok", max_context=4096)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.svc.hibernate_dir = str(Path(self.tmp.name) / "hib")
        self.svc.leases_dir = str(Path(self.tmp.name) / "leases")
        self.clock = Clock()
        self.svc.lease_clock = self.clock
        env = mock.patch.dict(os.environ, {"STRATA_SUSPEND_JOBS_DIR": str(Path(self.tmp.name) / "jobs")})
        env.start()
        self.addCleanup(env.stop)
        self.svc._available_ram_mib = lambda: 90000
        self.svc.free_vram_mib = lambda: 24000
        self.svc.RESUME_SETTLE_S = 0.0
        self.engine.released = False
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self._close_table)

    def _close_table(self):
        self.quiet.__exit__(None, None, None)
        t = getattr(self.svc, "_lease_table", None)
        if t is not None:
            t.close()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def req(self, path, body=None, timeout=30):
        r = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"},
                                   method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            with e:
                try:
                    return e.code, json.loads(e.read() or b"{}")
                except ValueError:
                    return e.code, {}

    def lease(self, mode="full"):
        s, b = self.req("/lease", {"cmd": "python job.py", "cwd": "C:/work", "mode": mode,
                                   "session": {"agent": "claude", "id": "sess-1"}})
        self.assertEqual(s, 200, b)
        return b

    # ------------------------------------------------------------ the checkpoint on /unload {"save": true}

    def test_unload_save_checkpoints_when_the_table_is_open(self):
        self.svc.lease_table()                           # main() opens it at the start; a save checkpoints
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)
        cp = json.loads((Path(self.svc.leases_dir) / "checkpoint.json").read_text(encoding="utf-8"))
        self.assertEqual(cp["leases"], [])               # no lease holds: the save still checkpoints
        self.assertTrue(any("hib-1-0.strs" in f for f in cp["files"]), cp["files"])
        self.assertIn("register", cp)

    def test_unload_save_without_a_table_writes_no_lease_folder(self):
        self.svc.leases_dir = None                       # no table opened: /unload {"save": true} stays as before
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)
        self.assertFalse((Path(self.svc.hibernate_dir) / "leases").exists())

    # ------------------------------------------------------------ the clean-shutdown mark

    def _cp(self):
        return json.loads((Path(self.svc.leases_dir) / "checkpoint.json").read_text(encoding="utf-8"))

    def test_the_clean_shutdown_mark_survives_a_running_beat(self):
        self.svc.start_lease_beat(0.01)                  # a beat every 10 ms
        time.sleep(0.1)
        self.svc.lease_beat(shutting_down=True)          # stops the beat thread, then marks in one atomic write
        time.sleep(0.5)                                  # ~50 beats would have run by now
        self.assertTrue(self._cp().get("shutting_down"), "a beat overwrote the clean-shutdown mark")
        self.assertIsNone(getattr(self.svc, "_lease_beat_thread", None))   # the beat thread is gone

    def test_stop_lease_beat_ends_the_thread(self):
        self.svc.start_lease_beat(0.01)
        time.sleep(0.05)
        self.svc.stop_lease_beat()
        self.assertIsNone(getattr(self.svc, "_lease_beat_thread", None))
        before = self._cp()["time"]
        time.sleep(0.1)
        self.assertEqual(self._cp()["time"], before)     # no beat after the stop

    # ------------------------------------------------------------ a lease on an unloaded model

    def test_lease_on_an_unloaded_model_holds_without_a_resume_after(self):
        self.engine.unload()
        b = self.lease("full")
        self.assertEqual(b["status"], "not loaded")
        self.assertFalse(self.svc.lease_table().get(b["id"])["resume_model"])
        self.req(f"/lease/{b['id']}/end", {"exit": 0})
        self.assertEqual(self.svc.lease_pass(), "")
        self.assertFalse(self.engine.running)

    def test_failed_suspend_ends_the_lease_again(self):
        from serve.server import SessionRefused
        self.engine.hibernate_error = SessionRefused("memory", "no room", False)
        s, _ = self.req("/lease", {"cmd": "x", "cwd": ".", "mode": "full", "session": None})
        self.assertEqual(s, 503)
        self.assertEqual(self.req("/leases")[1]["leases"], [])
        self.assertTrue(self.engine.running)             # nothing was stopped

    # ------------------------------------------------------------ the watcher's pass

    def test_a_resume_that_cannot_run_is_tried_again_next_pass(self):
        b = self.lease("full")
        self.engine.fail_always = True                   # the engine will not start: the resume waits
        self.req(f"/lease/{b['id']}/end", {"exit": 0})   # the end route's own pass fails and puts the resume back
        self.assertTrue(self.svc.lease_pass().startswith("waiting"))
        self.engine.fail_always = False
        self.assertEqual(self.svc.lease_pass(), "resumed")
        self.assertTrue(self.engine.running)

    def test_first_pass_has_no_pause_check(self):
        b = self.lease("full")
        self.clock.t += 600                              # no previous pass: the gap does not refresh anything
        self.svc.lease_pass()
        self.assertEqual(self.svc.lease_table().get(b["id"])["ended"], "expired")

    def test_the_watcher_loop_runs_the_lease_pass(self):
        b = self.lease("full")
        self.svc.start_hold_watch(every_s=0.2)
        self.clock.t += 46                               # the fake clock only moves here: the watcher's passes
        deadline = time.time() + 10                      # use real time for the wait, the fake one for the expiry
        while time.time() < deadline and self.svc.lease_table().get(b["id"])["ended"] is None:
            time.sleep(0.05)
        self.assertEqual(self.svc.lease_table().get(b["id"])["ended"], "expired")

    # ------------------------------------------------------------ the model-only key

    def test_model_key_may_lease_but_not_end_everyones_lease(self):
        from serve.server import model_key_control_ok
        self.assertTrue(model_key_control_ok("/lease", {"cmd": "x", "cwd": ".", "mode": "ram", "session": None}))
        self.assertTrue(model_key_control_ok("/lease/" + "ab" * 16 + "/renew", {}))
        self.assertTrue(model_key_control_ok("/lease/" + "ab" * 16 + "/end", {"exit": 0}))
        self.assertFalse(model_key_control_ok("/lease", {"cmd": "x", "cwd": ".", "mode": "ram", "hold_pids": [1]}))
        self.assertFalse(model_key_control_ok("/lease/" + "ab" * 16 + "/end", {"exit": 0, "force": True}))

    # ------------------------------------------------------------ the routes' bodies

    def test_bad_bodies_are_refused(self):
        self.assertEqual(self.req("/lease", {"cmd": "x", "cwd": "."})[0], 400)            # no mode
        self.assertEqual(self.req("/lease", {"cmd": "x", "cwd": ".", "mode": "full", "session": 5})[0], 400)
        b = self.lease("full")
        self.assertEqual(self.req(f"/lease/{b['id']}/end", {"exit": "x"})[0], 400)
        self.assertEqual(self.req("/lease/not-an-id/end", {"exit": 0})[0], 404)

    def test_chat_waits_for_a_lease_on_a_stream(self):
        import socket
        import serve.server as server
        b = self.lease("full")
        body = json.dumps({"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode()
        with mock.patch.object(server, "KEEPALIVE_S", 0.3):
            c = socket.create_connection(("127.0.0.1", self.httpd.server_address[1]))
            c.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                      b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
            deadline = time.time() + 5
            while time.time() < deadline and not self.svc.status.get("held_waiting"):
                time.sleep(0.05)
            self.assertEqual(self.svc.status.get("held_waiting"), 1)
            self.assertFalse(self.engine.running)        # nothing loaded over the lease
            c.close()
        self.req(f"/lease/{b['id']}/end", {"exit": 0})
        self.svc.lease_pass()
        self.assertTrue(self.engine.running)

    def test_two_wrappers_the_last_end_resumes_once(self):
        a, b = self.lease("ram"), self.lease("ram")      # both share one RAM-only suspend
        self.assertTrue(self.engine.running)
        self.assertTrue(self.svc.released())
        self.req(f"/lease/{a['id']}/end", {"exit": 0})
        self.svc.lease_pass()
        self.assertTrue(self.svc.released())             # b still holds: nothing came back
        self.req(f"/lease/{b['id']}/end", {"exit": 0})
        self.svc.lease_pass()
        self.assertTrue(self.engine.running)
        self.assertFalse(self.svc.released())
        self.assertFalse(self.svc.lease_table().take_resume())

    # ------------------------------------------------------------ queued while /lease was suspending

    def test_a_request_queued_during_a_suspend_waits_not_503(self):
        """The race the finding names: the request passed _wait_for_hold while the model was still loaded, then the
        lease's unload stopped it under the request before its own load.  The load must wait it out, not 503."""
        real_load = self.svc.load
        reached, go = threading.Event(), threading.Event()
        calls = []

        def slow_load(*a, **kw):
            calls.append(1)
            if len(calls) == 1:
                reached.set()
                self.assertTrue(go.wait(10))             # let /lease finish its suspend first
            return real_load(*a, **kw)

        self.svc.load = slow_load
        self.addCleanup(setattr, self.svc, "load", real_load)
        out = {}

        def ask():
            out["r"] = self.req("/v1/chat/completions", {"model": "m", "messages": [{"role": "user",
                                                                                    "content": "hi"}]}, timeout=60)
        th = threading.Thread(target=ask, daemon=True)
        th.start()
        self.assertTrue(reached.wait(10))                # the request is between the wait-check and its load
        b = self.lease("full")                           # the model stops under the queued request
        go.set()
        time.sleep(1.5)
        self.assertNotIn("r", out)                       # it waits: no 503 over the lease
        self.assertFalse(self.engine.running)
        self.req(f"/lease/{b['id']}/end", {"exit": 0})
        self.svc.lease_pass()
        th.join(30)
        self.assertEqual(out["r"][0], 200)               # answered once the model is back
        self.assertTrue(self.engine.running)

    # ------------------------------------------------------------ a lease starting in the gap before the generation

    def _lease_now(self):
        """A job's lease, as POST /lease makes it: the lease first (with /lease's own "was it loaded" rule, which
        counts a resume in flight as loaded), then the suspend."""
        lid = self.svc.lease_table().create(cmd="job", cwd=".", mode="full",
                                            model_loaded=self.svc.lease_model_loaded())["id"]
        deadline = time.time() + 30                     # "busy" while another thread has the turn (a resume under
        while self.svc.unload(save=True) == "busy" and time.time() < deadline:   # way): retry, as the wrapper
            time.sleep(0.05)                            # does - else the lease stands over a loaded model
        return lid

    def test_a_lease_after_the_load_waits_at_the_generation_not_503(self):
        """The request passed its load, then a lease suspended the model before its generation: the generate
        path gives the turn back, waits the lease out and takes the turn again - no 503, a normal 200."""
        real_load, box = self.svc.load, []

        def load(*a, **kw):
            r = real_load(*a, **kw)
            if not box:
                box.append(self._lease_now())            # the model stops right after this request's load
            return r
        self.svc.load = load
        self.addCleanup(setattr, self.svc, "load", real_load)

        def end_later():
            time.sleep(1.5)
            self.req(f"/lease/{box[0]}/end", {"exit": 0})
        threading.Thread(target=end_later, daemon=True).start()
        t0 = time.time()
        s, b = self.req("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                        timeout=60)
        self.assertEqual(s, 200, b)                      # no 503 over the lease
        self.assertIn("ok", json.dumps(b))
        self.assertGreater(time.time() - t0, 1.0)        # it waited the lease out
        self.assertEqual(self.svc.lease_live(), [])

    def test_a_stream_waits_out_a_second_lease_and_is_answered(self):
        """A stream waiting in _wait_for_hold: the first lease ends and a second starts right after it.  The
        stream keeps waiting (keep-alives, no error event) and is answered normally once both are over."""
        first = self._lease_now()
        real_pass, box, second = self.svc.lease_pass_quietly, [], threading.Event()

        def lease_pass_quietly(*a, **kw):
            r = real_pass(*a, **kw)
            if not box and not self.svc.lease_live():
                box.append(self._lease_now())            # a second job leases the model right away
                second.set()
            return r
        self.svc.lease_pass_quietly = lease_pass_quietly
        self.addCleanup(setattr, self.svc, "lease_pass_quietly", real_pass)

        def end_first():
            time.sleep(1.0)
            self.req(f"/lease/{first}/end", {"exit": 0})

        def end_second():
            self.assertTrue(second.wait(10))
            time.sleep(1.0)
            self.req(f"/lease/{box[0]}/end", {"exit": 0})
        threading.Thread(target=end_first, daemon=True).start()
        threading.Thread(target=end_second, daemon=True).start()

        import socket
        body = json.dumps({"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode()
        c = socket.create_connection(("127.0.0.1", self.httpd.server_address[1]))
        self.addCleanup(c.close)
        c.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                  b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        c.settimeout(30)
        got = b""
        while b"[DONE]" not in got:
            got += c.recv(65536)
        text = got.decode(errors="replace")
        self.assertIn("200 OK", text)
        self.assertNotIn('"error"', text)                # no error event: it waited the second lease out
        self.assertIn("ok", text)
        self.assertEqual(self.svc.lease_live(), [])

    # ------------------------------------------------------------ the resume after the lease waits for memory

    def _memory_short_until(self, free):
        """The job's memory shows as free only once `free` is set: until then the resume after the lease waits."""
        real = self.svc.resume_blocked
        self.svc.resume_blocked = lambda: real() if free.is_set() else "1000 MiB of RAM available, it needs 50000"
        self.addCleanup(setattr, self.svc, "resume_blocked", real)

    def test_a_request_that_waited_out_a_lease_also_waits_out_the_resume(self):
        """Live test 1 (Oct 23): the lease ended, the resume found the job's memory not free yet, and the waiting
        request was answered 503 although the model came back by itself 5 s later.  It must wait for that resume."""
        b, free, out = self.lease("full"), threading.Event(), {}
        self._memory_short_until(free)

        def ask():
            out["r"] = self.req("/v1/chat/completions", {"model": "m", "messages": [{"role": "user",
                                                                                    "content": "hi"}]}, timeout=60)
        th = threading.Thread(target=ask, daemon=True)
        th.start()
        time.sleep(1.0)
        self.req(f"/lease/{b['id']}/end", {"exit": 0})   # the end route's own resume fails: the memory is short
        time.sleep(2.0)
        self.assertNotIn("r", out)                       # it waits for the resume, no 503
        self.assertFalse(self.engine.running)
        free.set()
        th.join(30)
        self.assertEqual(out["r"][0], 200, out["r"])
        self.assertTrue(self.engine.running)

    def test_a_stream_that_waited_out_a_lease_also_waits_out_the_resume(self):
        import socket
        import serve.server as server
        b, free = self.lease("full"), threading.Event()
        self._memory_short_until(free)
        body = json.dumps({"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode()
        with mock.patch.object(server, "KEEPALIVE_S", 0.3):
            c = socket.create_connection(("127.0.0.1", self.httpd.server_address[1]))
            self.addCleanup(c.close)
            c.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                      b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
            time.sleep(1.0)
            self.req(f"/lease/{b['id']}/end", {"exit": 0})
            threading.Timer(2.0, free.set).start()
            c.settimeout(30)
            got = b""
            while b"[DONE]" not in got:
                got += c.recv(65536)
        text = got.decode(errors="replace")
        self.assertNotIn('"error"', text)                # no error event: it waited the resume out
        self.assertIn("ok", text)
        self.assertTrue(self.engine.running)

    def test_a_lease_after_the_load_also_waits_out_the_resume(self):
        """The generate path's wait (a lease started between the request's load and its generation): after the lease
        the resume waits for memory; the request waits for it too and is answered 200."""
        real_load, box, free = self.svc.load, [], threading.Event()
        self._memory_short_until(free)

        def load(*a, **kw):
            r = real_load(*a, **kw)
            if not box:
                box.append(self._lease_now())
            return r
        self.svc.load = load
        self.addCleanup(setattr, self.svc, "load", real_load)

        def end_later():
            time.sleep(1.0)
            self.req(f"/lease/{box[0]}/end", {"exit": 0})
            time.sleep(2.0)
            free.set()
        threading.Thread(target=end_later, daemon=True).start()
        t0 = time.time()
        s, b = self.req("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                        timeout=60)
        self.assertEqual(s, 200, b)
        self.assertGreater(time.time() - t0, 2.5)        # it waited for the memory, not only the lease
        self.assertTrue(self.engine.running)

    def test_the_wait_for_the_resume_is_bounded(self):
        """Memory that never comes free (another program took it): after lease_resume_wait_s the request is answered
        503 as before, and later requests are not held either; the watcher keeps trying the resume."""
        self.svc.lease_resume_wait_s = 1.0
        b, never = self.lease("full"), threading.Event()
        self._memory_short_until(never)
        self.req(f"/lease/{b['id']}/end", {"exit": 0})
        t0 = time.time()
        s, _ = self.req("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                        timeout=30)
        self.assertEqual(s, 503)
        self.assertLess(time.time() - t0, 10)
        t0 = time.time()
        s, _ = self.req("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                        timeout=30)
        self.assertEqual(s, 503)
        self.assertLess(time.time() - t0, 0.9)           # past the bound: not held again
        self.assertTrue(self.svc.lease_table().resume_pending())   # the watcher still has it to do
        self.assertFalse(self.engine.running)

    def test_a_lease_during_a_resume_counts_the_model_as_loaded(self):
        """The first lease ended and its resume is under way (the model is not up yet): a lease created now must
        count the model as loaded (lease_model_loaded()), else it records resume_model False and the model stays
        off for good once that lease ends."""
        real, in_resume, go_on = self.svc.ensure_loaded, threading.Event(), threading.Event()

        def ensure_loaded(*a, **kw):
            if kw.get("resume"):
                in_resume.set()                            # the resume is in flight: the model is not up yet
                self.assertTrue(go_on.wait(10))
            return real(*a, **kw)
        self.svc.ensure_loaded = ensure_loaded
        self.addCleanup(setattr, self.svc, "ensure_loaded", real)

        first = self._lease_now()
        threading.Thread(target=lambda: self.req(f"/lease/{first}/end", {"exit": 0}), daemon=True).start()
        self.assertTrue(in_resume.wait(10))
        self.assertFalse(self.engine.running)
        self.assertTrue(self.svc.lease_model_loaded())     # /lease's rule: loaded, released or a resume in flight
        lid = self.svc.lease_table().create(cmd="job2", cwd=".", mode="full",
                                            model_loaded=self.svc.lease_model_loaded())["id"]
        self.assertTrue(self.svc.lease_table().get(lid)["resume_model"])
        go_on.set()
        self.req(f"/lease/{lid}/end", {"exit": 0})
        self.svc.lease_pass()
        self.assertTrue(self.engine.running)               # the model came back after the second lease


if __name__ == "__main__":
    unittest.main(verbosity=2)
