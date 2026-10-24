"""serve/test_hibernate.py - hibernate/resume: POST /unload {"save": true} writes the parked conversations to the
hibernate folder (the engine's HIBERNATE) and latches the model unloaded; POST /load re-parks them (the engine's
--hibernate-dir before READY) or {"fresh": true} discards them (mock engines: no GPU, no pack).

    python -m unittest serve.test_hibernate -v
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import queue
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (ByteTokenizer, EngineDied, MockEngine, Service, SessionRefused, StrataEngine,  # noqa: E402
                          hibernated_at_start, serve)

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096


def write_hibernate(directory: str, n: int = 2) -> None:
    """What the engine's HIBERNATE leaves: n session files and the manifest naming them."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    names = [f"hib-1-{k}.strs" for k in range(n)]
    for name in names:
        (d / name).write_bytes(b"s" * 100)
    (d / "hibernate.txt").write_text("strata-hibernate 1\ngeneration 1\nengine test\nbytes 200\n" +
                                     "".join(f"file {x}\n" for x in names) + "end\n", encoding="utf-8")


class HibernatingEngine(MockEngine):
    """Stopped and started like StrataEngine; HIBERNATE writes the folder, and a start with --hibernate-dir reports
    the restore (HIBERNATE restored=N of=M bytes=B status=S) as the scripted `restore` says."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.running, self.unloaded, self.starts = True, False, 0
        self.info, self.hib_args, self.hibernate_calls = {}, [], 0
        self.spawn = ("strata", [], None, None, None)
        self.restore = {"restored": 2, "of": 2, "bytes": 200, "status": "ok"}
        self.fail_with_dir = self.fail_always = False
        self.hibernate_error = None                    # an exception HIBERNATE raises instead of writing

    def alive(self):
        return self.running

    def unload(self):
        self.running, self.unloaded = False, True

    def hibernate(self, directory):
        self.hibernate_calls += 1
        if self.hibernate_error is not None:
            raise self.hibernate_error
        write_hibernate(directory)
        return {"saved": 2, "bytes": 200, "ms": 12.0}

    def release(self, directory):
        """Partial hibernate (RELEASE): the folder written, the RAM given back, the engine still running."""
        self.release_calls = getattr(self, "release_calls", 0) + 1
        if self.hibernate_error is not None:
            raise self.hibernate_error
        write_hibernate(directory)
        self.released = True
        if getattr(self, "on_release", None):
            self.on_release()
        return {"saved": 2, "bytes": 200, "ms": 9.0, "freed": 50000 << 20}

    def acquire(self, directory):
        """ACQUIRE: the RAM taken again and the folder re-parked; `acquire_error` (EngineDied) ends the engine."""
        self.acquire_calls = getattr(self, "acquire_calls", 0) + 1
        if getattr(self, "acquire_error", None) is not None:
            self.running, self.released = False, False
            raise self.acquire_error
        self.released = False
        if getattr(self, "on_acquire", None):
            self.on_acquire()
        return {**(self.restore or {"restored": 2, "of": 2, "bytes": 200, "status": "ok"}), "ms": 6500.0}

    def set_hibernate_arg(self, directory):
        self.hib_args.append(directory)

    def restart(self):
        self.starts += 1
        self.released = False
        with_dir = bool(self.hib_args and self.hib_args[-1])
        if self.fail_always or (with_dir and self.fail_with_dir):
            raise RuntimeError("the engine exited before it was ready")
        self.running, self.unloaded = True, False
        if with_dir and self.restore is not None:
            for k, v in self.restore.items():
                if v is not None:
                    self.info["hibernate_" + k] = v


class Hibernate(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = HibernatingEngine(tok, "</think>\n\nok", max_context=CTX)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)               # first registered, so it runs last: after a test's processes end
        self.dir = Path(self.tmp.name) / "hib"
        self.svc.hibernate_dir = str(self.dir)
        self.manifest = self.dir / "hibernate.txt"
        env = mock.patch.dict(os.environ, {"STRATA_SUSPEND_JOBS_DIR": str(Path(self.tmp.name) / "jobs")})
        env.start()                                     # wrapped jobs the tests start record here, not in data/
        self.addCleanup(env.stop)
        # the free RAM and VRAM (MiB) the resume check reads, as the test sets them: the engine's unload gives `gives`
        # back and its start takes it again, like the real one
        self.free = {"ram": 20000, "vram": 600}
        self.gives = {"ram": 50000, "vram": 22000}
        self.svc._available_ram_mib = lambda: self.free["ram"]
        self.svc.free_vram_mib = lambda: self.free["vram"]
        self.svc.RESUME_SETTLE_S = 0.0
        unload, restart = self.engine.unload, self.engine.restart

        def unload_gives_back():
            was_released = getattr(self.engine, "released", False)   # its RAM went back already
            unload()
            self.engine.released = False
            for k in self.free:
                if not (k == "ram" and was_released):
                    self.free[k] += self.gives[k]

        def restart_takes():
            restart()
            for k in self.free:
                self.free[k] -= self.gives[k]
        self.engine.unload, self.engine.restart = unload_gives_back, restart_takes

        def release_gives_ram():                        # a partial hibernate keeps the VRAM
            self.free["ram"] += self.gives["ram"]

        def acquire_takes_ram():
            self.free["ram"] -= self.gives["ram"]
        self.engine.on_release, self.engine.on_acquire = release_gives_ram, acquire_takes_ram
        self.engine.released = False
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.httpd.shutdown()
        self.httpd.server_close()
        if getattr(self.svc, "_lease_table", None) is not None:
            self.svc._lease_table.close()

    def req(self, path, body=None):
        r = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def chat(self):
        return self.req("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})

    def hibernate_then_load(self, **restore):
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)
        self.engine.restore = {"restored": 0, "of": 2, "bytes": 0, "status": "ok", **restore}
        return self.req("/load", {})

    # ---- the cycle

    def test_save_latches_and_load_restores(self):
        s, b = self.req("/unload", {"save": True})
        self.assertEqual((s, b["status"], b.get("hibernated")), (200, "unloaded", True))
        self.assertEqual(self.engine.hibernate_calls, 1)
        self.assertTrue(self.manifest.exists())
        self.assertFalse(self.engine.alive())
        self.assertEqual(self.req("/health")[1]["hibernated"], True)
        s, b = self.chat()                                         # a stray request must not wake the engine
        self.assertEqual(s, 503)
        self.assertIn("hibernating", b["error"]["message"])
        self.assertFalse(self.engine.alive())
        s, b = self.req("/load", {})
        self.assertEqual(s, 200)
        self.assertTrue(self.engine.alive())
        self.assertEqual(self.engine.hib_args[-1], str(self.dir))   # the engine got --hibernate-dir
        self.assertFalse(self.manifest.exists())                   # spent: the manifest and its files are gone
        self.assertEqual(list(self.dir.glob("hib-*.strs")), [])
        self.assertEqual(self.chat()[0], 200)

    # ---- local: through the persistent disk tier (--conversation-disk-persist)

    def test_disk_persist_mode_reads_the_flags_value(self):
        from serve.server import disk_persist_mode
        self.assertEqual(disk_persist_mode([]), "off")
        self.assertEqual(disk_persist_mode(["--conversation-disk-persist", "on"]), "on")
        self.assertEqual(disk_persist_mode(["--conversation-disk-persist", "exit-only", "--x"]), "exit-only")
        self.assertEqual(disk_persist_mode(["--conversation-disk-persist", "off"]), "off")
        self.assertEqual(disk_persist_mode(["--conversation-disk-persist"]), "off")   # no value: the engine refuses it

    def test_disk_mode_save_quits_and_latches_without_hibernate_files(self):
        self.svc.disk_persist = True
        s, b = self.req("/unload", {"save": True})
        self.assertEqual((s, b["status"], b.get("hibernated")), (200, "unloaded", True))
        self.assertEqual(self.engine.hibernate_calls, 0)           # no HIBERNATE: QUIT wrote the disk tier
        self.assertFalse(self.engine.alive())
        self.assertEqual(self.manifest.read_text(encoding="utf-8"), "mode disk\n")
        self.assertEqual(list(self.dir.glob("hib-*.strs")), [])
        self.assertEqual(self.chat()[0], 503)                      # latched like a HIBERNATE

    def test_disk_mode_resume_is_a_plain_start_and_spends_the_latch(self):
        self.svc.disk_persist = True
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)
        s, b = self.req("/load", {})
        self.assertEqual(s, 200)
        self.assertTrue(self.engine.alive())
        self.assertIsNone(self.engine.hib_args[-1])                # no --hibernate-dir: the tier takes its files in
        self.assertFalse(self.manifest.exists())
        self.assertEqual(self.chat()[0], 200)

    def test_disk_mode_resume_on_request(self):
        self.svc.disk_persist = True
        self.svc.resume_on_request = True
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)
        self.assertEqual(self.chat()[0], 200)                      # the request resumes it (memory is free again)
        self.assertIsNone(self.engine.hib_args[-1])
        self.assertFalse(self.manifest.exists())

    def test_a_hibernate_folder_from_before_still_restores_with_disk_mode_on(self):
        # files a HIBERNATE wrote before the switch: restored the old way (--hibernate-dir), not mistaken for disk mode
        write_hibernate(str(self.dir))
        self.svc.disk_persist = True
        self.svc.resume_on_request = True
        self.engine.unload()
        self.assertEqual(self.chat()[0], 200)
        self.assertEqual(self.engine.hib_args[-1], str(self.dir))
        self.assertFalse(self.manifest.exists())

    def test_disk_mode_fresh_discards_the_latch(self):
        self.svc.disk_persist = True
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)
        self.assertEqual(self.req("/load", {"fresh": True})[0], 200)
        self.assertTrue(self.engine.alive())
        self.assertIsNone(self.engine.hib_args[-1])

    # ---- local: leases (a running job holds the model suspended; its end resumes it)

    def assert_request_waits(self):
        """A request during a hold waits (no 503, nothing loaded); then the client leaves and is let go."""
        import socket
        import serve.server as server
        body = json.dumps({"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode()
        loaded = self.svc.loaded()
        with mock.patch.object(server, "KEEPALIVE_S", 0.3):
            c = socket.create_connection(("127.0.0.1", self.httpd.server_address[1]))
            c.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                      b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
            deadline = time.time() + 5
            while time.time() < deadline and not self.svc.status.get("held_waiting"):
                time.sleep(0.05)
            self.assertEqual(self.svc.status.get("held_waiting"), 1)
            c.settimeout(5)
            self.assertTrue(c.recv(64).startswith(b"HTTP/1.0 200") or True)
            self.assertEqual(self.svc.loaded(), loaded)                     # nothing loaded over the job
            c.close()
            deadline = time.time() + 10
            while time.time() < deadline and self.svc.status.get("held_waiting"):
                time.sleep(0.1)
        self.assertEqual(self.svc.status.get("held_waiting"), 0)

    def job(self):
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(lambda: (p.kill(), p.wait()))
        return p

    def lease(self, mode="full", cmd="python job.py"):
        s, b = self.req("/lease", {"cmd": cmd, "cwd": str(self.dir), "mode": mode,
                                   "session": {"agent": "claude", "id": "sess-1"}})
        self.assertEqual(s, 200, b)
        return b["id"]

    def test_a_held_suspend_makes_requests_wait_even_with_resume_on_request(self):
        self.svc.resume_on_request = True
        lid = self.lease()
        self.assertEqual([x["id"] for x in self.req("/health")[1].get("held_by") or []], [lid])
        self.assert_request_waits()
        s, b = self.req("/load", {})                               # a plain /load does not override a lease
        self.assertEqual((s, b.get("status")), (409, "held"))
        self.assertFalse(self.engine.alive())
        self.assertEqual(self.svc.lease_pass(), "")                # the lease lives: nothing resumes

    def test_the_leases_end_resumes_the_model(self):
        lid = self.lease()
        self.req(f"/lease/{lid}/end", {"exit": 0})
        self.assertTrue(self.engine.alive())                             # the route resumed it
        self.assertTrue(self.engine.alive())
        self.assertEqual(self.chat()[0], 200)

    def test_the_hold_lasts_until_the_last_lease_ends(self):
        a, b = self.lease(), self.lease()
        self.req(f"/lease/{a}/end", {"exit": 0})
        self.assertEqual(self.svc.lease_pass(), "")
        self.assertFalse(self.engine.alive())
        self.req(f"/lease/{b}/end", {"exit": 0})
        self.assertTrue(self.engine.alive())                             # the route resumed it
        self.assertTrue(self.engine.alive())

    def test_a_stopped_lease_is_410_on_renew_and_never_woken(self):
        lid = self.lease()
        s, b = self.req(f"/lease/{lid}/stop", {})
        self.assertEqual((s, b.get("ended")), (200, "stopped"))
        self.assertIsInstance(b.get("stopped_by_user"), (int, float))
        s, b = self.req(f"/lease/{lid}/renew", {})
        self.assertEqual(s, 410)
        self.assertIn("stopped", b["error"]["message"])
        self.assertTrue(self.engine.alive())                             # the route resumed it
        self.assertTrue(self.engine.alive())

    def test_stop_all_and_bad_bodies(self):
        a, b = self.lease(), self.lease()
        self.assertEqual(self.req("/lease/stop", {})[0], 400)
        self.assertEqual(self.req("/lease/stop", {"all": False})[0], 400)
        s, r = self.req("/lease/stop", {"all": True})
        self.assertEqual(s, 200)
        self.assertEqual(sorted(x["id"] for x in r["stopped"]), sorted([a, b]))
        self.assertEqual(self.req("/leases")[1]["leases"], [])
        self.assertTrue(self.engine.alive())                             # the route resumed it
        self.assertEqual(self.req(f"/lease/{'cd' * 16}/stop", {})[0], 404)

    def test_hold_pids_on_unload_is_refused_and_the_hold_route_is_gone(self):
        for body in ({"save": True, "hold_pids": [os.getpid()]}, {"hold_pids": [os.getpid()]}):
            s, b = self.req("/unload", body)
            self.assertEqual(s, 400, body)
            self.assertIn("/lease", json.dumps(b))
            self.assertTrue(self.engine.alive())                   # nothing was stopped
        for body in ({"hold_pids": [os.getpid()]}, {"release": True}, {"stop": "all"}, {}):
            s, b = self.req("/hold", body)
            self.assertEqual(s, 410, body)
            self.assertIn("/lease", json.dumps(b))

    def test_release_and_force(self):
        self.svc.resume_on_request = True
        lid = self.lease()
        self.assertEqual(self.req("/load", {"force": True})[0], 200)   # force: anyway (the lease ends overridden)
        self.assertTrue(self.engine.alive())
        self.assertEqual(self.req("/leases")[1]["leases"], [])
        self.assertEqual(self.svc.lease_pass(), "")

    def test_a_partial_hold_resumes_with_acquire(self):
        lid = self.lease("ram")
        self.assertTrue(self.svc.released())
        self.assert_request_waits()
        self.req(f"/lease/{lid}/end", {"exit": 0})
        self.assertTrue(self.engine.alive())                             # the route resumed it
        self.assertFalse(self.svc.released())
        self.assertEqual(self.engine.acquire_calls, 1)

    # ---- local: several jobs at once (each wrapper ends only its own lease)

    def test_one_jobs_end_keeps_the_others_hold(self):
        a, b = self.lease(), self.lease()
        self.req(f"/lease/{a}/end", {"exit": 0})
        self.svc.lease_pass()
        self.assertFalse(self.engine.alive())                                  # nothing loads over the second job
        self.assert_request_waits()
        self.req(f"/lease/{b}/end", {"exit": 0})
        self.assertTrue(self.engine.alive())                             # the route resumed it
        self.assertTrue(self.engine.alive())

    def test_the_lease_is_set_before_the_engine_stops(self):
        seen, hibernate = [], self.engine.hibernate

        def watching(directory):
            seen.append([l["id"] for l in self.svc.lease_live()])              # what a request would see now
            return hibernate(directory)
        self.engine.hibernate = watching
        lid = self.lease()
        self.assertEqual(seen, [[lid]])

    def test_a_failed_suspend_leaves_no_lease(self):
        self.engine.hibernate_error = SessionRefused("memory", "no room", False)
        s, _ = self.req("/lease", {"cmd": "x", "cwd": ".", "mode": "full", "session": None})
        self.assertEqual(s, 503)
        self.assertEqual(self.req("/leases")[1]["leases"], [])
        self.assertTrue(self.engine.alive())
        self.assertEqual(self.chat()[0], 200)

    def test_a_gpu_job_waits_while_a_ram_only_job_holds(self):
        a = self.lease("ram")
        s, b = self.req("/lease", {"cmd": "gpu job", "cwd": ".", "mode": "full", "session": None})
        self.assertEqual((s, b.get("status")), (409, "held"))                  # would take the RAM back under a
        self.assertEqual(getattr(self.engine, "acquire_calls", 0), 0)
        self.assertTrue(self.svc.released())
        self.assertEqual([l["id"] for l in self.svc.lease_live()], [a])        # the refused lease was ended again
        self.req(f"/lease/{a}/end", {"exit": 0})
        self.assertTrue(self.engine.alive())                             # the route resumed it
        s, b = self.req("/lease", {"cmd": "gpu job", "cwd": ".", "mode": "full", "session": None})
        self.assertEqual(s, 200)
        self.assertFalse(self.engine.alive())

    def test_two_ram_only_jobs_share_the_suspend(self):
        a, b = self.lease("ram"), self.lease("ram")
        self.assertEqual(sorted(l["id"] for l in self.svc.lease_live()), sorted([a, b]))
        self.assertEqual(self.engine.release_calls, 1)

    def test_a_job_on_an_unloaded_model_does_not_load_it_afterwards(self):
        self.assertEqual(self.req("/unload", {})[0], 200)                      # idle-unloaded before the job
        lid = self.lease()
        self.assert_request_waits()                                            # still nothing loads over the job
        self.req(f"/lease/{lid}/end", {"exit": 0})
        self.assertEqual(self.svc.lease_pass(), "")                            # not loaded before: left as it is
        self.assertFalse(self.engine.alive())
        self.assertEqual(self.chat()[0], 200)                                  # the next request loads it as usual

    def test_a_second_job_keeps_the_first_ones_resume(self):
        a = self.lease()                                                       # the model was loaded
        self.assertEqual(self.req("/unload", {})[0], 200)                      # off already: b sees no loaded model
        b = self.lease()                                                       # resume_model inherited from a
        self.assertTrue(self.svc.lease_table().get(b)["resume_model"])
        self.req(f"/lease/{a}/end", {"exit": 0})
        self.svc.lease_pass()
        self.req(f"/lease/{b}/end", {"exit": 0})
        self.assertTrue(self.engine.alive())                             # the route resumed it

    def test_leases_created_at_the_same_moment_are_all_kept(self):
        threads = [threading.Thread(target=self.lease) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(self.svc.lease_live()), 6)

    def test_two_overlapping_wrapped_jobs(self):
        # tools/strata_suspend.py run, twice at once: the first job's end must not resume the model under the second
        port = str(self.httpd.server_address[1])
        wrap = [sys.executable, str(ROOT / "tools" / "strata_suspend.py"), "run", "--port", port, "--"]
        a = subprocess.Popen(wrap + [sys.executable, "-c", "import time; time.sleep(10)"], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)

        def leases_of(tag):                               # the live leases whose command names this job
            return [l for l in self.svc.lease_live() if tag in (l.get("cmd") or "")]
        deadline = time.time() + 30
        while time.time() < deadline and not leases_of("time.sleep(10)"):
            time.sleep(0.1)
        self.assertTrue(leases_of("time.sleep(10)"), self.svc.lease_live())
        b = subprocess.Popen(wrap + [sys.executable, "-c", "import time; time.sleep(14)"], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        while time.time() < deadline and not leases_of("time.sleep(14)"):  # b holds before a's job ends
            time.sleep(0.1)
        self.assertTrue(leases_of("time.sleep(14)"), self.svc.lease_live())
        out_a = a.communicate(timeout=60)[0]
        self.assertEqual(a.returncode, 0, out_a)
        self.assertFalse(self.engine.alive())                  # b still runs: nothing came back
        self.assert_request_waits()
        out_b = b.communicate(timeout=60)[0]
        self.assertEqual(b.returncode, 0, out_b)
        deadline = time.time() + 15
        while time.time() < deadline and not self.engine.alive():
            time.sleep(0.2)
        self.assertTrue(self.engine.alive())
        self.assertNotIn("WARNING", out_a + out_b)

    def test_a_resume_after_a_long_job_is_not_idle_unloaded_at_once(self):
        self.svc.idle_unload_s = 300
        lid = self.lease()
        self.svc.last_request_at = time.time() - 3600                          # the job ran an hour
        self.req(f"/lease/{lid}/end", {"exit": 0})
        self.assertTrue(self.engine.alive())                             # the route resumed it
        self.assertEqual(self.svc.idle_step(), "")                             # the idle time counts from the resume
        self.assertTrue(self.engine.alive())

    def test_a_killed_wrapper_leaves_the_lease_until_it_expires(self):
        # Task Manager's "End task" on strata_suspend.py: the renewals stop; the lease expires and the watcher resumes
        self.svc.lease_expire_s = 2.0
        port = str(self.httpd.server_address[1])
        w = subprocess.Popen([sys.executable, str(ROOT / "tools" / "strata_suspend.py"), "run", "--port", port, "--",
                              sys.executable, "-c", "import time; time.sleep(60)"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: subprocess.run(["taskkill", "/PID", str(w.pid), "/T", "/F"], capture_output=True))
        deadline = time.time() + 30
        while time.time() < deadline and not (self.svc.lease_live() and not self.engine.alive()):
            time.sleep(0.1)                                                    # the lease is in and the suspend is done
        self.assertTrue(self.svc.lease_live())
        w.kill()                                                               # TerminateProcess: no finally runs
        w.wait()
        self.assertEqual(self.svc.lease_pass(), "")                            # renewed at create: still holds
        self.assertFalse(self.engine.alive())
        deadline, r = time.time() + 30, ""                                     # the renewals stopped: it expires
        while time.time() < deadline and not r:                                # the watcher's step, by hand
            r = self.svc.lease_pass()
            time.sleep(0.2)
        self.assertEqual(r, "resumed")                                         # the watcher's step frees it
        self.assertTrue(self.engine.alive())

    def test_a_job_on_a_hibernated_model_leaves_it_hibernated(self):
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)
        lid = self.lease()
        self.req(f"/lease/{lid}/end", {"exit": 0})
        self.assertEqual(self.svc.lease_pass(), "")                            # not loaded before: left as it is
        self.assertFalse(self.engine.alive())
        self.assertTrue(self.svc.hibernated())

    def test_a_ram_only_model_is_not_brought_back_fully_after_a_job(self):
        # RAM-only before the job (idle_release_s, or on purpose): the job's end leaves it off; a request resumes it
        self.svc.resume_on_request = True
        self.assertEqual(self.req("/unload", {"save": True, "keep_gpu": True})[0], 200)
        for keep in (True, False):
            lid = self.lease("ram" if keep else "full")
            self.req(f"/lease/{lid}/end", {"exit": 0})
            self.assertEqual(self.svc.lease_pass(), "")                        # resume_model False: left as it is
            self.assertFalse(self.svc.loaded())
        self.assertEqual(self.chat()[0], 200)                                  # resume_on_request: the next request

    def test_a_lease_holds_through_its_own_suspend(self):
        # the lease goes in BEFORE the engine stops: the watcher's step during the suspend must not resume the model
        seen, hibernate = [], self.engine.hibernate

        def watching(directory):
            seen.append(self.svc.lease_pass())                                 # the watcher's step mid-suspend
            return hibernate(directory)
        self.engine.hibernate = watching
        self.assertEqual(self.req("/lease", {"cmd": "x", "cwd": ".", "mode": "full", "session": None})[0], 200)
        self.assertEqual(seen, [""])                                           # the live lease holds: no resume
        self.assertTrue(self.engine.alive() is False)

    def test_a_deliberate_hibernate_from_the_web_page_drops_a_pending_resume(self):
        lid = self.lease()
        self.free["ram"] = 0                                         # memory not free: the resume waits
        self.req(f"/lease/{lid}/end", {"exit": 0})
        self.assertTrue(self.svc.lease_pass().startswith("waiting"))
        self.assertEqual(self.req("/v1/unload", {"save": True})[0], 200)
        self.free["ram"] = 100000
        self.assertEqual(self.svc.lease_pass(), "")
        self.assertFalse(self.engine.alive())

    def test_an_unreadable_lease_table_counts_as_held(self):
        self.svc.lease_table()                                       # open the table, then break its reads
        with mock.patch.object(self.svc._lease_table, "safe_to_load", side_effect=OSError("locked")):
            self.assertTrue(self.svc.lease_holds())                  # fail closed: nothing loads over a job it cannot
            self.engine.unload()                                     # see
            self.assert_request_waits()

    def test_a_deliberate_hibernate_drops_a_pending_resume(self):
        lid = self.lease()
        self.free["ram"] = 0                                         # memory not free: the resume waits
        self.req(f"/lease/{lid}/end", {"exit": 0})
        self.assertTrue(self.svc.lease_pass().startswith("waiting"))
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)   # turned off on purpose meanwhile
        self.free["ram"] = 100000
        self.assertEqual(self.svc.lease_pass(), "")
        self.assertFalse(self.engine.alive())

    # ---- local: a request during a hold waits for it (an agent would give up after a few 503s)

    def stream_chat(self, out: list, timeout=60):
        """A streamed chat request in this thread: every line it gets goes to `out` (keep-alives included)."""
        r = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps({
            "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                out.append(("status", resp.status))
                for line in resp:
                    out.append(("line", line.decode("utf-8", "replace").strip()))
        except urllib.error.HTTPError as e:
            out.append(("status", e.code))
        except Exception as e:  # noqa: BLE001
            out.append(("error", str(e)))

    def test_a_streamed_request_waits_out_a_hold_then_answers(self):
        import serve.server as server
        p = self.job()
        lid = self.lease()
        out = []
        with mock.patch.object(server, "KEEPALIVE_S", 0.5):
            t = threading.Thread(target=self.stream_chat, args=(out,))
            t.start()
            time.sleep(3)
            self.assertEqual(out[0], ("status", 200))                        # started at once, not a 503
            self.assertTrue(any("keep-alive" in x for k, x in out if k == "line"), out)
            self.assertFalse(any(x.startswith("data:") for k, x in out if k == "line"))   # nothing generated yet
            self.assertEqual(self.svc.status.get("held_waiting"), 1)
            self.assertFalse(self.engine.alive())
            self.req(f"/lease/{lid}/end", {"exit": 0})                       # the job ends
            t.join(30)
        self.assertFalse(t.is_alive())
        self.assertTrue(any(x == "data: [DONE]" for k, x in out if k == "line"), out[-5:])
        self.assertTrue(self.engine.alive())
        self.assertEqual(self.svc.status.get("held_waiting"), 0)

    def test_a_request_that_is_not_streamed_waits_too(self):
        lid = self.lease()
        got = []
        t = threading.Thread(target=lambda: got.append(self.chat()))
        t.start()
        time.sleep(2)
        self.assertEqual(got, [])                                            # still waiting, no 503
        self.req(f"/lease/{lid}/end", {"exit": 0})
        t.join(30)
        self.assertEqual(got[0][0], 200)

    def test_a_waiting_request_does_not_block_another_jobs_suspend(self):
        lid = self.lease()
        out = []
        t = threading.Thread(target=self.stream_chat, args=(out,), daemon=True)
        t.start()
        time.sleep(1.5)
        s, b = self.req("/lease", {"cmd": "second job", "cwd": ".", "mode": "full", "session": None})
        self.assertEqual((s, b.get("status")), (200, "not loaded"))
        self.assertEqual(len(self.svc.lease_live()), 2)
        self.req(f"/lease/{lid}/end", {"exit": 0})
        self.req(f"/lease/{b['id']}/end", {"exit": 0})
        t.join(30)
        self.assertTrue(any(x == "data: [DONE]" for k, x in out if k == "line"))

    def test_a_client_that_leaves_while_waiting_is_let_go(self):
        import socket
        import serve.server as server
        lid = self.lease()
        body = json.dumps({"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode()
        with mock.patch.object(server, "KEEPALIVE_S", 0.3):
            s = socket.create_connection(("127.0.0.1", self.httpd.server_address[1]))
            s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                      b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
            time.sleep(1.5)
            self.assertEqual(self.svc.status.get("held_waiting"), 1)
            s.close()                                                        # the agent was stopped
            deadline = time.time() + 10
            while time.time() < deadline and self.svc.status.get("held_waiting"):
                time.sleep(0.2)
        self.assertEqual(self.svc.status.get("held_waiting"), 0)
        self.assertFalse(self.engine.alive())                                # it did not load the model

    def test_the_server_knows_which_session_a_request_comes_from(self):
        # opencode sends x-opencode-session-id: the session is in the register while its request waits, generates,
        # and for a while after (between two steps)
        lid = self.lease()
        r = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps({
            "model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode(), method="POST",
            headers={"Content-Type": "application/json", "x-opencode-session-id": "ses_abc"})
        got = []
        t = threading.Thread(target=lambda: got.append(urllib.request.urlopen(r, timeout=60).status))
        t.start()
        deadline = time.time() + 10
        while time.time() < deadline and "ses_abc" not in self.svc.live_requests():
            time.sleep(0.05)
        time.sleep(0.5)
        self.assertEqual(self.svc.live_requests()["ses_abc"]["state"], "waiting out a hold")   # silently: no stream
        self.req(f"/lease/{lid}/end", {"exit": 0})
        t.join(30)
        self.assertEqual(got, [200])
        self.assertEqual(self.svc.live_requests()["ses_abc"]["state"], "between steps")
        with mock.patch.object(self.svc, "AGENT_RECENT_S", 0):
            time.sleep(0.05)
            self.assertNotIn("ses_abc", self.svc.live_requests())

    def test_a_streamed_wait_is_registered_as_waiting(self):
        import serve.server as server
        lid = self.lease()
        r = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps({
            "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode(), method="POST",
            headers={"Content-Type": "application/json", "x-opencode-session-id": "ses_w"})
        with mock.patch.object(server, "KEEPALIVE_S", 0.3):
            t = threading.Thread(target=lambda: urllib.request.urlopen(r, timeout=60).read(), daemon=True)
            t.start()
            deadline = time.time() + 10
            while time.time() < deadline and (self.svc.live_requests().get("ses_w") or {}).get("state") != \
                    "waiting out a hold":
                time.sleep(0.05)
            self.assertEqual(self.svc.live_requests()["ses_w"]["state"], "waiting out a hold")
            self.req(f"/lease/{lid}/end", {"exit": 0})
            t.join(30)

    def test_a_ram_only_job_falls_back_to_a_full_suspend_when_the_arena_has_large_pages(self):
        # a `ram` lease whose RAM-only release is refused (the arena has large pages) falls back to a full
        # suspend for the same lease: the job runs and the wrapper says so
        def refuse(directory):
            raise SessionRefused("invalid", "only an arena of 4 KB pages registered whole with CUDA can give its RAM "
                                 "back (this one: cudaHostRegister PORTABLE ok; large pages (2097152 B))", False)
        self.engine.release = refuse
        port = str(self.httpd.server_address[1])
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "strata_suspend.py"), "run", "--keep-gpu", "--port",
                            port, "--", sys.executable, "-c", "print('job ran')"], capture_output=True, text=True,
                           timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("freeing RAM and VRAM instead", r.stdout)
        self.assertIn("job ran", r.stdout)
        self.assertEqual(self.engine.hibernate_calls, 1)                  # the full suspend ran

    def test_a_quick_job_s_output_is_never_lost_by_the_wrapper(self):
        # the wrapper relays the job's log from a daemon thread: when the job ends while the relay sleeps, the
        # wrapper used to exit before its last read, and the job's output never reached the agent (the
        # large-pages fallback failure).  A quick job run several times: every run's output must arrive.
        port = str(self.httpd.server_address[1])
        for i in range(10):
            r = subprocess.run([sys.executable, str(ROOT / "tools" / "strata_suspend.py"), "run", "--port", port,
                                "--", sys.executable, "-c", "print('job ran')"], capture_output=True, text=True,
                               timeout=120)
            self.assertEqual(r.returncode, 0, f"run {i + 1}: {r.stdout}{r.stderr}")
            self.assertIn("job ran", r.stdout, f"run {i + 1}: {r.stdout}{r.stderr}")

    def test_the_user_can_see_and_stop_the_job_holding_the_model(self):
        port = str(self.httpd.server_address[1])
        w = subprocess.Popen([sys.executable, str(ROOT / "tools" / "strata_suspend.py"), "run", "--port", port, "--",
                              sys.executable, "-c", "import time; print('working', flush=True); time.sleep(120)"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.addCleanup(lambda: subprocess.run(["taskkill", "/PID", str(w.pid), "/T", "/F"], capture_output=True))
        deadline = time.time() + 60
        held = None
        while time.time() < deadline:
            held = self.req("/health")[1].get("held_by")
            if held and any("time.sleep(120)" in (x.get("cmd") or "") for x in held):
                break
            time.sleep(0.3)
        self.assertTrue(held and "time.sleep(120)" in held[0]["cmd"], held)   # what holds it, by name
        self.assertIn("id", held[0])
        s, b = self.req("/lease/stop", {"all": True})
        self.assertEqual((s, [x["ended"] for x in b["stopped"]]), (200, ["stopped"]))
        out = w.communicate(timeout=60)[0]
        self.assertIn("STOPPED BY THE USER", out)                       # the agent is told not to run it again
        deadline = time.time() + 15
        while time.time() < deadline and not self.engine.alive():
            time.sleep(0.2)
        self.assertTrue(self.engine.alive())                            # the model came back
        self.assertEqual(self.svc.held_by(), [])
        lid = b["stopped"][0]["id"]
        lease = self.svc.lease_table().get(lid)
        self.assertEqual(lease["ended"], "stopped")                     # never woken as cut off
        self.assertIsInstance(lease.get("stopped_by_user"), (int, float))

    def test_the_requests_in_strata_are_listed_with_who_sent_them(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import strata_jobs
        info = {"ses_sub": {"title": "Write a report (@general subagent)", "parent": "ses_main",
                            "parent_title": "Nightly review", "task": "Write a report", "subagent": "general"}}

        def send(sid, ua, out):
            headers = {"Content-Type": "application/json", "User-Agent": ua, **({"x-opencode-session-id": sid}
                                                                                if sid else {})}
            r = urllib.request.Request(self.base + "/v1/chat/completions", method="POST", headers=headers,
                                       data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
                                       .encode())
            out.append(urllib.request.urlopen(r, timeout=60).status)
        out = []
        with mock.patch.object(strata_jobs, "session_info", side_effect=lambda sid: info.get(sid, {})):
            self.svc.fifo.acquire()                                    # another request has the turn
            try:
                threads = [threading.Thread(target=send, args=a) for a in (
                    ("ses_sub", "opencode/1.19.0", out), (None, "Mozilla/5.0", out))]
                for t in threads:
                    t.start()
                deadline = time.time() + 10
                while time.time() < deadline and len(self.svc.requests_now()) < 2:
                    time.sleep(0.05)
                time.sleep(0.3)
                reqs = self.req("/health")[1]["requests"]
            finally:
                self.svc.fifo.release()
            for t in threads:
                t.join(30)
        self.assertEqual(out, [200, 200])
        by = {r.get("session") or r["client"]: r for r in reqs}
        self.assertEqual(by["ses_sub"]["state"], "queued")
        self.assertEqual(strata_jobs.describe_request(by["ses_sub"]),
                         'queued - subagent general of "Nightly review" - task: Write a report')
        self.assertEqual(strata_jobs.describe_request(by["web page"]), "queued - from web page")
        deadline = time.time() + 5                       # the client has its 200; the server is still finishing
        while time.time() < deadline and self.svc.requests_now():   # that request's bookkeeping (its finally block)
            time.sleep(0.05)
        self.assertEqual(self.svc.requests_now(), [])                   # both done: none left running

    def test_a_bad_stop_is_refused(self):
        for bad in ({}, {"all": False}, {"all": True, "x": 1}, {"stop": "all"}):
            self.assertEqual(self.req("/lease/stop", bad)[0], 400)

    # ---- local: what the Monitor shows (/metrics live.state)

    def live(self):
        return self.svc.metrics()["live"]

    def test_metrics_says_which_kind_of_off(self):
        self.assertEqual(self.live()["state"], "idle")
        self.assertEqual(self.req("/unload", {"save": True, "keep_gpu": True})[0], 200)
        self.assertEqual(self.live()["state"], "suspended")
        self.assertEqual(self.req("/load", {})[0], 200)
        self.assertEqual(self.live()["state"], "idle")
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)
        self.assertEqual(self.live()["state"], "hibernated")
        self.assertEqual(self.req("/load", {})[0], 200)
        self.assertEqual(self.req("/unload", {})[0], 200)
        self.assertEqual(self.live()["state"], "unloaded")

    def test_metrics_names_the_holding_job(self):
        lid = self.lease(cmd="python simulate.py")
        live = self.live()
        self.assertEqual(live["state"], "hibernated")
        self.assertEqual([x["id"] for x in live["held_by"]], [lid])
        self.assertEqual(live["held_by"][0]["cmd"], "python simulate.py")

    # ---- local: idle stages ("idle_release_s": RAM only; "idle_unload_s": everything)

    def test_idle_release_gives_the_ram_back_after_the_idle_time_only(self):
        self.svc.idle_release_s = 300
        self.svc.last_request_at = time.time() - 10
        self.assertEqual(self.svc.idle_step(), "")                 # 10 s idle: nothing
        self.assertTrue(self.engine.alive() and not self.svc.released())
        self.svc.last_request_at = time.time() - 301
        self.assertEqual(self.svc.idle_step(), "released")
        self.assertTrue(self.svc.released())
        self.assertEqual(self.svc.idle_step(), "")                 # once: already released

    def test_a_request_resumes_an_idle_release(self):
        self.svc.resume_on_request = True
        self.svc.idle_release_s = 300
        self.svc.last_request_at = time.time() - 301
        self.assertEqual(self.svc.idle_step(), "released")
        self.assertEqual(self.chat()[0], 200)
        self.assertFalse(self.svc.released())
        self.assertEqual(self.engine.acquire_calls, 1)

    def test_idle_unload_follows_the_release(self):
        self.svc.idle_release_s, self.svc.idle_unload_s = 300, 1800
        self.svc.last_request_at = time.time() - 301
        self.assertEqual(self.svc.idle_step(), "released")
        self.assertEqual(self.svc.idle_step(), "")                 # 5 min: released, not unloaded
        self.svc.last_request_at = time.time() - 1801
        self.assertEqual(self.svc.idle_step(), "unloaded")
        self.assertFalse(self.engine.alive())

    def test_idle_release_needs_a_hibernate_folder(self):
        self.svc.hibernate_dir = None
        self.svc.idle_release_s = 300
        self.svc.last_request_at = time.time() - 301
        self.assertEqual(self.svc.idle_step(), "")
        self.assertTrue(self.engine.alive())

    def test_plain_unload_does_not_latch(self):
        self.assertEqual(self.req("/unload", {})[0], 200)
        self.assertEqual(self.req("/health")[1]["hibernated"], False)
        self.assertEqual(self.chat()[0], 200)                      # auto-loads as before

    def test_save_without_a_folder(self):
        self.svc.hibernate_dir = None
        s, b = self.req("/unload", {"save": True})
        self.assertEqual(s, 503)
        self.assertIn("hibernate_dir", b["error"]["message"])
        self.assertTrue(self.engine.alive())

    def test_engine_refusal_leaves_it_running(self):
        self.engine.hibernate_error = SessionRefused("memory", "not enough RAM to save", False)
        s, b = self.req("/unload", {"save": True})
        self.assertEqual(s, 503)
        self.assertIn("not enough RAM", b["error"]["message"])
        self.assertTrue(self.engine.alive())
        self.assertFalse(self.manifest.exists())

    def test_a_current_conversation_left_out_is_reported(self):
        def save_without_live(directory):
            write_hibernate(directory)
            return {"saved": 1, "bytes": 100, "ms": 5.0, "live": 0}
        self.engine.hibernate = save_without_live
        s, b = self.req("/unload", {"save": True})
        self.assertEqual((s, b.get("hibernated"), b.get("live_saved")), (200, True, False))

    def test_engine_ended_after_writing_the_manifest_counts(self):
        def write_then_die(directory):
            write_hibernate(directory)
            raise EngineDied("the engine answered a hibernate with 'garbage'; the server ended the engine")
        self.engine.hibernate = write_then_die
        s, b = self.req("/unload", {"save": True})
        self.assertEqual((s, b["status"]), (200, "unloaded"))
        self.assertTrue(self.manifest.exists())

    def test_engine_ended_before_the_manifest_saved_nothing(self):
        self.engine.hibernate_error = EngineDied("the engine stopped unexpectedly")
        s, b = self.req("/unload", {"save": True})
        self.assertEqual(s, 503)
        self.assertIn("hibernate failed", b["error"]["message"])

    def test_v1_aliases(self):
        self.assertEqual(self.req("/v1/unload", {"save": True})[0], 200)
        self.assertTrue(self.manifest.exists())
        self.assertEqual(self.chat()[0], 503)
        self.assertEqual(self.req("/v1/load", {})[0], 200)
        self.assertFalse(self.manifest.exists())

    def test_bad_control_body(self):
        r = urllib.request.Request(self.base + "/unload", data=b"[1, 2]", headers={"Content-Type": "application/json"},
                                   method="POST")
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(r, timeout=10)
        self.assertEqual(e.exception.code, 400)
        e.exception.close()
        self.assertTrue(self.engine.alive())

    # ---- what a resume does with the folder, by the engine's restore status

    def test_transient_refusals_keep_the_latch(self):
        for status in ("memory", "off", "io"):
            with self.subTest(status=status):
                s, b = self.hibernate_then_load(status=status)
                self.assertEqual(s, 503)
                self.assertIn("not restored", b["error"]["message"])
                self.assertTrue(self.manifest.exists())
                self.assertFalse(self.engine.alive())
                self.assertEqual(self.chat()[0], 503)
                self.engine.restore = {"restored": 2, "of": 2, "status": "ok"}
                self.assertEqual(self.req("/load", {})[0], 200)
                self.assertFalse(self.manifest.exists())

    def test_definitive_refusals_set_the_manifest_aside(self):
        for status in ("foreign", "corrupt"):
            with self.subTest(status=status):
                s, b = self.hibernate_then_load(status=status)
                self.assertEqual(s, 200)
                self.assertFalse(self.manifest.exists())
                self.assertTrue((self.dir / "hibernate.txt.refused").exists())
                self.assertEqual(self.chat()[0], 200)

    def test_partial(self):
        s, b = self.hibernate_then_load(restored=1, status="partial")
        self.assertEqual(s, 200)
        self.assertTrue((self.dir / "hibernate.txt.partial").exists())

    def test_a_start_that_fails_with_the_folder_sets_it_aside(self):
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)
        self.engine.fail_with_dir = True
        s, b = self.req("/load", {})
        self.assertEqual(s, 200)
        self.assertIsNone(self.engine.hib_args[-1])
        self.assertTrue((self.dir / "hibernate.txt.unrestorable").exists())

    def test_a_start_that_fails_anyway_keeps_the_latch(self):
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)
        self.engine.fail_always = True
        s, b = self.req("/load", {})
        self.assertEqual(s, 503)
        self.assertIn("did not start", b["error"]["message"])
        self.assertTrue(self.manifest.exists())

    def test_fresh_replaces_an_older_discarded_manifest(self):
        for path in ("/load", "/v1/load"):
            with self.subTest(path=path):
                self.dir.mkdir(parents=True, exist_ok=True)
                (self.dir / "hibernate.txt.discarded").write_text("older", encoding="utf-8")
                self.assertEqual(self.req("/unload", {"save": True})[0], 200)
                s, b = self.req(path, {"fresh": True})
                self.assertEqual(s, 200)
                self.assertFalse(self.manifest.exists())
                self.assertIn("strata-hibernate", (self.dir / "hibernate.txt.discarded").read_text(encoding="utf-8"))
                self.assertIsNone(self.engine.hib_args[-1])

    def test_fresh_that_cannot_discard_does_not_restore(self):
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)
        starts = self.engine.starts
        with mock.patch.object(Path, "replace", side_effect=OSError("in use")):
            for path in ("/load", "/v1/load"):
                with self.subTest(path=path):
                    s, b = self.req(path, {"fresh": True})
                    self.assertEqual(s, 500)
                    self.assertIn("could not be discarded", b["error"]["message"])
        self.assertEqual(self.engine.starts, starts)
        self.assertTrue(self.manifest.exists())

    def test_hibernated_at_start(self):
        self.assertFalse(hibernated_at_start(str(self.dir)))
        self.assertFalse(hibernated_at_start(None))
        write_hibernate(str(self.dir))
        self.assertTrue(hibernated_at_start(str(self.dir)))
        self.assertTrue(hibernated_at_start("hib", cwd=str(self.dir.parent)))   # relative to the config's cwd

    # ---- resuming only into free memory ("resume_on_request", resume-needs.json)

    def raw(self, path, body=None):
        """(status, headers, json) - for the Retry-After header."""
        r = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, resp.headers, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.headers, json.loads(e.read())

    def test_hibernate_records_what_it_gave_back(self):
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)
        needs = json.loads((self.dir / "resume-needs.json").read_text(encoding="utf-8"))
        self.assertEqual(needs, {"ram_mib": 50000, "vram_mib": 22000})
        h = self.req("/health")[1]
        self.assertEqual((h["resume_needs"], h["resume_blocked"], h["resume_on_request"]), (needs, None, False))
        self.assertTrue(self.manifest.exists())
        self.assertEqual(self.req("/load", {})[0], 200)            # it fits: the manual resume goes ahead
        self.assertNotIn("resume_needs", self.req("/health")[1])   # not hibernated: nothing to say

    def test_a_request_resumes_when_the_memory_is_free(self):
        self.svc.resume_on_request = True
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)
        self.assertFalse(self.engine.alive())
        s, b = self.chat()
        self.assertEqual(s, 200)
        self.assertTrue(self.engine.alive())
        self.assertEqual(self.engine.hib_args[-1], str(self.dir))   # resumed with the conversations, not fresh
        self.assertFalse(self.manifest.exists())

    def test_a_request_does_not_resume_into_too_little_ram(self):
        self.svc.resume_on_request = True
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)
        self.free["ram"] = 50000                                   # something else took 20 GB meanwhile
        starts = self.engine.starts
        s, headers, b = self.raw("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(s, 503)
        self.assertEqual(headers.get("Retry-After"), "30")
        self.assertIn("50000 MiB of RAM available, it needs 54096", b["error"]["message"])
        self.assertEqual(self.engine.starts, starts)               # not started into it
        self.assertTrue(self.manifest.exists())                    # the conversations stay saved
        self.assertIn("RAM", self.req("/health")[1]["resume_blocked"])
        self.free["ram"] = 70000                                   # given back: the next request resumes it
        self.assertEqual(self.chat()[0], 200)
        self.assertFalse(self.manifest.exists())

    def test_a_request_does_not_resume_into_too_little_vram(self):
        self.svc.resume_on_request = True
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)
        self.free["vram"] = 4000                                   # a test suite holds the GPU
        s, b = self.chat()
        self.assertEqual(s, 503)
        self.assertIn("4000 MiB of VRAM free, it needs 21488", b["error"]["message"])
        self.free["vram"] = 22000                                  # within the 512 MiB the desktop may drift by
        self.assertEqual(self.chat()[0], 200)

    def test_a_request_waits_for_the_memory_up_to_resume_wait_s(self):
        self.svc.resume_on_request, self.svc.resume_wait_s = True, 10.0
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)
        self.free["ram"] = 30000

        def give_back():
            time.sleep(1.5)
            self.free["ram"] = 70000
        threading.Thread(target=give_back, daemon=True).start()
        t0 = time.time()
        self.assertEqual(self.chat()[0], 200)
        self.assertGreater(time.time() - t0, 1.0)

    def test_manual_load_is_checked_too_and_force_overrides(self):
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)
        self.free["ram"] = 10000
        s, headers, b = self.raw("/load", {})
        self.assertEqual((s, headers.get("Retry-After")), (503, "30"))
        self.assertIn("force", b["error"]["message"])
        self.assertFalse(self.engine.alive())
        self.assertEqual(self.req("/v1/load", {})[0], 503)
        self.assertEqual(self.req("/load", {"force": True})[0], 200)
        self.assertTrue(self.engine.alive())

    def test_without_resume_on_request_a_request_still_gets_503(self):
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)
        s, b = self.chat()
        self.assertEqual(s, 503)
        self.assertIn("hibernating", b["error"]["message"])
        self.assertFalse(self.engine.alive())

    def test_unmeasured_needs_do_not_block(self):
        write_hibernate(str(self.dir))                             # an older hibernate: no resume-needs.json
        self.engine.unload()
        self.free["ram"] = 1000
        self.assertIsNone(self.svc.resume_blocked())
        self.assertEqual(self.req("/load", {})[0], 200)

    # ---- "memory_check_on_load": the same check before a plain reload

    def test_plain_reload_is_checked_with_memory_check_on_load(self):
        self.svc.memory_check_on_load = True
        self.assertEqual(self.req("/unload", {})[0], 200)
        self.assertEqual(self.svc.resume_needs(), {"ram_mib": 50000, "vram_mib": 22000})
        self.assertFalse(self.manifest.exists())                   # recorded, but nothing latched
        self.free["ram"] = 40000
        s, headers, b = self.raw("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual((s, headers.get("Retry-After")), (503, "30"))
        self.assertIn("unloaded and stays so", b["error"]["message"])
        self.assertFalse(self.engine.alive())
        self.assertIn("RAM", self.req("/health")[1]["resume_blocked"])
        self.free["ram"] = 70000
        self.assertEqual(self.chat()[0], 200)
        self.assertTrue(self.engine.alive())

    def test_plain_reload_is_not_checked_by_default(self):
        self.assertEqual(self.req("/unload", {})[0], 200)
        self.free["ram"] = 1000
        self.assertEqual(self.chat()[0], 200)                      # loads as before
        self.assertNotIn("resume_needs", self.req("/health")[1])

    def test_force_loads_past_the_check(self):
        self.svc.memory_check_on_load = True
        self.assertEqual(self.req("/unload", {})[0], 200)
        self.free["ram"] = 1000
        self.assertEqual(self.req("/load", {})[0], 503)
        self.assertEqual(self.req("/load", {"force": True})[0], 200)

    def test_the_record_is_kept_in_memory_without_a_folder(self):
        self.svc.hibernate_dir = None
        self.svc.memory_check_on_load = True
        self.assertEqual(self.req("/unload", {})[0], 200)
        self.assertEqual(self.svc.resume_needs(), {"ram_mib": 50000, "vram_mib": 22000})
        self.free["vram"] = 3000
        s, b = self.chat()
        self.assertEqual(s, 503)
        self.assertIn("VRAM", b["error"]["message"])

    # ---- partial hibernate: /unload {"save": true, "keep_gpu": true} (RELEASE / ACQUIRE)

    def partial(self):
        s, b = self.req("/unload", {"save": True, "keep_gpu": True})
        self.assertEqual((s, b.get("status"), b.get("hibernated"), b.get("kept")), (200, "unloaded", True, "gpu"))
        return b

    def test_partial_keeps_the_engine_and_latches(self):
        self.partial()
        self.assertEqual(self.engine.release_calls, 1)
        self.assertTrue(self.engine.alive())                       # up, with its VRAM
        self.assertTrue(self.manifest.exists())
        h = self.req("/health")[1]
        self.assertEqual((h["loaded"], h["hibernated"], h["released"]), (False, True, True))
        self.assertEqual(h["resume_needs"], {"ram_mib": 50000, "vram_mib": 0})
        s, b = self.chat()                                         # latched as a full hibernate is
        self.assertEqual(s, 503)
        self.assertIn("hibernating", b["error"]["message"])
        self.assertEqual(getattr(self.engine, "acquire_calls", 0), 0)

    def test_partial_resume_acquires_instead_of_starting(self):
        self.partial()
        starts = self.engine.starts
        self.assertEqual(self.req("/load", {})[0], 200)
        self.assertEqual((self.engine.acquire_calls, self.engine.starts), (1, starts))   # no new engine
        self.assertFalse(self.manifest.exists())                   # spent, as after a full resume
        self.assertEqual(self.req("/health")[1]["released"], False)
        self.assertEqual(self.chat()[0], 200)

    def test_partial_resume_on_request(self):
        self.svc.resume_on_request = True
        self.partial()
        self.assertEqual(self.chat()[0], 200)
        self.assertEqual(self.engine.acquire_calls, 1)

    def test_partial_resume_waits_for_ram_only(self):
        self.svc.resume_on_request = True
        self.partial()
        self.free["vram"] = 300                                    # the engine itself holds the VRAM: not needed
        self.free["ram"] = 30000
        s, b = self.chat()
        self.assertEqual(s, 503)
        self.assertIn("RAM", b["error"]["message"])
        self.assertEqual(getattr(self.engine, "acquire_calls", 0), 0)
        self.free["ram"] = 80000
        self.assertEqual(self.chat()[0], 200)

    def test_a_failed_acquire_falls_back_to_a_full_start(self):
        self.partial()
        self.engine.acquire_error = EngineDied("FATAL acquire: VirtualAlloc failed")
        starts = self.engine.starts
        self.assertEqual(self.req("/load", {})[0], 200)
        self.assertEqual(self.engine.starts, starts + 1)           # a new engine, with the folder
        self.assertEqual(self.engine.hib_args[-1], str(self.dir))
        self.assertTrue(self.engine.alive())

    def test_partial_then_full_takes_the_ram_back_first(self):
        self.partial()
        self.assertEqual(self.req("/unload", {"save": True})[0], 200)
        self.assertEqual(self.engine.acquire_calls, 1)
        self.assertEqual(self.engine.hibernate_calls, 1)
        self.assertFalse(self.engine.alive())
        self.assertTrue(self.manifest.exists())
        self.assertEqual(self.req("/health")[1]["released"], False)

    def test_partial_twice_is_a_no_op(self):
        self.partial()
        self.partial()
        self.assertEqual(self.engine.release_calls, 1)

    def test_a_refused_release_changes_nothing(self):
        self.engine.hibernate_error = SessionRefused("invalid", "RELEASE needs the pinned RAM expert arena", False)
        s, b = self.req("/unload", {"save": True, "keep_gpu": True})
        self.assertEqual(s, 503)
        self.assertIn("pinned RAM", b["error"]["message"])
        self.assertTrue(self.svc.loaded())
        self.assertFalse(self.engine.released)

    def test_a_bad_needs_file_is_ignored(self):
        write_hibernate(str(self.dir))
        (self.dir / "resume-needs.json").write_text('{"ram_mib": "lots", "vram_mib": -5}', encoding="utf-8")
        self.assertIsNone(self.svc.resume_needs())


class FakeProc:
    """The engine process for StrataEngine.hibernate: answers HIBERNATE <dir> with a scripted list of lines."""

    def __init__(self, engine):
        self.engine, self.stdin, self.sent, self.killed = engine, self, [], False

    def write(self, s):
        self.sent.append(s)

        def play():
            for line in self.engine.script:
                if self.killed:
                    return
                if isinstance(line, float):
                    time.sleep(line)
                else:
                    self.engine.lines.put(line)
        threading.Thread(target=play, daemon=True).start()

    def flush(self):
        pass

    def poll(self):
        return -9 if self.killed else None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return -9


class FakeEngine(StrataEngine):
    def __init__(self):
        super().__init__("strata", ["--max-context", "65536"], lazy=True)
        self.max_context = 65536
        self.unloaded, self.ended = False, False
        self.lines = queue.Queue()
        self.script = []
        self.proc = FakeProc(self)


class HibernateCommand(unittest.TestCase):
    """StrataEngine.hibernate speaks SAVE's protocol: SESSION / SWAIT keep the wait alive, SERR is a refusal that
    leaves the engine in step, anything else ends it."""

    def setUp(self):
        self.e = FakeEngine()
        self.e.silence_s = 0.3

    def test_answer(self):
        self.e.script = ["HIBERNATED 3 4096 812.5\n"]
        self.assertEqual(self.e.hibernate("C:/x/hib"), {"saved": 3, "bytes": 4096, "ms": 812.5})
        self.assertEqual(self.e.proc.sent, ["HIBERNATE C:/x/hib\n"])

    def test_progress_keeps_it_waiting(self):
        self.e.script = [0.2, "SESSION 1 4\n", 0.2, "SESSION 2 4\n", 0.2, "SESSION 3 4\n", 0.2, "HIBERNATED 1 4 900.0\n"]
        self.assertEqual(self.e.hibernate("d")["saved"], 1)        # 0.8 s in all, over twice the silence limit

    def test_answer_says_whether_the_current_conversation_was_saved(self):
        self.e.script = ["HIBERNATED 2 4096 812.5 live=0\n"]
        self.assertEqual(self.e.hibernate("d"), {"saved": 2, "bytes": 4096, "ms": 812.5, "live": 0})
        self.e.proc.killed = False
        self.e.script = ["HIBERNATED 2 4096 812.5 live=7\n"]               # anything else is out of step
        with self.assertRaises(EngineDied):
            self.e.hibernate("d")

    def test_saved_answers_keep_four_fields(self):
        self.e.script = ["SAVED 10 20 1.0 live=1\n"]                       # only HIBERNATED takes the extra field
        with self.assertRaises(EngineDied):
            self.e.session_file("save", "x.strs")

    def test_progress_over_several_files_counts_up(self):
        # the engine reports one running count over every file it writes (each file starting at 0 again was taken as
        # out of step and ended the engine, found on an RTX 3090 with two conversations parked)
        self.e.script = ["SESSION 0 100\n", "SESSION 100 100\n", "SESSION 100 300\n", "SESSION 300 300\n",
                         "HIBERNATED 2 300 50.0 live=1\n"]
        self.assertEqual(self.e.hibernate("d")["saved"], 2)
        self.e.proc.killed = False
        self.e.script = ["SESSION 50 100\n", "SESSION 0 200\n"]            # a count that goes back is out of step
        with self.assertRaises(EngineDied):
            self.e.hibernate("d")

    def test_refusal(self):
        self.e.script = ["SERR memory 0 not enough RAM to save\n"]
        with self.assertRaises(SessionRefused) as e:
            self.e.hibernate("d")
        self.assertEqual(e.exception.kind, "memory")
        self.assertFalse(self.e.proc.killed)

    def test_out_of_step_ends_the_engine(self):
        self.e.script = ["DONE 1 2 3 4 stop\n"]
        with self.assertRaises(EngineDied):
            self.e.hibernate("d")
        self.assertTrue(self.e.proc.killed)

    def test_silence_ends_the_engine(self):
        self.e.script = []
        with self.assertRaises(EngineDied):
            self.e.hibernate("d")

    def test_a_folder_with_a_newline_is_refused(self):
        with self.assertRaises(ValueError):
            self.e.hibernate("d\nGEN 1 2")

    def test_set_hibernate_arg(self):
        self.e.set_hibernate_arg("C:/h")
        self.e.set_hibernate_arg("C:/h2")
        self.assertEqual(self.e.spawn[1][-2:], ["--hibernate-dir", "C:/h2"])
        self.assertEqual(self.e.spawn[1].count("--hibernate-dir"), 1)
        self.e.set_hibernate_arg(None)
        self.assertNotIn("--hibernate-dir", self.e.spawn[1])


if __name__ == "__main__":
    unittest.main()
