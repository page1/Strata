"""tools/strata_suspend.py - suspend Strata around a GPU/RAM-intensive run, and bring it back.

    python tools/strata_suspend.py run -- python my_simulation.py
    python tools/strata_suspend.py suspend          # then, when the run is over:
    python tools/strata_suspend.py resume
    python tools/strata_suspend.py resume --fresh   # discard the saved conversations
    python tools/strata_suspend.py status

`run` suspends (waits until Strata is idle, saves every parked conversation to the run config's
"hibernate_dir", ends the engine - its VRAM and RAM come straight back), runs your command, and
resumes in every case, including an error or Ctrl+C: the engine restarts and re-parks the
conversations, so long sessions continue where they were.  While suspended, requests to Strata
answer 503 instead of loading the model back over your run.

Needs "hibernate_dir" in strata-<model>.json (the run config) and a recent engine.
Port and API key are read from the run config; --port / --api-key override.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_PORT = 8080


def find_config(port: int | None) -> dict:
    """The run config (strata-<model>.json) in the repo root, preferring the one on `port`."""
    root = Path(__file__).resolve().parent.parent
    configs = sorted(root.glob("strata-*.json"))
    for c in configs:
        try:
            cfg = json.loads(c.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if port is None or int(cfg.get("port", DEFAULT_PORT)) == port:
            return cfg
    for c in configs:
        try:
            return json.loads(c.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
    return {}


def user_key_file() -> Path:
    """local: this account's key for its own Strata requests - one line in %APPDATA%\\Strata\\api-key (only this
    account can read its profile), for a server whose keys live in "api_key_file" rather than in the run config."""
    return Path(os.environ.get("APPDATA") or Path.home()) / "Strata" / "api-key"


def client_key(cfg: dict) -> str | None:
    """The key for our own requests: STRATA_API_KEY, else this account's key file, else the run config's api_key."""
    if os.environ.get("STRATA_API_KEY", "").strip():
        return os.environ["STRATA_API_KEY"].strip()
    try:
        key = user_key_file().read_text(encoding="utf-8-sig").strip()
        if key:
            return key
    except OSError:
        pass
    key = cfg.get("api_key")
    if isinstance(key, list):
        key = key[0] if key else None
    return key or None


def http(url: str, method: str = "GET", body: dict | None = None, api_key: str | None = None, timeout: float = 600.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        with e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            except ValueError:
                return e.code, {}
    except (urllib.error.URLError, OSError) as e:
        return 0, {"error": {"message": str(e)}}


SUSPEND_TIMEOUT_S = 3600.0   # saving the parked conversations runs at roughly 10 s per GiB


# ---- local: a record per wrapped job, and waking the agent session a job was cut off from: tools/strata_jobs.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_jobs as J  # noqa: E402
sys.path.append(str(Path(__file__).resolve().parents[1]))
from serve import winjob  # noqa: E402 - the kill-on-close job object (Windows); a no-op elsewhere


def default_job_log() -> str:
    """A new log file for a wrapped job, under the repo's data/job-logs/."""
    d = Path(__file__).resolve().parent.parent / "data" / "job-logs"
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f"{int(time.time() * 1000)}-{os.getpid()}.log")


def list_leases(leases_dir) -> int:
    """`strata_suspend.py jobs`: the leases in a folder, newest first - state (live/released/expired/stopped/
    overridden), exit, cmd, cwd, log."""
    d = Path(leases_dir)
    if not d.is_dir():
        print(f"no lease folder {d}")
        return 0
    rows = []
    for p in d.glob("*.json"):
        if p.name in ("checkpoint.json", "checkpoint.prev.json", "state.json"):
            continue
        try:
            l = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(l, dict) or not l.get("id"):
            continue
        rows.append((l.get("created") or 0, l.get("ended") or "live", l.get("exit"), l.get("cmd"), l.get("cwd"),
                     l.get("log")))
    if not rows:
        print("no leases")
        return 0
    for created, state, exit_, cmd, cwd, log in sorted(rows, reverse=True):
        ex = f" exit {exit_}" if exit_ is not None else ""
        print(f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(created)) if created else '?'}  [{state}]{ex}  "
              f"in {cwd}: {cmd or '(no command)'}  (log {log})")
    return 0


def hibernated_after_all(base: str, key: str | None, polls: int = 30) -> bool:
    """The suspend request ran out on this side: did the server finish hibernating anyway?"""
    for _ in range(polls):
        st, health = http(base + "/health", api_key=key, timeout=10)
        if st == 200 and health.get("hibernated"):
            return True
        if st == 200 and not health.get("loaded", True):
            return False                                   # ended without the latch: nothing was saved
        time.sleep(2.0)
    return False


def suspend(base: str, key: str | None, wait: int, keep_gpu: bool = False) -> int:
    """A plain suspend with no job (the `suspend` action): the model stays off until POST /load or a request."""
    deadline, held_note = time.time() + wait, 0.0
    while True:
        st, body = http(base + "/unload", "POST", {"save": True, **({"keep_gpu": True} if keep_gpu else {})}, key,
                        timeout=SUSPEND_TIMEOUT_S)
        if st == 0 and hibernated_after_all(base, key):
            st, body = 200, {"status": "unloaded", "hibernated": True}
        status = body.get("status")
        if st == 200 and status == "not loaded":           # off already (another job's suspend, an idle unload)
            print("Strata is off already (nothing to free)", flush=True)
            return 0
        if st == 200 and status == "unloaded":
            if keep_gpu:
                print("Strata suspended (RAM only): the experts' RAM is free, the GPU keeps the model and the current "
                      "conversation; resume (or POST /load) takes the RAM again in seconds", flush=True)
            else:
                print("Strata suspended: the engine ended, its VRAM and RAM are free; "
                      "resume (or POST /load) brings the conversations back", flush=True)
            if body.get("live_saved") is False:
                print("WARNING: the current conversation was not saved (not enough free RAM to park it); it is "
                      "read again from the start after resuming", flush=True)
            return 0
        if st == 409 and status == "busy" and time.time() < deadline:
            print("Strata is busy (a request runs or waits); retrying ...", flush=True)
            time.sleep(2.0)
            continue
        if st == 409 and status == "held":                   # another job runs in a RAM-only suspend: queue behind it
            now = time.time()
            if now >= held_note:
                print("waiting: another job holds Strata suspended RAM-only; this one needs the GPU too and starts "
                      "when that job ends", flush=True)
                held_note = now + 60.0
            time.sleep(5.0)
            deadline = time.time() + wait                    # the busy retries count from the end of the wait
            continue
        msg = str(body.get("error", {}).get("message", status))
        if keep_gpu and st == 503 and "4 KB pages" in msg:
            # the expert arena got large pages (after a fresh boot Windows often has the memory for them): its RAM
            # cannot be given back alone - free RAM and VRAM instead, which the job gets too
            print("Strata cannot give its RAM back alone this run (its expert arena uses large pages): freeing RAM "
                  "and VRAM instead", flush=True)
            keep_gpu = False
            continue
        print(f"suspend failed (HTTP {st}): {msg}")
        return 1


def pid_alive(pid: int) -> bool:
    """Unknown (no psutil, no access) counts as alive: its hold is still wanted."""
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:  # noqa: BLE001
        return True


def stop_tree(pid: int) -> None:
    """The job and every process it started (no psutil: the job only)."""
    try:
        import psutil
        p = psutil.Process(pid)
        procs = p.children(recursive=True) + [p]
        for x in procs:
            try:
                x.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(procs, timeout=10)
    except Exception:  # noqa: BLE001
        try:
            os.kill(pid, 9)
        except OSError:
            pass


# ---- local: leases (serve/leases.py): the wrapper's lock is the lease itself - its renewals keep the model off;
# if the wrapper dies the renewals stop and the server expires the lease (45 s).  No pid holds, no ack loops.

def session_from_env(env) -> dict | None:
    """Which agent session runs this job, from its environment: Claude Code carries CLAUDE_CODE_SESSION_ID;
    opencode's shell carries OPENCODE_SESSION_ID (its install is a compiled binary and its shell env shows only
    OPENCODE_PID - no other session-id variable found offline, so OPENCODE_SESSION_ID is the one to set)."""
    cid = env.get("CLAUDE_CODE_SESSION_ID")
    if cid:
        return {"agent": "claude", "id": cid}
    oid = env.get("OPENCODE_SESSION_ID")
    if oid:
        return {"agent": "opencode", "id": oid}
    return None


def session_at_creation(cmd: list, cwd: str, env: dict, db=J.OPENCODE_DB, now=None) -> dict | None:
    """The session to name in the lease, found ONCE when the lease is made (the waker never guesses later).  In an
    opencode shell (OPENCODE_PID, also when it inherited a Claude id from a Claude shell that started opencode):
    OPENCODE_SESSION_ID, else the bash call that runs this command, still fresh in opencode's database - never the
    Claude id here.  Otherwise Claude Code's own session id."""
    if env.get("OPENCODE_PID"):
        sid = env.get("OPENCODE_SESSION_ID") or J.find_opencode_session(
            {"cmd": cmd, "cwd": cwd, "started": now or time.time()}, db)
        return {"agent": "opencode", "id": sid} if sid else None
    return session_from_env(env)


def request_lease(base: str, key: str | None, body: dict, wait: float) -> tuple[int, dict]:
    """POST /lease, waiting the way the old suspend did.  409 "busy" (a request runs or waits): retry every 2 s until
    `wait` seconds have passed, then answer the refusal.  409 "held" (another job holds a RAM-only suspend and this
    one needs a full one): wait without a limit, noting it at most once a minute; the busy deadline counts again from
    the end of each such wait.  503 "4 KB pages" for a RAM-only lease: the lease is retried as a full one (body's
    "mode" changes).  Returns (status, response) of the last answer."""
    deadline, held_note = time.time() + wait, 0.0
    while True:
        st, resp = http(base + "/lease", "POST", body, key, timeout=SUSPEND_TIMEOUT_S)
        msg = str(resp.get("error", {}).get("message", ""))
        if body.get("mode") == "ram" and st == 503 and "4 KB pages" in msg:
            # the expert arena got large pages (after a fresh boot Windows often has the memory for them): its RAM
            # cannot be given back alone - the lease takes the full suspend instead, which the job gets too
            print("Strata cannot give its RAM back alone this run (its expert arena uses large pages): freeing RAM "
                  "and VRAM instead", flush=True)
            body["mode"] = "full"
            continue
        if st == 409 and resp.get("status") == "busy" and time.time() < deadline:
            print("Strata is busy (a request runs or waits); retrying ...", flush=True)
            time.sleep(2.0)
            continue
        if st == 409 and resp.get("status") == "held":        # queue behind the RAM-only job: no limit
            now = time.time()
            if now >= held_note:
                print("waiting: another job holds Strata suspended RAM-only; this one needs the GPU too and starts "
                      "when that job ends", flush=True)
                held_note = now + 60.0
            time.sleep(5.0)
            deadline = time.time() + wait
            continue
        return st, resp


class _ThisProcess:
    """This wrapper's own process, for winjob.contain: -1 is the current-process pseudo-handle (GetCurrentProcess)."""
    _handle = -1


_IN_JOB = []                                     # [True] once this wrapper is in its kill-on-close job

# POSIX: a killed wrapper never reaches the killpg at the end of run_leased, so its job would outlive it while
# its lease expires.  A reaper in its own session holds the read end of a pipe; the wrapper keeps the write
# end open for the whole run.  When the write end closes - the wrapper died, however it died, or ended
# normally (it writes a byte first to stand the reaper down) - the reaper kills the job's process group.
_REAPER_SRC = ("import os, signal, sys\n"
               "pid, fd = int(sys.argv[1]), int(sys.argv[2])\n"
               "try:\n"
               "    msg = os.read(fd, 1)\n"
               "except OSError:\n"
               "    msg = b''\n"
               "if not msg:\n"
               "    try:\n"
               "        os.killpg(pid, signal.SIGKILL)\n"
               "    except OSError:\n"
               "        pass\n")


def run_leased(base: str, key: str | None, cmd: list, cwd: str | None = None, keep_gpu: bool = False,
               session: dict | None = None, renew_s: float = 10.0, max_hold_s: float | None = None,
               env: dict | None = None, on_start=None, log: str | None = None, on_lease=None,
               wait: float = 120) -> int:
    """Run `cmd` while Strata is suspended for a lease: POST /lease, start the job with its output going to its own
    log (default: a new file under the repo's data/job-logs/; `--log` chooses one), renew every `renew_s` s from a
    thread, POST /lease/<id>/end when it exits.  The log is named in the lease (the waker's note reads it) and
    relayed to this process's stdout as it grows; this function waits for the relay's last read before returning, so
    the job's final output always reaches the agent.  A renewal answered 410 (the lock was overridden): the job's
    whole tree is killed and this returns 75.  Any other renewal failure (5xx, refused, timeout): the job keeps
    running and the next renewal tries again (the server may be restarting; it re-creates an unknown lease id).
    `max_hold_s`: a longer job is killed (whole tree), the lease ended, return 124.  `on_lease(lid)`: called once
    the lease is made.  `wait`: seconds to retry while Strata is busy (see request_lease).  Returns the job's exit
    code."""
    log = log or default_job_log()
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    body = {"cmd": " ".join(cmd), "cwd": cwd or os.getcwd(), "mode": "ram" if keep_gpu else "full",
            "session": session if session is not None else session_from_env(env if env is not None else os.environ),
            "log": log}
    st, resp = request_lease(base, key, body, wait)
    if st != 200:
        print(f"lease refused (HTTP {st}): {resp.get('error', {}).get('message')}; the job is not started", flush=True)
        return 1
    lid = resp.get("id")
    if on_lease is not None:
        on_lease(lid)

    stop_renew, lost, stopped = threading.Event(), threading.Event(), threading.Event()

    def renewer():
        while not stop_renew.wait(renew_s):
            st2, b2 = http(f"{base}/lease/{lid}/renew", "POST", {}, key, timeout=5)
            if st2 == 410:                       # the lock is lost: stopped by the user, or a forced load overrode it
                (stopped if "stopped" in str(b2.get("error", {}).get("message", "")) else lost).set()
                return
            # any other failure: keep the job, try again next time (the server may be restarting)

    # the wrapper itself goes in its job first, so the job (and a launcher's real interpreter) is born in it: when the
    # wrapper is killed the job goes with it, and its lease expires with nothing running (Windows). Elsewhere the job
    # gets its own process group, which is killed on the way out below (a killed wrapper cannot run that).
    if os.name == "nt":
        if not _IN_JOB and winjob.contain(_ThisProcess()):
            _IN_JOB.append(True)
        elif not _IN_JOB:
            print("WARNING: this wrapper could not be put in a job: if it is killed, its job keeps running", flush=True)
    with open(log, "ab") as job_out:
        proc = subprocess.Popen(J.program(cmd), cwd=cwd, stdout=job_out, stderr=subprocess.STDOUT,
                                **({} if os.name == "nt" else {"start_new_session": True}))
        reaper_w = None
        if os.name != "nt":
            r, reaper_w = os.pipe()
            try:
                subprocess.Popen([sys.executable, "-c", _REAPER_SRC, str(proc.pid), str(r)],
                                 start_new_session=True, pass_fds=(r,), close_fds=True)
            finally:
                os.close(r)
        if on_start is not None:
            on_start(proc)
        relay_th = threading.Thread(target=J.relay, args=(proc, log), daemon=True)
        relay_th.start()
        renew_th = threading.Thread(target=renewer, daemon=True)
        renew_th.start()
        started = time.time()
        try:
            while proc.poll() is None:
                if stopped.is_set():
                    stop_tree(proc.pid)
                    print("\nSTOPPED BY THE USER: this job's lease was ended with `strata_suspend.py stop` (Strata "
                          "needed it gone). Do not run it again unless the user asks you to.", flush=True)
                    code = 143
                    break
                if lost.is_set():
                    stop_tree(proc.pid)
                    code = 75
                    break
                if max_hold_s is not None and time.time() - started >= max_hold_s:
                    stop_tree(proc.pid)
                    code = 124
                    break
                time.sleep(0.1)
            else:
                code = proc.wait()
        except KeyboardInterrupt:
            stop_tree(proc.pid)
            code = 130
        stop_renew.set()
        renew_th.join(timeout=10)                      # no renewal after the end
    if os.name != "nt":                              # what the job started and left running, in its own group
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, 9)
        if reaper_w is not None:                     # ended normally: the group is already gone; stand the
            with contextlib.suppress(OSError):       # reaper down before closing the pipe, so it never
                os.write(reaper_w, b"x")             # kills a pgid that has been recycled since
            with contextlib.suppress(OSError):
                os.close(reaper_w)
    st3, resp3 = http(f"{base}/lease/{lid}/end", "POST", {"exit": code}, key, timeout=30)
    if st3 == 0:
        print(f"WARNING: could not reach the server to end the lease: {resp3.get('error', {}).get('message')}; "
              "the server expires it within seconds once the renewals stop", flush=True)
    relay_th.join(10)                                # the relay's last read before this process exits (6d's join)
    return code


def leases_supported(base: str, key: str | None) -> bool:
    """The server has the lease API (GET /leases answers 200 with its lease list)."""
    st, body = http(base + "/leases", api_key=key, timeout=10)
    return st == 200 and isinstance(body.get("leases"), list)


def lease_folder_from_config(cfg: dict) -> str | None:
    """The lease folder the server uses: "leases_dir" in the run config, else <hibernate_dir>/leases."""
    d = cfg.get("leases_dir")
    if d:
        return str(d)
    return str(Path(cfg["hibernate_dir"]) / "leases") if cfg.get("hibernate_dir") else None


def resume(base: str, key: str | None, fresh: bool) -> int:
    st, body = http(base + "/load", "POST", {"fresh": True} if fresh else {}, key)
    if st == 200:
        _, health = http(base + "/health")
        print(f"Strata resumed: loaded={health.get('loaded')} hibernated={health.get('hibernated')}")
        return 0
    print(f"resume failed (HTTP {st}): {body.get('error', {}).get('message')}")
    return 1


def hold_pid(base: str, key: str | None, pid: int, keep_gpu: bool = False, renew_s: float = 10.0,
             wait: float = 120) -> int:
    """Hold Strata for a process that is already running: a lease renewed while process `pid` lives (psutil, with its
    create time so a reused pid does not keep it), ended when the process exits.  -> 0."""
    try:
        import psutil
        started = psutil.Process(pid).create_time()
    except Exception:  # noqa: BLE001 - no psutil or gone already: hold on liveness alone
        started = None

    def alive():
        try:
            import psutil
            p = psutil.Process(pid)
            return started is None or abs(p.create_time() - started) < 1.0
        except Exception:  # noqa: BLE001
            return pid_alive(pid)

    body = {"cmd": f"hold --pid {pid}", "cwd": os.getcwd(), "mode": "ram" if keep_gpu else "full",
            "session": session_from_env(os.environ), "log": None}
    st, resp = request_lease(base, key, body, wait)
    if st != 200:
        print(f"lease refused (HTTP {st}): {resp.get('error', {}).get('message')}")
        return 1
    lid = resp.get("id")
    print(f"holding Strata for pid {pid} (lease {lid}): nothing loads the model until it ends", flush=True)
    while alive():
        st2, _b = http(f"{base}/lease/{lid}/renew", "POST", {}, key, timeout=5)
        if st2 == 410:                                   # stopped or overridden: the hold is no longer wanted
            break
        time.sleep(renew_s)
    http(f"{base}/lease/{lid}/end", "POST", {"exit": None}, key, timeout=30)
    print(f"pid {pid} ended: the lease is released and Strata resumes by itself", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["suspend", "resume", "run", "launch", "status", "jobs", "stop", "wake",
                                       "hold", "release", "restart"])
    ap.add_argument("--all", action="store_true", help="stop: every live lease (every wrapped job holding Strata)")
    ap.add_argument("--id", help="stop: the lease id to end (from `strata_suspend.py status`)")
    ap.add_argument("--port", type=int)
    ap.add_argument("--pid", type=int,
                    help="hold: suspend Strata until this running process ends (local)")
    ap.add_argument("--api-key")
    ap.add_argument("--fresh", action="store_true", help="resume: discard the saved conversations")
    ap.add_argument("--wait", type=int, default=120,
                    help="suspend/run/launch/hold: retry while Strata is busy up to this many seconds (a RAM-only "
                         "job's suspend is waited for without limit)")
    ap.add_argument("--keep-gpu", action="store_true",
                    help="suspend/run/launch/hold: give only the RAM back (the GPU keeps the model; resumes in seconds)")
    ap.add_argument("--max-hold", type=float, metavar="MINUTES",
                    help="run: a job running longer than this is killed (whole tree) and its lease ended "
                         "(exit 124); default: no limit")
    ap.add_argument("--lease", action="store_true",
                    help="run: accepted and ignored (the lease is the only lock now)")
    ap.add_argument("--log", help="launch/run: the job's log file (launch default: data/suspend-run-<time>.log; "
                                  "run default: next to the job record); launch passes it to the run it starts")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="suspend/launch: wait this many seconds first (local: lets an agent's reply finish)")
    ap.add_argument("--detach", action="store_true",
                    help="suspend/launch: return at once and start from a background process (local: for an agent on "
                         "Strata, e.g. an opencode command - see docs/LOCAL_PERFORMANCE.md)")
    ap.add_argument("--leases-dir", help="jobs: the lease folder to list (default: the run config's)")
    ap.add_argument("--opencode-url", default=J.OPENCODE_URL, help="wake: the opencode server (opencode serve)")
    ap.add_argument("--after", type=float, default=0.0, help="wake: wait this many seconds first")
    ap.add_argument("--dry-run", action="store_true", help="wake: print each note instead of sending it")
    ap.add_argument("cmd", nargs="*", help="run/launch: the command after -- to run while Strata is suspended")
    a = ap.parse_args()
    if a.action == "jobs":
        cfg = find_config(a.port)
        folder = a.leases_dir or lease_folder_from_config(cfg)
        if not folder:
            print("jobs: no lease folder (set \"leases_dir\" or \"hibernate_dir\" in the run config, or --leases-dir)")
            return 2
        return list_leases(folder)
    if a.action == "wake":
        if a.after > 0:
            time.sleep(a.after)
        cfg = find_config(a.port)
        port = a.port or int(cfg.get("port", DEFAULT_PORT))
        key = a.api_key or client_key(cfg)
        base = f"http://127.0.0.1:{port}"
        if leases_supported(base, key):                    # the lease waker is the only waker (rule 3: the lease
            folder = lease_folder_from_config(cfg)         # names the session, the checkpoint names the rest)
            if folder:
                def mark(entry):                           # the waker marks through the server, never a file itself
                    if entry.get("lease"):
                        st, _b = http(f"{base}/lease/{entry['lease']}/woken", "POST", {}, key, timeout=30)
                    else:
                        st, _b = http(base + "/lease/session-woken", "POST",
                                      {"agent": entry["agent"], "id": entry["id"]}, key, timeout=30)
                    return st == 200

                J.wake_leases(folder, mark, None, url=a.opencode_url, dry_run=a.dry_run)
        return 0
    cfg = find_config(a.port)
    port = a.port or int(cfg.get("port", DEFAULT_PORT))
    key = a.api_key or client_key(cfg)
    base = f"http://127.0.0.1:{port}"

    if a.action == "status":
        st, health = http(base + "/health", api_key=key)
        if st == 0:
            print(f"nothing answers on port {port}")
            return 1
        if health.get("released"):
            state = "suspended, RAM only (the GPU keeps the model; the next request resumes it in seconds)"
        elif health.get("hibernated"):
            state = "hibernated (RAM and VRAM free; the next request resumes it)"
        elif health.get("loaded"):
            state = "loaded"
        else:
            state = "not loaded (the next request loads it)"
        print(f"Strata on port {port}: {state}; model {health.get('model')}")
        if health.get("requests"):
            print(f"REQUESTS ({len(health['requests'])}):")
            for r in health["requests"]:
                print(f"  {J.describe_request(r)}")
        st2, ls = http(base + "/leases", api_key=key)
        for l in (ls.get("leases") or []) if st2 == 200 else []:
            who = f' - {l["session"]["agent"]} session "{l["session"]["id"]}"' if isinstance(l.get("session"), dict) \
                and l["session"].get("id") else ""
            print(f"  lease {l['id']}: {l.get('cmd') or '(no command)'}\n    in {l.get('cwd')}, "
                  f"{int((l.get('age_s') if l.get('age_s') is not None else time.time() - (l.get('created') or 0))) // 60} "
                  f"min{who}")
        return 0
    if a.action == "stop":
        if not a.id and not a.all:
            print("stop needs --id <lease id> (from `strata_suspend.py status`) or --all")
            return 2
        if a.all:
            st, body = http(base + "/lease/stop", "POST", {"all": True}, key, timeout=120)
            stopped = body.get("stopped") or [] if st == 200 else []
        else:
            st, body = http(f"{base}/lease/{a.id}/stop", "POST", {}, key, timeout=120)
            stopped = [body] if st == 200 else []
        if st != 200:
            print(f"stop failed (HTTP {st}): {body.get('error', {}).get('message')}")
            return 1
        for l in stopped:
            print(f"stopped: {l.get('cmd') or l.get('id')} (lease {l.get('id')})")
        if stopped:
            print("each job ends at its wrapper's next renewal (within ~10 s), and its own process tree goes with it")
        return 0
    if a.action == "hold":
        if not a.pid:
            print("hold needs --pid N: the job's process id (Strata resumes by itself when it ends)")
            return 2
        return hold_pid(base, key, a.pid, keep_gpu=a.keep_gpu, wait=a.wait)
    if a.action == "restart":
        # local: the server saves the conversations and ends; tools/strata_service.py (run it as a service or boot task) starts it again on
        # the code now in the folder - a deploy without an administrator. Needs the full key.
        st, body = http(base + "/restart", "POST", {}, key, timeout=SUSPEND_TIMEOUT_S)
        if st != 200:
            print(f"restart failed (HTTP {st}): {body.get('error', {}).get('message')}")
            return 1
        print(f"Strata restarting (conversations: {body.get('saved')}); waiting for it to answer ...", flush=True)
        time.sleep(3)
        for _ in range(120):
            s2, health = http(base + "/health", api_key=key, timeout=5)
            if s2 == 200:
                print("Strata is back" + (" (hibernated: the next request resumes the conversations)"
                                          if health.get("hibernated") else ""))
                return 0
            time.sleep(1)
        print("Strata did not answer within 2 min: is tools/strata_service.py running the server (as a service or "
              "boot task)? See the server's console log")
        return 1
    if a.action == "release":
        print("release is gone: the lease is the only lock - `stop` ends a job's lease, `resume` brings the model back")
        return 2
    if a.action == "launch" and a.detach:
        # local: as suspend --detach, for an agent on Strata that starts a long job: return now; a background process
        # waits for the agent's reply, then starts the job's console (its `run` makes the lease)
        cmd = a.cmd or []
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        if not cmd:
            print("launch needs a command: strata_suspend.py launch --detach -- python my_job.py")
            return 2
        log = Path(__file__).resolve().parent.parent / "data" / "strata-suspend-detached.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        child = [sys.executable, str(Path(__file__).resolve()), "launch", "--delay", str(a.delay or 8.0),
                 "--wait", str(a.wait), *(["--keep-gpu"] if a.keep_gpu else []),
                 *(["--port", str(a.port)] if a.port else []), *(["--api-key", a.api_key] if a.api_key else []),
                 *(["--log", a.log] if a.log else []), "--", *cmd]
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if sys.platform == "win32" else 0
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} launch in {a.delay or 8.0:.0f} s: {' '.join(cmd)}\n")
            subprocess.Popen(child, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             creationflags=flags, start_new_session=sys.platform != "win32", close_fds=True)
        print(f"In about {a.delay or 8.0:.0f} s the job starts in its own window with a lease on Strata "
              f"({'RAM only' if a.keep_gpu else 'RAM and VRAM'}); Strata stays suspended while the job runs - requests "
              f"wait - and resumes by itself when the job ends, however it ends. Log: {log}")
        return 0
    if a.action == "suspend" and a.detach:
        # local: an agent running ON Strata cannot suspend it inline - its next request (the reply that reports the
        # tool's result) would resume the model at once.  Suspend from a background process after the reply.
        log = Path(__file__).resolve().parent.parent / "data" / "strata-suspend-detached.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        child = [sys.executable, str(Path(__file__).resolve()), "suspend", "--delay", str(a.delay or 8.0),
                 "--wait", str(a.wait), *(["--keep-gpu"] if a.keep_gpu else []),
                 *(["--port", str(a.port)] if a.port else []), *(["--api-key", a.api_key] if a.api_key else [])]
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if sys.platform == "win32" else 0
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} {'free RAM only' if a.keep_gpu else 'free RAM and VRAM'}"
                    f" in {a.delay or 8.0:.0f} s\n")
            subprocess.Popen(child, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             creationflags=flags, start_new_session=sys.platform != "win32", close_fds=True)
        what = "its RAM (the GPU keeps the model)" if a.keep_gpu else "its RAM and VRAM"
        print(f"Strata will free {what} in about {a.delay or 8.0:.0f} s, once it is idle; the next request resumes "
              f"it. Log: {log}")
        return 0
    if a.action == "suspend":
        if a.delay > 0:
            time.sleep(a.delay)
        return suspend(base, key, a.wait, a.keep_gpu)
    if a.action == "resume":
        return resume(base, key, a.fresh)

    if a.action == "launch":
        cmd = a.cmd or []
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        if not cmd:
            print("launch needs a command: strata_suspend.py launch -- python my_simulation.py")
            return 2
        if a.delay > 0:
            time.sleep(a.delay)
        log = a.log or str(Path(__file__).resolve().parent.parent / "data" /
                           f"suspend-run-{time.strftime('%Y%m%d-%H%M%S')}.log")
        child = [sys.executable, str(Path(__file__).resolve()), "run", "--wait", str(a.wait),
                 *(["--keep-gpu"] if a.keep_gpu else []),
                 *(["--port", str(a.port)] if a.port else []), *(["--api-key", a.api_key] if a.api_key else []),
                 "--log", log, "--"] + cmd
        flags = subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, "CREATE_NEW_CONSOLE") else 0
        proc = subprocess.Popen(child, creationflags=flags)
        print(f"task started in its own console window (pid {proc.pid}); its `run` makes the lease on Strata")
        print(f"progress: the window, or tail the log: {log}")
        print(f"Strata resumes by itself when the task exits (exit code goes in the log); "
              f"check with: python tools/strata_suspend.py status")
        return 0

    cmd = a.cmd or []
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("run needs a command: strata_suspend.py run -- python my_simulation.py")
        return 2
    if http(base + "/health")[0] == 0:
        print(f"nothing answers on port {port}: Strata is not running, running the command as is")
        log = a.log or default_job_log()
        with open(log, "ab") as job_out:
            proc = subprocess.Popen(J.program(cmd), stdout=job_out, stderr=subprocess.STDOUT)
        return J.relay(proc, log)
    # the lease path (serve/leases.py): the job's renewals are the lock; no pid holds, no acknowledgement loops
    return run_leased(base, key, cmd, keep_gpu=a.keep_gpu, wait=a.wait,
                      max_hold_s=(a.max_hold * 60.0 if a.max_hold else None), log=a.log,
                      session=session_at_creation(cmd, os.getcwd(), os.environ), env={})


if __name__ == "__main__":
    sys.exit(main())
