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


def start_job(cmd: list, rec: Path | None) -> tuple[subprocess.Popen, str | None]:
    """The job through the shim (`_job`): its output goes to the record's log and it records its own exit code, so a
    killed wrapper neither takes the job down nor loses how it ended.  No record: the job directly, as before."""
    if rec is None:
        return subprocess.Popen(J.program(cmd)), None
    log = str(rec.with_suffix(".log"))
    return subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_job", "--record", str(rec),
                             "--log", log, "--", *cmd]), log


def finish_job(rec: Path | None, proc: subprocess.Popen | None, code) -> None:
    """The wrapper delivered the result (or the job was interrupted here, in front of the agent)."""
    if rec is None:
        return
    r = J.read(rec) or {}
    if r.get("state") == J.STOPPED:
        print("\nSTOPPED BY THE USER: this job was ended with `strata_suspend.py stop` (Strata needed it gone). Do not "
              "run it again unless the user asks you to.", flush=True)
        J.update(rec, delivered=True)
        return
    extra = {}
    if r.get("state") == "running" and (proc is None or proc.poll() is not None):   # the shim itself was stopped
        extra = {"state": "done", "exit": code, "ended": time.time()}
    J.update(rec, delivered=True, **extra)


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


def suspend(base: str, key: str | None, wait: int, keep_gpu: bool = False, hold_pids: list | None = None) -> int:
    """hold_pids (local): nothing resumes Strata while one of these processes runs - no agent request, no plain POST
    /load - and the server resumes it by itself when the last one ends, however it ends."""
    deadline, held_note = time.time() + wait, 0.0
    while True:
        st, body = http(base + "/unload", "POST", {"save": True, **({"keep_gpu": True} if keep_gpu else {}),
                                                   **({"hold_pids": hold_pids} if hold_pids else {})}, key,
                        timeout=SUSPEND_TIMEOUT_S)
        if st == 0 and hibernated_after_all(base, key):
            st, body = 200, {"status": "unloaded", "hibernated": True}
        status = body.get("status")
        if st == 200 and status == "not loaded":           # off already (another job's suspend, an idle unload)
            if body.get("held_by"):
                print(f"Strata is off already; held for pid {', '.join(map(str, body['held_by']))}: nothing loads it "
                      "until they all end", flush=True)
            else:
                print("Strata is off already (nothing to free)", flush=True)
            return 0
        if st == 200 and status == "unloaded":
            if keep_gpu:
                print("Strata suspended (RAM only): the experts' RAM is free, the GPU keeps the model and the current "
                      "conversation; resume (or POST /load) takes the RAM again in seconds", flush=True)
            else:
                print("Strata suspended: the engine ended, its VRAM and RAM are free; "
                      "resume (or POST /load) brings the conversations back", flush=True)
            if body.get("held_by"):
                print(f"held for pid {', '.join(map(str, body['held_by']))}: nothing resumes Strata until "
                      f"{'it ends' if len(body['held_by']) == 1 else 'they all end'}; then the server resumes it",
                      flush=True)
            elif hold_pids:
                print("WARNING: the hold was not set (the process had ended?): any request can resume Strata",
                      flush=True)
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
                print(f"waiting: another job (pid {', '.join(map(str, body.get('held_by') or []))}) holds Strata "
                      "suspended RAM-only; this one needs the GPU too and starts when that job ends", flush=True)
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


def hold(base: str, key: str | None, pids: list) -> bool:
    st, body = http(base + "/hold", "POST", {"hold_pids": pids}, key, timeout=30)
    if st == 400 and "no such process" in str(body.get("error", {}).get("message")):
        return False                                     # the job ended already: the wrapper's own hold covers it
    if st != 200:
        print(f"WARNING: could not hold Strata for pid {pids} (HTTP {st}): {body.get('error', {}).get('message')}")
    return st == 200


def job_pids(pid: int) -> list:
    """The job and the processes it has started so far (on Windows a venv's python.exe is a launcher that runs the
    real interpreter as its child: holding both keeps the hold while either lives).  Without psutil: the job only."""
    time.sleep(1.0)                                      # let a launcher start its child
    try:
        import psutil
        return [pid] + [c.pid for c in psutil.Process(pid).children(recursive=True)]
    except Exception:  # noqa: BLE001 - no psutil, or the job already ended
        return [pid]


def release(base: str, key: str | None) -> None:
    """Every hold ends (an explicit resume)."""
    http(base + "/hold", "POST", {"release": True}, key, timeout=30)


def release_own(base: str, key: str | None, pids: list) -> None:
    """This job is done: only its own holders are released (other jobs may still run in the suspend).  The server
    resumes the model when the last holder is gone - and only if it was loaded when the first job suspended it."""
    st, body = http(base + "/hold", "POST", {"release_pids": pids}, key, timeout=SUSPEND_TIMEOUT_S)
    if st != 200:
        print(f"WARNING: could not release the hold (HTTP {st}): {body.get('error', {}).get('message')}; the server "
              "resumes Strata by itself within seconds once the job's processes have ended", flush=True)
    elif body.get("held_by"):
        print(f"Strata stays suspended: another job still holds it (pid {', '.join(map(str, body['held_by']))}); "
              "the last one to end resumes it", flush=True)
    elif body.get("resumed"):
        print("Strata resumed: the conversations are re-parked", flush=True)
    elif body.get("waiting"):
        print(f"Strata resumes once memory is free: {body['waiting']}", flush=True)
    else:
        print("Strata was not loaded before this job: left as it was (the next request loads it)", flush=True)


def resume(base: str, key: str | None, fresh: bool) -> int:
    release(base, key)                                   # an explicit resume ends any hold
    st, body = http(base + "/load", "POST", {"fresh": True} if fresh else {}, key)
    if st == 200:
        _, health = http(base + "/health")
        print(f"Strata resumed: loaded={health.get('loaded')} hibernated={health.get('hibernated')}")
        return 0
    print(f"resume failed (HTTP {st}): {body.get('error', {}).get('message')}")
    return 1


def run_and_resume(a) -> int:
    """The child side of `launch`: run the command in this (new) console, teeing its output to the
    log, then resume Strata - whatever the command's exit code.  The parent has already suspended."""
    log = Path(a.log) if a.log else Path(__file__).resolve().parent.parent / "data" / \
        f"suspend-run-{time.strftime('%Y%m%d-%H%M%S')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = a.cmd or []
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    cfg = find_config(a.port)
    port = a.port or int(cfg.get("port", DEFAULT_PORT))
    key = a.api_key or cfg.get("api_key")
    base = f"http://127.0.0.1:{port}"
    log.write_text(f"# strata_suspend launch: {' '.join(cmd)}\n# started {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                   encoding="utf-8")
    me = [os.getpid()]
    try:                                                           # a venv's python.exe is a launcher: launch held it
        import psutil
        parent = psutil.Process().parent()
        if parent and "_run_and_resume" in " ".join(parent.cmdline()):
            me.append(parent.pid)
    except Exception:  # noqa: BLE001 - no psutil: the watcher frees the launcher's hold when it ends
        pass
    hold(base, key, [os.getpid()])                                 # this interpreter itself (launch held the launcher)
    rec = J.start(cmd, log=str(log))
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_job", "--log", str(log),   # appends the
                             *(["--record", str(rec)] if rec else []), "--", *cmd])                    # job's output
    hold(base, key, [proc.pid])                                    # at once: a kill of this console cannot orphan it
    J.update(rec, job=[{"pid": proc.pid, "started": J.proc_started(proc.pid)}])
    held = job_pids(proc.pid)
    hold(base, key, held)                                          # + its children (a venv launcher's interpreter)
    J.update(rec, job=[{"pid": x, "started": J.proc_started(x)} for x in held])
    code = J.relay(proc, str(log))                                 # the new console shows progress live
    finish_job(rec, proc, code)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        release_own(base, key, me)                                 # only this console: live leftovers hold on
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n# exit code {code}\n# " + out.getvalue())
    print(f"\n[done: exit {code}] " + out.getvalue().strip())
    print(f"log: {log}")
    return code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["suspend", "resume", "run", "launch", "status", "jobs", "stop", "wake",
                                       "hold", "release", "dismiss", "_run_and_resume", "_job"])
    ap.add_argument("--all", action="store_true", help="stop: every wrapped job that holds Strata")
    ap.add_argument("--port", type=int)
    ap.add_argument("--pid", type=int, action="append",
                    help="hold: suspend Strata until this running process ends (repeatable; local)")
    ap.add_argument("--api-key")
    ap.add_argument("--fresh", action="store_true", help="resume: discard the saved conversations")
    ap.add_argument("--wait", type=int, default=120, help="suspend: retry while busy up to this many seconds")
    ap.add_argument("--keep-gpu", action="store_true",
                    help="suspend/run/launch: give only the RAM back (the GPU keeps the model; resumes in seconds)")
    ap.add_argument("--log", help="launch: the log file (default: data/suspend-run-<time>.log)")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="suspend: wait this many seconds first (local: lets an agent's reply finish)")
    ap.add_argument("--detach", action="store_true",
                    help="suspend: return at once and suspend from a background process (local: for an agent on "
                         "Strata, e.g. an opencode command - see docs/LOCAL_PERFORMANCE.md)")
    ap.add_argument("--record", action="append",                    # _job: the job's record
                    help="dismiss: a cut-off job's record (its file name from `jobs`; repeatable)")
    ap.add_argument("--why", default="", help="dismiss: why it is not needed (kept in the record)")
    ap.add_argument("--opencode-url", default=J.OPENCODE_URL, help="wake: the opencode server (opencode serve)")
    ap.add_argument("--after", type=float, default=0.0, help="wake: wait this many seconds first")
    ap.add_argument("--dry-run", action="store_true", help="wake: print each note instead of sending it")
    ap.add_argument("--strata-last-alive", type=float, help=argparse.SUPPRESS)   # wake: the previous run's heartbeat
    ap.add_argument("cmd", nargs="*", help="run/launch: the command after -- to run while Strata is suspended")
    a = ap.parse_args()
    if a.action == "_run_and_resume":
        return run_and_resume(a)
    if a.action == "_job":
        cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
        return J.exec_job(Path(a.record[0]) if a.record else None, cmd, a.log)
    if a.action == "jobs":
        return J.list_jobs()
    if a.action == "dismiss":
        if not a.record:
            print("dismiss needs --record NAME (a CUT OFF job's record, as `strata_suspend.py jobs` names it)")
            return 2
        return 0 if J.dismiss(a.record, a.why) == len(a.record) else 1
    if a.action == "wake":
        if a.after > 0:
            time.sleep(a.after)
        J.wake_all(a.opencode_url, dry_run=a.dry_run, strata_alive=a.strata_last_alive)
        return 0
    cfg = find_config(a.port)
    port = a.port or int(cfg.get("port", DEFAULT_PORT))
    key = a.api_key or cfg.get("api_key")
    base = f"http://127.0.0.1:{port}"

    if a.action == "status":
        st, health = http(base + "/health")
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
        if health.get("held_by"):
            print("LOCKED: requests wait until these jobs end, then the model comes back by itself")
            for h in J.describe_holders(health["held_by"]):
                if h.get("wrapped"):
                    who = f' - {h.get("agent")} session "{h.get("title") or h.get("session")}"' if h.get("session") \
                        else ""
                    print(f"  pid {', '.join(map(str, h['pids']))}: {h['command']}\n    in {h['folder']}, running "
                          f"{h['running_s'] // 60} min{who}")
                else:
                    print(f"  pid {', '.join(map(str, h['pids']))}: {h.get('command') or '(gone)'}")
                print(f"    -> {J.advice(h)}")
        recs = J.records()
        cut = [r for _, r in recs if J.state(r) in (J.CUT_OFF, J.UNDELIVERED) and not r.get("woken")
               and not J.not_needed(r, recs)]
        if cut:
            print(f"{len(cut)} wrapped job(s) did not return their result and their session was not told yet (a crash, "
                  "a power cut, Task Manager): see strata_suspend.py jobs; the next Strata start tells their sessions "
                  "(strata_suspend.py dismiss --record NAME if one is no longer needed)")
        return 0
    if a.action == "stop":
        if not a.pid and not a.all:
            print("stop needs --pid N (a holder from `strata_suspend.py status`) or --all")
            return 2
        st, body = http(base + "/hold", "POST", {"stop": "all" if a.all else a.pid}, key, timeout=120)
        if st != 200:
            print(f"stop failed (HTTP {st}): {body.get('error', {}).get('message')}")
            return 1
        for h in body.get("stopped") or []:
            if h.get("stopped"):
                print(f"stopped: {h['command']} (pid {', '.join(map(str, h['pids']))})")
                if h.get("loop"):
                    print(f"  WARNING: {J.advice(h)}")
            else:
                print(f"left alone: pid {', '.join(map(str, h['pids']))} - {J.advice(h)}")
        left = body.get("held_by") or []
        print("Strata is coming back" if not left else f"still held by pid {', '.join(map(str, left))}")
        return 0
    if a.action == "hold":
        if not a.pid:
            print("hold needs --pid N: the job's process id (Strata resumes by itself when it ends)")
            return 2
        return suspend(base, key, a.wait, a.keep_gpu, hold_pids=a.pid)
    if a.action == "release":
        release(base, key)
        print("hold released (the model stays as it is; the next request, or resume, brings it back)")
        return 0
    if a.action == "launch" and a.detach:
        # local: as suspend --detach, for an agent on Strata that starts a long job: return now; a background process
        # waits for the agent's reply, then suspends Strata held by the job and opens the job's console
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
        print(f"In about {a.delay or 8.0:.0f} s Strata suspends ({'RAM only' if a.keep_gpu else 'RAM and VRAM'}) and "
              f"the job starts in its own window; Strata stays suspended while the job runs - requests get 503 - and "
              f"resumes by itself when the job ends, however it ends. Log: {log}")
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
        if suspend(base, key, a.wait, a.keep_gpu, hold_pids=[os.getpid()]) != 0:
            return 1
        log = a.log or str(Path(__file__).resolve().parent.parent / "data" /
                           f"suspend-run-{time.strftime('%Y%m%d-%H%M%S')}.log")
        child = [sys.executable, str(Path(__file__).resolve()), "_run_and_resume",
                 "--log", log]
        if a.port:
            child += ["--port", str(a.port)]
        if a.api_key:
            child += ["--api-key", a.api_key]
        child += ["--"] + cmd
        flags = subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, "CREATE_NEW_CONSOLE") else 0
        proc = subprocess.Popen(child, creationflags=flags)
        hold(base, key, [proc.pid])                      # the console window holds it now (this process ends)
        print(f"task started in its own console window (pid {proc.pid})")
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
    J.notes_here()
    rec = J.start(cmd)
    if http(base + "/health")[0] == 0:
        print(f"nothing answers on port {port}: Strata is not running, running the command as is")
        proc, code = None, None
        try:
            proc, log = start_job(cmd, rec)
            J.update(rec, job=[{"pid": proc.pid, "started": J.proc_started(proc.pid)}])
            J.update(rec, job=[{"pid": x, "started": J.proc_started(x)} for x in job_pids(proc.pid)])
            code = J.relay(proc, log) if log else proc.wait()
        except KeyboardInterrupt:
            code = 130
        finally:
            finish_job(rec, proc, code)
        return code
    if suspend(base, key, a.wait, a.keep_gpu, hold_pids=[os.getpid()]) != 0:
        J.update(rec, state="not started", ended=time.time(), delivered=True)
        return 1
    proc, code = None, None
    try:
        proc, log = start_job(cmd, rec)
        hold(base, key, [proc.pid])                              # at once: if this wrapper is killed, the job holds it
        J.update(rec, job=[{"pid": proc.pid, "started": J.proc_started(proc.pid)}])
        held = job_pids(proc.pid)
        hold(base, key, held)                                    # + its children (a venv launcher's interpreter)
        J.update(rec, job=[{"pid": x, "started": J.proc_started(x)} for x in held])
        code = J.relay(proc, log) if log else proc.wait()
    except KeyboardInterrupt:
        code = 130
    finally:
        finish_job(rec, proc, code)
        release_own(base, key, [os.getpid()])                  # only this wrapper (another job may still run, and
        #                                                        a job still alive after Ctrl+C keeps holding)
    return code


if __name__ == "__main__":
    sys.exit(main())
