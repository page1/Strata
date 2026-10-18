"""tools/strata_jobs.py - local: a record per job that tools/strata_suspend.py wraps, and waking the agent session a
job was cut off from.

Every wrapped job gets data/suspend-jobs/<ms>-<pid>.json plus <ms>-<pid>.log (its output, written by the job itself,
so it survives the wrapper).  A small shim (`strata_suspend.py _job`) runs the job and records its exit code; the
wrapper relays the log to the agent and then marks the result delivered.  After a power cut, a crash or Task Manager:

- "cut off": the job's processes are gone without an exit recorded (the PC restarted, or they were killed);
- "finished, result not delivered": the job ended, but the wrapper that relays its result to the agent did not.

`strata_suspend.py jobs` lists them.  `strata_suspend.py wake` (the Strata server runs it a few seconds after it
starts, with "wake_cut_off_jobs": true) finds each such job's opencode session - the bash call that ran it, in
opencode's session database - and sends that session a message through the opencode server (`opencode run --attach
--session`): what was cut off and why, the end of its output, and the files it wrote.  The session takes its turn on
the opencode server, so `opencode attach` joins it later.  A Claude Code session that ran the job (its transcript has
the Bash call) gets the same note through `claude --resume <id> -p`; its next look at that session shows the turn.

`wake` also tells sessions that were cut off in the middle of a turn without a wrapped job: after a PC restart, the
opencode and Claude Code sessions whose last turn never finished (active in the hours before the PC went down - the
last thing any agent wrote before this boot, however long the outage); and the opencode sessions whose request failed
because Strata was not running (after the previous Strata run's last heartbeat).  Once each.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

JOBS_DIR = Path(os.environ.get("STRATA_SUSPEND_JOBS_DIR")                  # (tests point it elsewhere)
                or Path(__file__).resolve().parent.parent / "data" / "suspend-jobs")
JOBS_KEEP_S = 7 * 86400


def jobs_dir() -> Path:
    """The records' folder, looked up at each use: STRATA_SUSPEND_JOBS_DIR when set (a server that imported this
    module earlier must still follow it), else JOBS_DIR."""
    env = os.environ.get("STRATA_SUSPEND_JOBS_DIR")
    return Path(env) if env else JOBS_DIR
WAKE_TRIES = 3                                   # a job whose session is not found is tried at this many starts
OPENCODE_URL = "http://127.0.0.1:4096"
OPENCODE_DB = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"       # Claude Code's transcripts: <project>/<session id>.jsonl
LOG_TAIL_LINES, LOG_TAIL_CHARS = 60, 6000
CRASH_WINDOW_S = 7200                            # no boot history (not Windows): active this long before it went down
WAITS_FOR_USER = {"question", "AskUserQuestion", "ExitPlanMode"}   # a turn paused on the user, not cut off
STRATA_DOWN_WINDOW_S = 3600                      # no Strata heartbeat known: a failed request this recent counts
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", ".venv1", "venv", ".idea", ".mypy_cache",
             ".pytest_cache", ".ruff_cache", "build", "dist"}


# ---- processes

def proc_started(pid: int) -> float | None:
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:  # noqa: BLE001 - gone, or no psutil
        return None


def pid_exists(pid: int) -> bool:
    """Without psutil.  Never os.kill(pid, 0) on Windows: that terminates the process."""
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))      # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259           # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def alive(pid: int, started) -> bool:
    """The same process (pid + start time).  Unknown start times (no psutil): the pid still exists - so a running
    job is never reported as cut off."""
    t = proc_started(pid)
    if t is not None and started is not None:
        return abs(t - float(started)) < 1.0
    return pid_exists(pid)


def boot_time() -> float | None:
    try:
        import psutil
        return psutil.boot_time()
    except Exception:  # noqa: BLE001
        return None


# ---- records

def _write(path: Path, rec: dict) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")   # per writer: the wrapper and the shim may both write
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    for attempt in range(20):                        # Windows: no replace while a reader has it open
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def start(cmd: list, log: str | None = None) -> Path | None:
    """Record a job before it starts (the caller is its wrapper).  Never fails the job: None if it cannot write."""
    try:
        jobs_dir().mkdir(parents=True, exist_ok=True)
        now = time.time()
        for old in jobs_dir().glob("*.json"):              # this tool's own, over a week old - but a cut-off job whose
            try:                                         # session was not told yet stays (an outage of weeks)
                if now - old.stat().st_mtime > JOBS_KEEP_S and old.name != "interrupted.json":
                    r = read(old) or {}
                    if r.get("woken") or state(r) not in (CUT_OFF, UNDELIVERED):
                        for f in (old, old.with_suffix(".log"), old.with_suffix(".wake.log")):
                            f.unlink(missing_ok=True)
            except OSError:
                pass
        path = jobs_dir() / f"{int(now * 1000)}-{os.getpid()}.json"
        rec = {"state": "running", "cmd": cmd, "cwd": os.getcwd(), "started": now,
               "wrapper": {"pid": os.getpid(), "started": proc_started(os.getpid())}, "job": [],
               "log": log or str(path.with_suffix(".log"))}
        if os.environ.get("CLAUDE_CODE_SESSION_ID"):
            rec["claude_session"] = os.environ["CLAUDE_CODE_SESSION_ID"]
        _write(path, rec)
        return path
    except OSError:
        return None


def read(path: Path) -> dict | None:
    for _ in range(20):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            time.sleep(0.05)
        except (OSError, ValueError):
            return None
    return None


def update(path: Path | None, **fields) -> None:
    if path is None:
        return
    try:
        rec = read(path)
        if rec is None:
            return
        rec.update(fields)
        _write(path, rec)
    except OSError:
        pass


def records() -> list[tuple[Path, dict]]:
    out = []
    for f in sorted(jobs_dir().glob("*-*.json")) if jobs_dir().exists() else []:   # <ms>-<pid>.json, not the ledger
        rec = read(f)
        if rec:
            out.append((f, rec))
    return out


CUT_OFF, UNDELIVERED = "cut off", "finished, result not delivered"


def state(rec: dict) -> str:
    """"running", "running (wrapper gone)", "done", CUT_OFF, UNDELIVERED or "not started"."""
    wrapper = rec.get("wrapper") or {}
    wrapper_alive = bool(wrapper.get("pid")) and alive(wrapper["pid"], wrapper.get("started"))
    if rec.get("state") == "done":
        return "done" if rec.get("delivered") or wrapper_alive else UNDELIVERED
    if rec.get("state") != "running":
        return rec.get("state") or "?"
    if wrapper_alive:
        return "running"
    if any(alive(j["pid"], j.get("started")) for j in rec.get("job") or []):
        return "running (wrapper gone)"
    return CUT_OFF


def cause(rec: dict) -> str:
    """Why a job did not get its result back to the agent, from what can be checked."""
    st, started = state(rec), rec.get("started") or 0
    if st == UNDELIVERED:
        return (f"The job itself finished (exit code {rec.get('exit')}) at {_when(rec.get('ended'))}, but the process "
                "that relays its result to you was ended first, so the result never reached you.")
    boot = boot_time()
    if boot and boot > started:
        return (f"Windows started at {_when(boot)}, after this job began at {_when(started)}: the PC was restarted "
                "or went down (a restart, a power cut or a crash) and every process stopped with it - this job, its "
                "wrapper and Strata. The failure was system-wide; it was not caused by the job's code.")
    return ("The PC did not restart, but the job's processes were ended from outside without reporting an exit "
            "(killed - for example Task Manager's End process tree - or a crash of the process that ran it). That is "
            "most likely not the job's code, but check its output.")


def _when(t) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) if t else "an unknown time"


# ---- the shim and the relay

def program(cmd: list) -> list:
    """The command with its program found as a shell finds it.  Windows does not start a relative path written with
    forward slashes (`.venv/Scripts/python.exe`: "cannot find the file"), which agents write; `shutil.which`
    finds it (and a bare name on PATH, with its extension).  Unchanged when it finds nothing: Popen then says why."""
    if not cmd:
        return cmd
    found = shutil.which(cmd[0])
    return [found, *cmd[1:]] if found else list(cmd)


def exec_job(path: Path | None, cmd: list, log: str) -> int:
    """`strata_suspend.py _job`: run the job with its output going to the log (not through the wrapper, so a killed
    wrapper does not take the job down with a broken pipe), and record its exit code - also when the wrapper is gone."""
    with open(log, "ab") as out:
        try:
            proc = subprocess.Popen(program(cmd), stdout=out, stderr=subprocess.STDOUT)
        except OSError as e:
            out.write(f"\n! could not start the command: {e}\n".encode())
            update(path, state="done", exit=127, ended=time.time())
            return 127
        code = proc.wait()
    if path is not None and (read(path) or {}).get("state") != STOPPED:   # a stop by the user stays recorded
        update(path, state="done", exit=code, ended=time.time())
    tell_if_undelivered(path)
    return code


def tell_if_undelivered(path: Path | None) -> None:
    """The job has ended: if its wrapper is gone, nobody will relay the result to the agent - tell its session now
    (not at the next Strata start).  A live wrapper delivers it itself (the record then reads "done")."""
    if path is None:
        return
    rec = read(path) or {}
    if state(rec) == UNDELIVERED and not rec.get("woken"):
        import contextlib
        with open(path.with_suffix(".wake.log"), "a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            print(f"# {_when(time.time())} the job ended; its wrapper is gone: telling its session", flush=True)
            wake(only=path)


def relay(proc: subprocess.Popen, log: str) -> int:
    """The wrapper: copy the job's log to this process's output as it grows, until the shim ends."""
    pos, out = 0, getattr(sys.stdout, "buffer", None)

    def pump():
        nonlocal pos
        try:
            with open(log, "rb") as f:
                f.seek(pos)
                chunk = f.read()
        except OSError:
            return
        if chunk:
            pos += len(chunk)
            if out is not None:
                out.write(chunk)
                out.flush()
            else:
                sys.stdout.write(chunk.decode("utf-8", "replace"))
                sys.stdout.flush()
    while proc.poll() is None:
        pump()
        time.sleep(0.2)
    pump()
    return proc.returncode


# ---- telling the agent

def _same_dir(a, b) -> bool:
    return bool(a and b) and os.path.normcase(os.path.normpath(str(a))) == os.path.normcase(os.path.normpath(str(b)))


def run_again(rec: dict, recs: list | None = None) -> dict | None:
    """The later record of the same command in the same folder, if any: the agent already ran the job again (it was
    told by the next job's note, or saw the failure), so nobody is woken for this one."""
    for _, r in (recs if recs is not None else records()):
        if (r.get("started") or 0) > (rec.get("started") or 0) and r.get("cmd") == rec.get("cmd") and \
                _same_dir(r.get("cwd"), rec.get("cwd")):
            return r
    return None


def not_needed(rec: dict, recs: list | None = None) -> str | None:
    """Why nobody is woken for this cut-off job: dismissed (`strata_suspend.py dismiss`) or run again; else None."""
    if rec.get("dismissed"):
        return f"dismissed {_when(rec['dismissed'])[5:16]}" + (f": {rec['dismissed_why']}" if rec.get("dismissed_why") else "")
    again = run_again(rec, recs)
    return f"run again {_when(again.get('started'))[11:16]}" if again else None


def dismiss(names: list, why: str = "") -> int:
    """`strata_suspend.py dismiss --record NAME`: a cut-off job whose result is not needed any more (the agent moved
    on, or the user did) - nobody is woken for it.  NAME: the record's file name (or path) as `jobs` shows its log.
    -> how many were dismissed."""
    n = 0
    for name in names:
        f = Path(name)
        f = f if f.is_absolute() else jobs_dir() / f.name
        f = f.with_suffix(".json")
        rec = read(f)
        if not rec:
            print(f"dismiss: no record {f}")
            continue
        if state(rec) not in (CUT_OFF, UNDELIVERED):
            print(f"dismiss: {f.name} is {state(rec)}, not cut off - left alone")
            continue
        update(f, dismissed=time.time(), dismissed_why=why)
        print(f"dismissed {f.name}: {subprocess.list2cmdline(rec.get('cmd') or [])} - nobody is woken for it")
        n += 1
    return n


def _tokens(cmd: list) -> list[str]:
    """What must appear in the agent's command line: each argument (paths by their last part)."""
    out = []
    for x in cmd:
        x = str(x).replace("\\", "/")
        out.append((x.rsplit("/", 1)[-1] if "/" in x else x).lower())
    return [t for t in out if t]


def find_opencode_session(rec: dict, db: Path = OPENCODE_DB) -> str | None:
    """The opencode session whose bash call ran this wrapped job: a strata_suspend call with the job's arguments,
    in the job's folder, started within a few minutes before the job (read-only)."""
    if not db.exists():
        return None
    started = float(rec.get("started") or 0)
    want = _tokens(rec.get("cmd") or [])
    try:
        con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return None
    try:
        rows = con.execute(
            "SELECT p.session_id, p.time_created, json_extract(p.data, '$.state.input.command'), "
            "json_extract(p.data, '$.state.input.workdir'), s.directory FROM part p JOIN session s ON s.id = "
            "p.session_id WHERE json_extract(p.data, '$.type') = 'tool' AND json_extract(p.data, '$.tool') = 'bash' "
            "AND p.time_created BETWEEN ? AND ?", (int((started - 600) * 1000), int((started + 60) * 1000))).fetchall()
    except sqlite3.Error:
        return None
    finally:
        con.close()
    best = None
    for sid, created, command, workdir, directory in rows:
        line = (command or "").replace("\\", "/").lower()
        if "strata_suspend" not in line or not all(t in line for t in want):
            continue
        if not (_same_dir(workdir, rec.get("cwd")) or (not workdir and _same_dir(directory, rec.get("cwd")))):
            continue
        gap = abs(started - created / 1000)
        if best is None or gap < best[0]:
            best = (gap, sid)
    return best[1] if best else None


def find_claude_session(rec: dict, projects: Path | None = None) -> tuple[str, str] | None:
    """The Claude Code session that ran this wrapped job: the record's session id (CLAUDE_CODE_SESSION_ID), but only
    if that session's transcript has a Bash or PowerShell call that ran strata_suspend with the job's arguments, near
    its start - an opencode run started from a Claude shell inherits the id, and must not wake the Claude session.
    -> (session id, the session's folder) or None."""
    sid = rec.get("claude_session")
    if not sid:
        return None
    started, want = float(rec.get("started") or 0), _tokens(rec.get("cmd") or [])
    for f in (projects or CLAUDE_PROJECTS).glob(f"*/{sid}.jsonl"):
        cwd, hit = None, False
        try:
            with open(f, encoding="utf-8", errors="replace") as lines:
                for line in lines:
                    if '"tool_use"' not in line and cwd is not None:
                        continue
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    cwd = cwd or e.get("cwd")
                    for c in ((e.get("message") or {}).get("content") or []) if isinstance(e, dict) else []:
                        if not (isinstance(c, dict) and c.get("type") == "tool_use"
                                and c.get("name") in ("Bash", "PowerShell")):
                            continue
                        cmd = str((c.get("input") or {}).get("command") or "").replace("\\", "/").lower()
                        if "strata_suspend" in cmd and all(t in cmd for t in want) and _near(e, started):
                            hit = True
        except OSError:
            continue
        if hit and cwd:
            return sid, cwd
    return None


def _near(e: dict, started: float) -> bool:
    """A transcript entry's time within a few minutes before the job (no time: accepted)."""
    ts = e.get("timestamp")
    if not ts:
        return True
    try:
        from datetime import datetime
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return True
    return started - 600 <= t <= started + 60


def claude_exe() -> list[str] | None:
    import shutil
    found = shutil.which("claude")
    if not found:
        exe = Path.home() / ".local" / "bin" / ("claude.exe" if os.name == "nt" else "claude")
        found = str(exe) if exe.exists() else None
    return [found] if found else None


def log_tail(log: str | None) -> str:
    try:
        text = Path(log).read_bytes().decode("utf-8", "replace") if log else ""
    except OSError:
        return "(no output was saved)"
    lines = text.splitlines()[-LOG_TAIL_LINES:]
    tail = "\n".join(lines)[-LOG_TAIL_CHARS:]
    return tail if tail.strip() else "(the job printed nothing before it stopped)"


def files_written(rec: dict, limit: int = 15) -> list[str]:
    """Files under the job's folder written since it started (its partial output), newest first."""
    root, since, found, seen = Path(rec.get("cwd") or "."), float(rec.get("started") or 0) - 1, [], 0
    log = os.path.normcase(str(rec.get("log") or ""))
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".venv")]
        if Path(dirpath).relative_to(root).parts and len(Path(dirpath).relative_to(root).parts) > 4:
            dirnames[:] = []
        for name in filenames:
            seen += 1
            if seen > 50000:
                break
            p = Path(dirpath) / name
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_mtime >= since and os.path.normcase(str(p)) != log:
                found.append((st.st_mtime, p, st.st_size))
    found.sort(reverse=True)
    return [f"{p.relative_to(root)}  ({size:,} bytes, written {_when(t)})" for t, p, size in found[:limit]]


def wake_message(rec: dict) -> str:
    files = files_written(rec)
    return "\n".join([
        "[Automatic note from Strata's job watcher, sent when Strata started.]",
        "",
        f"A job you ran through strata_suspend.py did not return its result to you: {state(rec)}.",
        f"Command: {subprocess.list2cmdline(rec.get('cmd') or [])}",
        f"Folder: {rec.get('cwd')}",
        f"Started: {_when(rec.get('started'))}; its output was last written at "
        f"{_when(_mtime(rec.get('log')))}.",
        "",
        "What happened: " + cause(rec),
        "",
        f"The last lines of its output (all of it: {rec.get('log')}):",
        "```",
        log_tail(rec.get("log")),
        "```",
        "",
        "Files in that folder written since it started (its partial output):" if files else
        "No files in that folder were written since it started.",
        *[f"- {x}" for x in files],
        "",
        "Decide what to do: run it again (through the same strata_suspend.py wrapper), continue from the partial "
        "output, or report the failure. Check that any partial output file is complete before you use it.",
    ])


def _mtime(p) -> float | None:
    try:
        return Path(p).stat().st_mtime if p else None
    except OSError:
        return None


def opencode_exe() -> list[str] | None:
    """The opencode program itself (not npm's .cmd shim: cmd.exe would cut a message at its first line break)."""
    appdata = os.environ.get("APPDATA")
    if appdata:
        exe = Path(appdata) / "npm" / "node_modules" / "opencode-ai" / "bin" / "opencode.exe"
        if exe.exists():
            return [str(exe)]
    import shutil
    found = shutil.which("opencode.exe") or (shutil.which("opencode") if os.name != "nt" else None)
    return [found] if found else None


def opencode_up(url: str) -> bool:
    import urllib.error
    import urllib.request
    try:
        urllib.request.urlopen(url + "/", timeout=5).close()
        return True
    except urllib.error.HTTPError:
        return True                                  # it answered (it asks for its password): it is up
    except (urllib.error.URLError, OSError):
        return False


def agent_env() -> dict:
    """This environment without another Claude Code session's variables: started from a Claude shell, `claude` would
    take itself for a nested session (and a wrapped job would record that session as its own)."""
    return {k: v for k, v in os.environ.items()
            if not (k == "CLAUDECODE" or k.startswith(("CLAUDE_CODE_", "CLAUDE_AGENT_SDK_", "CLAUDE_PREVIEW_"))
                    or k in ("CLAUDE_PID", "CLAUDE_EFFORT"))}


LAUNCH_CHECK_S = 12.0                            # an agent program that fails at once (not signed in, no such session)


def _launch(popen, args: list, cwd, log: Path) -> bool:
    """Start the agent program with the note, in the background.  It must not fail at once: `claude` exits within
    seconds when its own sign-in has expired, opencode when the session is unknown - then the session counts as not
    told (the next Strata start tries again) and the reason is in the log."""
    flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
    with open(log, "ab") as out:
        try:
            proc = popen(args, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=cwd,
                         creationflags=flags, start_new_session=os.name != "nt", close_fds=True, env=agent_env())
            deadline = time.time() + LAUNCH_CHECK_S
            while proc is not None and hasattr(proc, "poll") and time.time() < deadline:
                code = proc.poll()
                if code is not None:
                    if code != 0:
                        out.flush()
                        tail = log.read_bytes().decode("utf-8", "replace").strip().splitlines()[-1:]
                        print(f"wake: {Path(args[0]).name} failed at once (exit {code}): "
                              f"{tail[0] if tail else 'no output'}", flush=True)
                        return False
                    break
                time.sleep(0.5)
            return True
        except OSError as e:
            out.write(f"could not start {args[0]}: {e}\n".encode())
            return False


def wake(url: str = OPENCODE_URL, db: Path = OPENCODE_DB, dry_run: bool = False, popen=subprocess.Popen,
         projects: Path | None = None, only: Path | None = None) -> int:
    """Send each cut-off job's session its note (once): an opencode session through the opencode server, a Claude
    Code session through `claude --resume`.  only: that record alone.  -> the number of sessions told."""
    recs = records()
    todo = [(f, r) for f, r in ([(only, read(only) or {})] if only else recs)
            if state(r) in (CUT_OFF, UNDELIVERED) and not r.get("woken") and int(r.get("wake_tries") or 0) < WAKE_TRIES
            and not not_needed(r, recs)]
    if not todo:
        print("wake: no cut-off jobs to report", flush=True)
        return 0
    up = None                                        # the opencode server: asked once, when a session needs it
    woken = 0
    for f, rec in todo:
        msg = wake_message(rec)
        sid = find_opencode_session(rec, db)
        if sid:
            if dry_run:
                print(f"wake (dry run): opencode session {sid}:\n{msg}\n", flush=True)
                continue
            up = opencode_up(url) if up is None else up
            exe = opencode_exe()
            if not up or not exe:
                print(f"wake: {f.name} belongs to opencode session {sid}, but "
                      f"{'no opencode server answers at ' + url if not up else 'the opencode program was not found'}; "
                      "the next Strata start tries again", flush=True)
                continue
            if _launch(popen, exe + ["run", "--attach", url, "--session", sid, "--dir", str(rec.get("cwd")), msg],
                       rec.get("cwd"), f.with_suffix(".wake.log")):
                update(f, woken=time.time(), session=sid, agent="opencode")
                woken += 1
                print(f"wake: opencode session {sid} told about {f.name} ({state(rec)})", flush=True)
            continue
        claude = find_claude_session(rec, projects)
        if claude:
            sid, cwd = claude
            if dry_run:
                print(f"wake (dry run): Claude Code session {sid} in {cwd}:\n{msg}\n", flush=True)
                continue
            exe = claude_exe()
            if not exe:
                print(f"wake: {f.name} belongs to Claude Code session {sid}, but the claude program was not found",
                      flush=True)
                continue
            if _launch(popen, exe + ["--resume", sid, "-p", msg], cwd, f.with_suffix(".wake.log")):
                update(f, woken=time.time(), session=sid, agent="claude")
                woken += 1
                print(f"wake: Claude Code session {sid} told about {f.name} ({state(rec)})", flush=True)
            continue
        update(f, wake_tries=int(rec.get("wake_tries") or 0) + 1)
        print(f"wake: no session found for {f.name} ({subprocess.list2cmdline(rec.get('cmd') or [])})", flush=True)
    return woken


def wake_all(url: str = OPENCODE_URL, db: Path = OPENCODE_DB, dry_run: bool = False, popen=subprocess.Popen,
             projects: Path | None = None, strata_alive: float | None = None) -> int:
    """`strata_suspend.py wake`: the sessions whose wrapped jobs were cut off, then those cut off mid-turn.
    strata_alive: the previous Strata run's last heartbeat (the server passes it; see serve/server.py)."""
    n = wake(url, db, dry_run, popen, projects)
    told = {r.get("session") for _, r in records() if r.get("woken")}
    return n + wake_interrupted(url, db, projects, told, dry_run, popen, strata_alive=strata_alive)


# ---- the live register: who is working right now (written every minute by Strata's heartbeat)

LIVE_PREV = Path(__file__).resolve().parent.parent / "data" / "strata-alive.prev.json"   # the previous run's last
LIVE_COVERS_S = 180                              # a register this close to the moment the PC went down covers it
CLAUDE_FRESH_S = 600                             # a Claude Code transcript written this recently: that session works
CLAUDE_SHELL_MARKS = ("shell-snapshots", "CLAUDE_CODE_SHELL_LAUNCHER_SCRIPT")   # its Bash / PowerShell tool shells


def _norm(s: str) -> str:
    return " ".join(str(s).replace("\\", "/").replace('"', " ").replace("'", " ").lower().split())


def _descendants(names: tuple) -> list:
    """The processes under every running process with one of these names (an agent and what it started)."""
    try:
        import psutil
    except ImportError:
        return []
    out = []
    for p in psutil.process_iter(["name"]):
        if (p.info["name"] or "").lower() in names:
            try:
                out += p.children(recursive=True)
            except Exception:  # noqa: BLE001 - it ended meanwhile, or no access
                pass
    return out


def live_register(strata_requests: dict | None = None, db: Path = OPENCODE_DB, projects: Path | None = None,
                  now: float | None = None) -> dict:
    """Which agent sessions are working now, and why:
    - opencode sessions with a request in Strata (generating, waiting out a hold, or between two steps: the server
      passes them, from the x-opencode-session-id header);
    - opencode sessions whose bash call is still running (opencode's database names the command, and a process
      under opencode runs it);
    - Claude Code sessions with a command running (a process under Claude Code carries CLAUDE_CODE_SESSION_ID);
    - Claude Code sessions that wrote to their transcript in the last CLAUDE_FRESH_S (thinking, between calls)."""
    now = now or time.time()
    oc = {sid: f"a request in Strata ({r.get('state')})" for sid, r in (strata_requests or {}).items()}
    cl = {}
    running = []
    if db.exists():
        try:
            con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=5)
            try:
                running = con.execute(
                    "SELECT session_id, json_extract(data, '$.state.input.command') FROM part WHERE "
                    "json_extract(data, '$.type') = 'tool' AND json_extract(data, '$.tool') = 'bash' AND "
                    "json_extract(data, '$.state.status') = 'running' AND time_created > ?",
                    (int((now - 7 * 86400) * 1000),)).fetchall()
            finally:
                con.close()
        except sqlite3.Error:
            running = []
    if running:
        lines = []
        for p in _descendants(("opencode.exe", "opencode")):
            try:
                lines.append(_norm(" ".join(p.cmdline())))
            except Exception:  # noqa: BLE001
                pass
        for sid, command in running:
            want = _norm(command or "")
            if want and sid not in oc and any(want[:160] in line or all(t in line for t in want.split()[:8])
                                              for line in lines):
                oc[sid] = "a command running"
    for p in _descendants(("claude.exe", "claude")):
        try:
            # only Claude Code's own command shells (Bash: its shell-snapshots; PowerShell: its launcher script) -
            # an idle session's MCP servers carry the session id too, and run all the time
            if not any(m in " ".join(p.cmdline()) for m in CLAUDE_SHELL_MARKS):
                continue
            sid = p.environ().get("CLAUDE_CODE_SESSION_ID")
        except Exception:  # noqa: BLE001 - no access
            sid = None
        if sid and sid not in cl:
            cl[sid] = "a command running"
    for f in (projects or CLAUDE_PROJECTS).glob("*/*.jsonl"):
        try:
            if now - f.stat().st_mtime <= CLAUDE_FRESH_S and f.stem not in cl:
                cl[f.stem] = "active in the last 10 minutes"
        except OSError:
            pass
    _add_parents(oc, db)
    return {"time": now, "opencode": oc, "claude": cl}


def _add_parents(oc: dict, db: Path) -> None:
    """A subagent is a child session (opencode's task tool): while it works, its parent's turn waits on it - the
    parent is the session to tell after a crash.  Up to a few levels (a subagent's own subagents)."""
    if not oc or not db.exists():
        return
    try:
        con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return
    try:
        todo = list(oc)
        for _ in range(4):
            if not todo:
                break
            rows = con.execute(f"SELECT id, parent_id FROM session WHERE id IN ({','.join('?' * len(todo))}) "
                               "AND parent_id IS NOT NULL", todo).fetchall()
            todo = []
            for child, parent in rows:
                if parent not in oc:
                    oc[parent] = f"a subagent working (session {child}: {oc[child]})"
                    todo.append(parent)
    except sqlite3.Error:
        pass
    finally:
        con.close()


def read_live(path: Path | None = None) -> dict | None:
    """The previous Strata run's last register (the server keeps it aside at its start)."""
    try:
        data = json.loads((path or LIVE_PREV).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    live = data.get("live") if isinstance(data, dict) else None
    return live if isinstance(live, dict) and live.get("time") else None


# ---- turns cut off without a wrapped job

def _ms(t) -> float:
    return float(t or 0) / 1000


def previous_boot(boot: float | None) -> float | None:
    """When Windows started before this boot (its event log: Kernel-General event 12 is each start).  A session that
    was mid-turn with its last activity after it was cut off by the outage that ended that run - however long that
    run was, and however long the session's last command ran.  None: no history (the caller guesses a window)."""
    if not boot or os.name != "nt":
        return None
    try:
        r = subprocess.run(["wevtutil", "qe", "System", "/q:*[System[Provider[@Name='Microsoft-Windows-Kernel-General']"
                            " and (EventID=12)]]", "/c:4", "/rd:true", "/f:xml"], capture_output=True, text=True,
                           timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    import re
    from datetime import datetime
    starts = []
    for t in re.findall(r"SystemTime='([^']+)'", r.stdout or ""):
        try:
            starts.append(datetime.fromisoformat(t[:26].rstrip("Z") + "+00:00").timestamp())
        except ValueError:
            continue
    earlier = [t for t in starts if t < boot - 60]       # the newest start is this boot itself
    return max(earlier) if earlier else None


def went_down_at(boot: float | None, db: Path = OPENCODE_DB, projects: Path | None = None) -> float | None:
    """When the PC went down before this boot: the last thing any agent wrote before Windows started (an opencode
    session's last update, a Claude Code transcript's last write).  After an outage of days that is days before the
    boot - the boot time itself says nothing about when the power went."""
    if not boot:
        return None
    last = None
    if db.exists():
        try:
            con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=5)
            try:
                t = con.execute("SELECT max(time_updated) FROM session WHERE time_updated < ?",
                                (int(boot * 1000),)).fetchone()[0]
                last = _ms(t) if t else None
            finally:
                con.close()
        except sqlite3.Error:
            pass
    for f in (projects or CLAUDE_PROJECTS).glob("*/*.jsonl"):
        try:
            t = f.stat().st_mtime
        except OSError:
            continue
        if t < boot and (last is None or t > last):
            last = t
    return last


def opencode_interrupted(db: Path = OPENCODE_DB, down: float | None = None, strata_alive: float | None = None,
                         now: float | None = None, since_boot: float | None = None) -> list[dict]:
    """opencode sessions to tell: mid-turn when the PC went down (last active between the previous boot `since_boot`
    and `down`; without boot history within CRASH_WINDOW_S before `down`), or a
    request that failed because Strata was not running - after Strata's last heartbeat `strata_alive` (without one:
    in the last STRATA_DOWN_WINDOW_S).  Top-level sessions only (a subagent's session belongs to its parent's turn);
    archived ones and a user's own abort are left alone."""
    if not db.exists():
        return []
    now = now or time.time()
    failed_since = strata_alive - 60 if strata_alive else now - STRATA_DOWN_WINDOW_S
    lower = since_boot if since_boot else (down - CRASH_WINDOW_S if down else None)
    since = min(lower if lower else now, failed_since)
    try:
        con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return []
    out = []
    try:
        rows = con.execute(
            "SELECT s.id, s.directory, s.time_updated, m.id, m.data FROM session s "
            "JOIN message m ON m.session_id = s.id "
            "WHERE s.time_archived IS NULL AND s.parent_id IS NULL AND s.time_updated >= ? AND m.time_created = "
            "(SELECT max(time_created) FROM message WHERE session_id = s.id)", (int(since * 1000),)).fetchall()
        for sid, directory, updated, mid, data in rows:
            try:
                m = json.loads(data)
            except ValueError:
                continue
            last = _ms(updated)
            err = m.get("error") or {}
            if down and lower and lower <= last <= down:
                parts = [json.loads(d) for (d,) in con.execute("SELECT data FROM part WHERE message_id = ?", (mid,))]
                open_tools = [x.get("tool") for x in parts if x.get("type") == "tool"
                              and (x.get("state") or {}).get("status") in ("running", "pending")]
                if any(t in WAITS_FOR_USER for t in open_tools):
                    continue                             # it was waiting for the user's answer, not working
                # a step that ended in tool calls is never the end of a turn: the next step never ran
                if m.get("role") == "user" or (m.get("role") == "assistant" and not err and (
                        not (m.get("time") or {}).get("completed") or m.get("finish") == "tool-calls")) or open_tools:
                    out.append({"agent": "opencode", "session": sid, "cwd": directory, "at": last, "why": "restart"})
                    continue
            msg = str((err.get("data") or {}).get("message") or "")
            url = str(((err.get("data") or {}).get("metadata") or {}).get("url") or "")
            done = _ms((m.get("time") or {}).get("completed")) or last
            if m.get("role") == "assistant" and err.get("name") == "APIError" and done >= failed_since \
                    and (("Unable to connect" in msg and "127.0.0.1" in url) or "Strata is suspended" in msg):
                out.append({"agent": "opencode", "session": sid, "cwd": directory, "at": done, "why": "strata-down",
                            "error": msg})
    except sqlite3.Error:
        pass
    finally:
        con.close()
    return out


def _claude_mid_turn(path: Path) -> tuple[bool, str | None]:
    """(the transcript ends in the middle of a turn, the session's folder)."""
    cwd, last = None, None
    try:
        with open(path, encoding="utf-8", errors="replace") as lines:
            for line in lines:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(e, dict):
                    continue
                cwd = e.get("cwd") or cwd
                if e.get("type") in ("user", "assistant") and not e.get("isSidechain") and not e.get("isMeta"):
                    last = e
    except OSError:
        return False, None
    if not last:
        return False, cwd
    content = (last.get("message") or {}).get("content")
    if last.get("type") == "user":
        return True, cwd                             # a prompt or a tool result the model never answered
    calls = [c for c in content or [] if isinstance(c, dict) and c.get("type") == "tool_use"] \
        if isinstance(content, list) else []
    if any(c.get("name") in WAITS_FOR_USER for c in calls):
        return False, cwd                            # it was waiting for the user's answer, not working
    return bool(calls) or not (last.get("message") or {}).get("stop_reason"), cwd


def claude_interrupted(projects: Path | None = None, down: float | None = None,
                       since_boot: float | None = None) -> list[dict]:
    """Claude Code sessions mid-turn when the PC went down: a transcript last written between the previous boot and
    `down` (without boot history: within CRASH_WINDOW_S before it) that ends inside a turn."""
    if not down:
        return []
    lower = since_boot or down - CRASH_WINDOW_S
    out = []
    for f in (projects or CLAUDE_PROJECTS).glob("*/*.jsonl"):
        try:
            t = f.stat().st_mtime
        except OSError:
            continue
        if not (lower <= t <= down):
            continue
        mid, cwd = _claude_mid_turn(f)
        if mid and cwd:
            out.append({"agent": "claude", "session": f.stem, "cwd": cwd, "at": t, "why": "restart"})
    return out


def interrupted_message(item: dict, boot: float | None, down: float | None = None) -> str:
    if item["why"] == "restart":
        stop = (f"Windows was shut down or restarted at {_when(item['orderly'])} " if item.get("orderly") else
                f"the PC was restarted or went down at about {_when(down or item['at'])} (a restart, a power cut or a "
                "crash) ")
        what = (stop + f"and Windows started again at {_when(boot)}; every process stopped" +
                (", including Strata and this session." if item["agent"] == "opencode"
                 else ", including this session."))
        todo = ("Pick up where you left off. First check the state of what you were doing - files you were writing, "
                "commands or jobs you had started (they were stopped too) - and redo any step that may not have "
                "completed.")
    elif "Strata is suspended" in str(item.get("error")):
        what = ("Strata, the local model server, was held for a job that needed the PC's memory and refused your "
                "request; that job has ended and Strata is back.")
        todo = "Pick up where you left off; nothing else of yours was stopped."
    else:
        what = (f"Strata, the local model server, was not running when your request reached it "
                f"({item.get('error')}); it is running again now.")
        todo = "Pick up where you left off; nothing else of yours was stopped."
    return "\n".join([
        "[Automatic note from Strata's watcher, sent when Strata started.]",
        "",
        f"Your last turn in this session was cut off at about {_when(item['at'])} and never finished"
        + (f" (you had {item['working']})." if item.get("working") else "."),
        "What happened: " + what + " It was not caused by anything you did or by your code.",
        "",
        todo,
    ])


def wake_interrupted(url: str = OPENCODE_URL, db: Path = OPENCODE_DB, projects: Path | None = None,
                     skip: set | None = None, dry_run: bool = False, popen=subprocess.Popen,
                     boot: float | None = None, now: float | None = None, strata_alive: float | None = None,
                     since_boot: float | None = None, live: dict | None = None,
                     live_path: Path | None = None) -> int:
    """Tell each session cut off mid-turn (see opencode_interrupted / claude_interrupted), once per event."""
    boot = boot if boot is not None else boot_time()
    down = went_down_at(boot, db, projects)
    prev = previous_boot(boot) if since_boot is None else since_boot
    items = opencode_interrupted(db, down, strata_alive, now, prev) + claude_interrupted(projects, down, prev)
    live = read_live(live_path) if live is None else live
    if down and live and boot and down - LIVE_COVERS_S <= float(live["time"]) <= boot:
        # Strata's register, written within a few minutes of the moment the PC went down, names who was working:
        # a session that only looks mid-turn (killed earlier in that run) is not in it
        kept = []
        for item in items:
            reason = (live.get(item["agent"]) or {}).get(item["session"])
            if item["why"] != "restart" or reason:
                orderly = {"orderly": float(live["time"])} if live.get("shutting_down") and item["why"] == "restart" \
                    else {}
                kept.append({**item, **({"working": reason} if reason else {}), **orderly})
            else:
                print(f"wake: {item['agent']} session {item['session']} looks mid-turn but was not working when the "
                      "PC went down (not in Strata's last register): left alone", flush=True)
        items = kept
    ledger_file = jobs_dir() / "interrupted.json"
    try:
        ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ledger = {}
    told, up = 0, None
    for item in items:
        key = f"{item['agent']}:{item['session']}:{int(item['at'])}"
        if key in ledger or item["session"] in (skip or set()):
            continue
        msg = interrupted_message(item, boot, down)
        if dry_run:
            print(f"wake (dry run): {item['agent']} session {item['session']} ({item['why']}):\n{msg}\n", flush=True)
            continue
        log = jobs_dir() / f"interrupted-{item['session']}.wake.log"
        jobs_dir().mkdir(parents=True, exist_ok=True)
        if item["agent"] == "opencode":
            up = opencode_up(url) if up is None else up
            exe = opencode_exe()
            if not (up and exe):
                print(f"wake: opencode session {item['session']} was cut off, but no opencode server answers at {url}",
                      flush=True)
                continue
            ok = _launch(popen, exe + ["run", "--attach", url, "--session", item["session"], "--dir", item["cwd"],
                                       msg], item["cwd"], log)
        else:
            exe = claude_exe()
            ok = bool(exe) and _launch(popen, exe + ["--resume", item["session"], "-p", msg], item["cwd"], log)
        if ok:
            ledger[key] = time.time()
            told += 1
            print(f"wake: {item['agent']} session {item['session']} told its turn was cut off ({item['why']})",
                  flush=True)
    if told:
        try:
            jobs_dir().mkdir(parents=True, exist_ok=True)
            _write(ledger_file, ledger)
        except OSError:
            pass
    return told


# ---- who holds Strata, and stopping it (strata_suspend.py status / stop, the web page's Locked card)

LOOP_WINDOW_S = 1800                             # the same job started this often within this long: a restart loop
LOOP_STARTS = 3
STOPPED = "stopped"                              # the user stopped it (strata_suspend.py stop): never woken for


def _rec_pids(rec: dict) -> list[int]:
    out = [int(rec["wrapper"]["pid"])] if (rec.get("wrapper") or {}).get("pid") else []
    return out + [int(j["pid"]) for j in rec.get("job") or [] if j.get("pid")]


def _session_title(sid: str, db: Path = OPENCODE_DB) -> str | None:
    try:
        con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=5)
        try:
            r = con.execute("SELECT title FROM session WHERE id = ?", (sid,)).fetchone()
            return r[0] if r else None
        finally:
            con.close()
    except sqlite3.Error:
        return None


def _cmdline(pid: int) -> str | None:
    try:
        import psutil
        return subprocess.list2cmdline(psutil.Process(pid).cmdline())
    except Exception:  # noqa: BLE001
        return None


def session_info(sid: str, db: Path = OPENCODE_DB) -> dict:
    """An opencode session, for the request list: its title and agent; for a subagent (a child session, opencode's
    task tool) also its parent session's title and the task it was given (the parent's task call: description,
    subagent type).  Read-only; {} when it cannot be read."""
    if not sid or not db.exists():
        return {}
    try:
        con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return {}
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(session)")}
        row = con.execute(f"SELECT title, parent_id, directory{', agent' if 'agent' in cols else ''} FROM session "
                          "WHERE id = ?", (sid,)).fetchone()
        if not row:
            return {}
        out = {"title": row[0], "folder": row[2], **({"agent": row[3]} if len(row) > 3 and row[3] else {})}
        if row[1]:
            parent = con.execute("SELECT title FROM session WHERE id = ?", (row[1],)).fetchone()
            out.update(parent=row[1], parent_title=parent[0] if parent else None)
            task = con.execute(
                "SELECT json_extract(data, '$.state.input.description'), json_extract(data, "
                "'$.state.input.subagent_type') FROM part WHERE session_id = ? AND json_extract(data, '$.tool') = "
                "'task' AND json_extract(data, '$.state.metadata.sessionId') = ? ORDER BY time_created DESC LIMIT 1",
                (row[1], sid)).fetchone()
            if task:
                out.update(task=task[0], subagent=task[1])
        return out
    except sqlite3.Error:
        return {}
    finally:
        con.close()


def describe_request(r: dict) -> str:
    """One line for a request in Strata: its state, how long, and who sent it (and for a subagent, for what)."""
    mins = r.get("waiting_s", 0) // 60
    t = f"{r.get('state')} ({mins} min)" if mins else r.get("state")
    if not r.get("session"):
        return f"{t} - from {r.get('client') or 'a client'}"
    who = f'opencode session "{r.get("title") or r.get("session")}"'
    if r.get("parent"):
        who = (f'subagent{" " + r["subagent"] if r.get("subagent") else ""} of "{r.get("parent_title") or r["parent"]}"'
               + (f' - task: {r["task"]}' if r.get("task") else ""))
    return f"{t} - {who}"


def describe_holders(pids: list, db: Path = OPENCODE_DB, projects: Path | None = None,
                     now: float | None = None) -> list[dict]:
    """The holders, one entry per wrapped job (its wrapper and job processes together): the command, its folder,
    when it started, the agent session that ran it, and how often that same job was started in the last
    LOOP_WINDOW_S (an agent restarting it: stopping the job alone does not help).  A held process that is no
    wrapped job (strata_suspend.py hold --pid) gets an entry of its own with its command line."""
    now = now or time.time()
    left, out = [int(p) for p in pids or [] if p], []
    recs = records()
    for f, rec in recs:
        if rec.get("state") not in ("running",):
            continue
        mine = [p for p in _rec_pids(rec) if p in left]
        if not mine:
            continue
        left = [p for p in left if p not in mine]
        same = [r for _, r in recs if r.get("cmd") == rec.get("cmd") and _same_dir(r.get("cwd"), rec.get("cwd"))
                and now - float(r.get("started") or 0) <= LOOP_WINDOW_S]
        entry = {"pids": mine, "record": f.name, "command": subprocess.list2cmdline(rec.get("cmd") or []),
                 "folder": rec.get("cwd"), "started": rec.get("started"),
                 "running_s": round(now - float(rec.get("started") or now)), "starts": len(same),
                 "loop": len(same) >= LOOP_STARTS, "wrapped": True}
        sid = find_opencode_session(rec, db)
        if sid:
            entry.update(agent="opencode", session=sid, title=_session_title(sid, db))
        else:
            claude = find_claude_session(rec, projects)
            if claude:
                entry.update(agent="claude", session=claude[0], title=f"Claude Code in {claude[1]}")
        out.append(entry)
    for p in left:
        out.append({"pids": [p], "command": _cmdline(p), "wrapped": False})
    return out


def advice(entry: dict) -> str:
    """What to do about one holder, in a line."""
    if not entry.get("wrapped"):
        return ("not a wrapped job (held with strata_suspend.py hold): end that program, or `strata_suspend.py "
                "release` to let Strata come back over it")
    who = f'{entry.get("agent")} session "{entry.get("title") or entry.get("session")}"' if entry.get("session") \
        else "the agent that started it"
    if entry.get("loop"):
        return (f"started {entry['starts']} times in the last {LOOP_WINDOW_S // 60} min by {who}: stop that session "
                "first (Esc in opencode / Claude Code), or it starts the job again - then `strata_suspend.py stop`")
    return f"`strata_suspend.py stop` ends it (and tells {who} not to run it again); or wait for it to finish"


def stop_jobs(pids: list | None, all_held: list) -> list[dict]:
    """End the wrapped jobs that hold Strata - these pids' jobs, or all (pids None).  Each is marked stopped by the
    user first: the waker never tells its session it was cut off, and a wrapper still alive tells the agent not to
    run it again.  Only a wrapped job's own processes (the job and the shim that runs it) are ended; a holder that is
    no wrapped job is left alone (see advice).  -> what was done, per holder."""
    holders = describe_holders(all_held)
    want = None if pids is None else {int(p) for p in pids}
    done = []
    for h in holders:
        if want is not None and not want & set(h["pids"]):
            continue
        if not h.get("wrapped"):
            done.append({**h, "stopped": False, "why": "not a wrapped job: left alone"})
            continue
        f = jobs_dir() / h["record"]
        rec = read(f) or {}
        update(f, state=STOPPED, stopped_by_user=time.time(), ended=time.time())
        job = [int(j["pid"]) for j in rec.get("job") or [] if alive(j["pid"], j.get("started"))]
        for pid in job:                                  # the shim and the job under it, whole trees
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            else:
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass
        done.append({**h, "stopped": True, "ended": job})
    return done


# ---- for the agent and the user

def notes_here(cwd: str | None = None) -> None:
    """At a wrapped job's start: jobs from this folder that did not get their result back, once each."""
    here = cwd or os.getcwd()
    for f, rec in records():
        if _same_dir(rec.get("cwd"), here) and not rec.get("reported") and state(rec) in (CUT_OFF, UNDELIVERED):
            print(f"NOTE: an earlier job in this folder did not return its result ({state(rec)}; started "
                  f"{_when(rec.get('started'))}): {subprocess.list2cmdline(rec.get('cmd') or [])}. {cause(rec)} "
                  f"Its output: {rec.get('log')}. Run it again if its result is still needed.", flush=True)
            update(f, reported=time.time())


def list_jobs() -> int:
    jobs = records()
    if not jobs:
        print("no wrapped jobs in the last 7 days")
        return 0
    for f, rec in jobs[-30:]:
        st = state(rec)
        label = st.upper() if st in (CUT_OFF, UNDELIVERED) else ("stopped by you" if st == STOPPED else st)
        extra = f" exit {rec.get('exit')}" if st in ("done", UNDELIVERED) else ""
        woke = f"  [session told {_when(rec.get('woken'))}]" if rec.get("woken") else ""
        why = not_needed(rec, jobs) if st in (CUT_OFF, UNDELIVERED) and not woke else None
        if why:
            woke = f"  [{why}: nobody is woken for it]"
        print(f"{_when(rec.get('started'))[:16]}  {label}{extra}  in {rec.get('cwd')}: "
              f"{subprocess.list2cmdline(rec.get('cmd') or [])}  (log {rec.get('log')}){woke}")
    return 0
