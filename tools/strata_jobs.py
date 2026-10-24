"""tools/strata_jobs.py - local: waking the agent sessions that were working when Strata (or the PC) went down.

Who was working comes from exactly two places, and nothing guesses:

- an expired lease: its wrapper stopped renewing (killed, crashed, the PC went down), so the job never reported its
  result back - the lease names the session that ran it (named once, when the lease was made);
- the sessions that had a request in Strata when it went down: the lease checkpoint's register, kept aside at the
  next server start as checkpoint.prev.json (serve/server.py: Service.start_lease_beat).

`strata_suspend.py wake` (the Strata server runs it a few seconds after it starts, with "wake_cut_off_jobs": true)
tells exactly those sessions, once each, through the opencode server (`opencode run --attach --session`) or
`claude --resume <id> -p`: what happened and the end of the job's output.  A clean shutdown (the checkpoint marked
"shutting_down") tells only the expired leases: its register is not a list of sessions cut off mid-request.
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

OPENCODE_URL = "http://127.0.0.1:4096"
OPENCODE_DB = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"       # Claude Code's transcripts: <project>/<session id>.jsonl
LOG_TAIL_LINES, LOG_TAIL_CHARS = 60, 6000
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".mypy_cache",
             ".pytest_cache", ".ruff_cache", "build", "dist"}
LEASE_AGENTS = ("opencode", "claude")
LEASE_EXPIRED = "lease expired"                        # a lease-path job whose wrapper stopped renewing (see wake_leases)


def _data_dir() -> Path:
    """The repo's data folder: the job logs and the wake logs live here."""
    return Path(__file__).resolve().parent.parent / "data"


def _json_file(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _when(t) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) if t else "an unknown time"


def _mtime(p) -> float | None:
    try:
        return Path(p).stat().st_mtime if p else None
    except OSError:
        return None


def _same_dir(a, b) -> bool:
    return bool(a and b) and os.path.normcase(os.path.normpath(str(a))) == os.path.normcase(os.path.normpath(str(b)))


# ---- the session that ran a wrapped job, found once when the lease is made (the waker never guesses later)

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


# ---- the job's program, its log and the relay

def program(cmd: list) -> list:
    """The command with its program found as a shell finds it.  Windows does not start a relative path written with
    forward slashes (`.venv/Scripts/python.exe`: "cannot find the file"), which agents write; `shutil.which`
    finds it (and a bare name on PATH, with its extension).  Unchanged when it finds nothing: Popen then says why."""
    if not cmd:
        return cmd
    found = shutil.which(cmd[0])
    return [found, *cmd[1:]] if found else list(cmd)


def relay(proc: subprocess.Popen, log: str) -> int:
    """The wrapper: copy the job's log to this process's output as it grows, until the job ends."""
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


def kill_tree(pid: int) -> None:
    """End a process and everything under it. psutil first: taskkill is refused even on the account's own processes
    in an ssh session (the agent's, key login), psutil is not. Without psutil: taskkill / kill."""
    try:
        import psutil
        try:
            p = psutil.Process(int(pid))
            kids = p.children(recursive=True)
            p.kill()
            for k in kids:
                try:
                    k.kill()
                except psutil.Error:
                    pass
            psutil.wait_procs([p, *kids], timeout=5)
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            pass
        return
    except ImportError:
        pass
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


# ---- the note a session is told

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
        f"A job you ran through strata_suspend.py did not return its result to you: {rec.get('state') or 'cut off'}.",
        f"Command: {subprocess.list2cmdline(rec.get('cmd') or [])}",
        f"Folder: {rec.get('cwd')}",
        f"Started: {_when(rec.get('started'))}; its output was last written at "
        f"{_when(_mtime(rec.get('log')))}.",
        "",
        "What happened: The lease that held Strata for this job expired: its wrapper stopped renewing it (killed, "
        "crashed, or the PC went down), so the job never reported its result back to you.",
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


def register_message(entry: dict) -> str:
    """The note for a session of the checkpoint's register (no lease): Strata went down while its request was in
    flight."""
    return "\n".join([
        "[Automatic note from Strata's watcher, sent when Strata started.]",
        "",
        "Strata, the local model server, went down while this session had a request in flight (it was in Strata's "
        "last live register), so your last turn never got its answer and the session was cut off. It was not caused "
        "by anything you did or by your code.",
        "",
        "Check what you were doing and continue: re-send the request that was in flight, or pick up where you left "
        "off.",
    ])


# ---- the opencode / Claude Code programs, and starting one with the note

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


def claude_exe() -> list[str] | None:
    import shutil
    found = shutil.which("claude")
    if not found:
        exe = Path.home() / ".local" / "bin" / ("claude.exe" if os.name == "nt" else "claude")
        found = str(exe) if exe.exists() else None
    return [found] if found else None


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


# ---- an opencode session, for the request list and a register session's folder

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


# ---- who to wake from the lease table (serve/leases.py). Read only: the server may hold the table's writer lock.

def to_wake_from_leases(leases_dir: Path, clean_shutdown: bool | None = None) -> list[dict]:
    """The sessions to wake: every expired lease with a session that was not woken yet, and, after an unclean
    shutdown, every session the previous run's checkpoint (checkpoint.prev.json) lists as in flight.  The register
    and the clean-shutdown mark both come from that file: clean_shutdown=None reads the mark from it (a missing mark
    means unclean).  One entry per (agent, id); a lease's entry wins over an in-flight one.  A missing or unreadable
    file is skipped."""
    leases_dir = Path(leases_dir)
    if not leases_dir.is_dir():
        return []
    expired = []
    for p in leases_dir.glob("*.json"):
        if p.name in ("checkpoint.json", "checkpoint.prev.json", "state.json"):
            continue
        lease = _json_file(p)
        if not isinstance(lease, dict) or lease.get("ended") != "expired" or lease.get("woken"):
            continue
        s = lease.get("session")
        if isinstance(s, dict) and isinstance(s.get("agent"), str) and isinstance(s.get("id"), str) and s["id"]:
            expired.append(lease)
    out: dict[tuple, dict] = {}
    for lease in sorted(expired, key=lambda l: (l.get("created") or 0, str(l.get("id")))):
        s = lease["session"]
        out.setdefault((s["agent"], s["id"]), {"agent": s["agent"], "id": s["id"], "lease": lease.get("id"),
                                               "cause": "expired"})
    cp = _json_file(leases_dir / "checkpoint.prev.json")
    if clean_shutdown is None:
        clean_shutdown = bool(isinstance(cp, dict) and cp.get("shutting_down"))
    if not clean_shutdown:
        register = cp.get("register") if isinstance(cp, dict) else None
        if isinstance(register, dict):
            told = {tuple(x) for x in (cp.get("woken") or []) if isinstance(x, list) and len(x) == 2}
            for agent in LEASE_AGENTS:
                sessions = register.get(agent)
                if not isinstance(sessions, dict):
                    continue
                for sid in sessions:
                    if (agent, sid) in told:
                        continue                           # the waker told it already (POST /lease/session-woken)
                    out.setdefault((agent, sid), {"agent": agent, "id": sid, "lease": None, "cause": "in-flight"})
    return list(out.values())


def _lease_record(lease: dict) -> dict:
    """A wake_message record made from a lease: its command, folder, start and log."""
    return {"state": LEASE_EXPIRED, "cmd": [str(lease.get("cmd") or "")], "cwd": lease.get("cwd"),
            "started": lease.get("created"), "log": lease.get("log")}


def _claude_transcript_cwd(sid: str, projects: Path | None = None) -> str | None:
    """The session's own folder: the "cwd" of the first entry that has one in its transcript."""
    for f in (projects or CLAUDE_PROJECTS).glob(f"*/{sid}.jsonl"):
        try:
            with open(f, encoding="utf-8", errors="replace") as lines:
                for line in lines:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(e, dict) and e.get("cwd"):
                        return e["cwd"]
        except OSError:
            continue
    return None


def wake_leases(leases_dir, mark_woken, clean_shutdown, url: str = OPENCODE_URL, projects: Path | None = None,
                dry_run: bool = False, popen=subprocess.Popen, opencode=None, claude=None) -> list[dict]:
    """Tell exactly the sessions rule 3 lists (to_wake_from_leases), nothing else: an expired lease's session and,
    after an unclean shutdown, the checkpoint's register sessions not yet told.  mark_woken(entry) is the server's
    API (the waker never writes a lease file itself); it is called only after a launch that did not fail.  -> the
    entries that were told."""
    entries = to_wake_from_leases(leases_dir, clean_shutdown)
    if not entries:
        return []
    leases_dir = Path(leases_dir)
    up = None                                        # the opencode server: asked once, when a session needs it
    told = []
    for entry in entries:
        lease = None
        if entry.get("lease"):
            lease = _json_file(leases_dir / f"{entry['lease']}.json")
            if not isinstance(lease, dict):
                lease = None
        msg = wake_message(_lease_record(lease)) if lease else register_message(entry)
        if dry_run:
            print(f"wake (dry run): {entry['agent']} session {entry['id']} ({entry['cause']}):\n{msg}\n", flush=True)
            continue
        logs = _data_dir() / "job-logs"
        log = logs / f"lease-{entry['id']}.wake.log"
        logs.mkdir(parents=True, exist_ok=True)
        if entry["agent"] == "opencode":
            up = opencode_up(url) if up is None else up
            exe = opencode or opencode_exe()
            if not (up and exe):
                print(f"wake: opencode session {entry['id']} is to be told, but "
                      f"{'no opencode server answers at ' + url if not up else 'the opencode program was not found'}; "
                      "the next Strata start tries again", flush=True)
                continue
            cwd = (lease or {}).get("cwd") or None
            if lease is None:                        # a register session: its folder from opencode's database
                cwd = session_info(entry["id"]).get("folder") or None
            args = exe + ["run", "--attach", url, "--session", entry["id"]]
            if cwd:
                args += ["--dir", str(cwd)]
            args.append(msg)
            ok = _launch(popen, args, cwd, log)
        else:
            exe = claude or claude_exe()
            if not exe:
                print(f"wake: Claude Code session {entry['id']} is to be told, but the claude program was not found",
                      flush=True)
                continue
            cwd = _claude_transcript_cwd(entry["id"], projects) or (lease or {}).get("cwd")
            ok = _launch(popen, exe + ["--resume", entry["id"], "-p", msg], cwd, log)
        if not ok:
            continue                                 # a failed launch: not marked, the next start tries again
        try:
            res = mark_woken(entry)
        except Exception as e:  # noqa: BLE001 - the marking is the server's; the run goes on
            print(f"wake: marking {entry['agent']} session {entry['id']} as told failed: {e}", flush=True)
            res = False
        if res is False:
            print(f"wake: the server did not mark {entry['agent']} session {entry['id']} as told", flush=True)
        told.append(entry)
        print(f"wake: {entry['agent']} session {entry['id']} told ({entry['cause']})", flush=True)
    return told
