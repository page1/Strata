"""leases: the lease table of the Strata lock (the holds.json replacement).

A heavy job's wrapper registers a lease here and renews it every ~10 s; a lease nobody renews for expire_s (45 s)
expires, so a crash or a power cut needs no special handling: the renewals just stop.  Several agents create, renew
and end leases at the same time, so the table is guarded by one RLock and every file is written through a unique
temp name with flush + fsync + os.replace (retried on Windows sharing), as _write_holds in serve/server.py does.

One process at a time owns the table: an exclusive, non-blocking OS lock on <folder>/.writer.lock; a second opener
gets LeaseTableBusy.  Each lease lives in <folder>/<id>.json (id = uuid4().hex); checkpoint.json and state.json are
the table's own files.  A lease file that does not parse is renamed to <name>.json.bad and holds the model as an
unknown owner until expire_s has passed since this table was opened; *.tmp files are writes that were cut off and
are ignored."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

_HEX = frozenset("0123456789abcdef")


class LeaseEnded(Exception):
    """the lease you tried to renew has ended (released, expired or overridden)."""


class LeaseTableBusy(Exception):
    """another process holds this table's writer lock."""


def _looks_like_id(name: str) -> bool:
    return len(name) == 32 and all(c in _HEX for c in name)


class LeaseTable:
    def __init__(self, path: Path, expire_s: float = 45.0, clock=time.time):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.expire_s = float(expire_s)
        self.clock = clock
        self._lock = threading.RLock()
        self._fh = None
        self._acquire_writer_lock()
        self._opened = self.clock()
        self._leases: dict[str, dict] = {}
        self._unknown: dict[str, float] = {}          # unreadable id -> when this table opened
        self._resume_pending = False
        self._load()

    # ---------------------------------------------------------------- writer lock

    def _acquire_writer_lock(self) -> None:
        self._fh = open(self.dir / ".writer.lock", "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self._fh.seek(0)                             # lock byte 0, as close() unlocks byte 0: msvcrt locks
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)   # from the current position (a+b is at the end)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._fh.close()
            self._fh = None
            raise LeaseTableBusy(f"{self.dir}: another process holds the lease table") from None

    def close(self) -> None:
        if self._fh is not None:
            try:
                if os.name == "nt":
                    import msvcrt
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            self._fh.close()
            self._fh = None

    # ---------------------------------------------------------------- loading

    def _load(self) -> None:
        for p in sorted(self.dir.glob("*.json")):
            if p.name in ("checkpoint.json", "state.json"):
                continue
            if not _looks_like_id(p.stem):
                continue
            lease = None
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("id") == p.stem:
                    lease = data
            except (OSError, ValueError):
                pass
            if lease is None:
                try:
                    os.replace(p, p.with_name(p.name + ".bad"))
                except OSError:
                    pass
                self._unknown[p.stem] = self._opened
            else:
                lease.setdefault("ended", None)
                self._leases[p.stem] = lease
        try:
            state = json.loads((self.dir / "state.json").read_text(encoding="utf-8"))
            self._resume_pending = bool(state.get("resume_pending"))
        except (OSError, ValueError, AttributeError):
            self._resume_pending = False

    # ---------------------------------------------------------------- writing

    def _write(self, path: Path, obj: dict) -> None:
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "w", encoding="utf-8") as out:
            json.dump(obj, out)
            out.flush()
            os.fsync(out.fileno())                       # on the disk before it replaces the old one (power cuts)
        for attempt in range(20):                        # Windows: no replace while a reader has it open
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)

    def _lease_path(self, lease_id: str) -> Path:
        return self.dir / f"{lease_id}.json"

    def _write_state(self) -> None:
        self._write(self.dir / "state.json", {"resume_pending": self._resume_pending})

    def _live(self) -> list[dict]:
        return [l for l in self._leases.values() if l["ended"] is None]

    def _note_period_end(self, ended: list[dict]) -> None:
        """The last live lease of a held period ended: if that period had resume_model, the resume is pending."""
        if not self._live() and any(l.get("resume_model") for l in ended):
            self._resume_pending = True
            self._write_state()

    # ---------------------------------------------------------------- the contract

    def _new_resume_model(self, model_loaded: bool) -> bool:
        """resume_model for a lease starting now: inherited from a live lease; if none is live and a resume is
        pending, the new held period inherits it (the pending flag moves into the period and is cleared); otherwise
        it is whether the model was loaded when this first lease started."""
        live = self._live()
        if live:
            return any(bool(l.get("resume_model")) for l in live)
        if self._resume_pending:
            self._resume_pending = False
            self._write_state()
            return True
        return bool(model_loaded)

    def create(self, cmd, cwd, mode, session=None, account=None, model_loaded=False, log=None) -> dict:
        with self._lock:
            now = self.clock()
            lease = {"id": uuid.uuid4().hex, "account": account, "cmd": cmd, "cwd": cwd, "mode": mode,
                     "session": session, "created": now, "renewed": now, "ended": None, "exit": None,
                     "resume_model": self._new_resume_model(model_loaded), "woken": False, "version": 1, "log": log}
            self._leases[lease["id"]] = lease
            self._write(self._lease_path(lease["id"]), lease)
            return dict(lease)

    def renew(self, lease_id: str) -> dict:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:                            # the server lost its files; the holder is clearly alive
                now = self.clock()
                lease = {"id": lease_id, "account": None, "cmd": "", "cwd": "", "mode": "full", "session": None,
                         "created": now, "renewed": now, "ended": None, "exit": None,
                         "resume_model": self._new_resume_model(False), "woken": False,
                         "version": 1}
                self._leases[lease_id] = lease
                self._write(self._lease_path(lease_id), lease)
                return dict(lease)
            if lease["ended"] is not None:
                raise LeaseEnded(f"lease {lease_id} ended ({lease['ended']})")
            lease["renewed"] = self.clock()
            lease["version"] += 1
            self._write(self._lease_path(lease_id), lease)
            return dict(lease)

    def end(self, lease_id: str, exit=None) -> dict:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise KeyError(f"no lease {lease_id}")
            if lease["ended"] is None:
                lease["ended"] = "released"
                lease["exit"] = exit
                lease["version"] += 1
                self._write(self._lease_path(lease_id), lease)
                self._note_period_end([lease])
            return dict(lease)

    def stop(self, lease_id: str) -> dict:
        """The user ends this lease: ended "stopped", stopped_by_user the time.  An already ended lease is returned
        unchanged (stopping it twice is fine); an unknown id raises KeyError.  The caller then runs the pass that
        brings the model back when this was the last live lease."""
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise KeyError(f"no lease {lease_id}")
            if lease["ended"] is None:
                lease["ended"] = "stopped"
                lease["stopped_by_user"] = self.clock()
                lease["version"] += 1
                self._write(self._lease_path(lease_id), lease)
                self._note_period_end([lease])
            return dict(lease)

    def stop_all(self) -> list[dict]:
        """Stop every live lease (the web page's Stop button, `stop --all`): each ends "stopped" with stopped_by_user;
        -> the leases that were stopped."""
        with self._lock:
            stopped = []
            for lease in self._leases.values():
                if lease["ended"] is None:
                    lease["ended"] = "stopped"
                    lease["stopped_by_user"] = self.clock()
                    lease["version"] += 1
                    self._write(self._lease_path(lease["id"]), lease)
                    stopped.append(dict(lease))
            if stopped:
                self._note_period_end(stopped)
            return stopped

    def expire_pass(self) -> list[dict]:
        with self._lock:
            now = self.clock()
            gone = []
            for lease in self._leases.values():
                # after a server start a lease that was live on disk gets expire_s from the start (its holder renews
                # once the server answers again); the file keeps the last real renewal
                if lease["ended"] is None and max(lease["renewed"], self._opened) < now - self.expire_s:
                    lease["ended"] = "expired"
                    lease["version"] += 1
                    self._write(self._lease_path(lease["id"]), lease)
                    gone.append(dict(lease))
            if gone:
                self._note_period_end(gone)
            if now - self._opened >= self.expire_s:      # unknown-owner holds stop after the grace
                self._unknown = {}
            return gone

    def refresh_all(self) -> list[dict]:
        """A pause (the PC slept or this process froze): every live lease is renewed to now, so the gap in the
        renewals does not expire anything."""
        with self._lock:
            now = self.clock()
            out = []
            for lease in self._leases.values():
                if lease["ended"] is None:
                    lease["renewed"] = now
                    lease["version"] += 1
                    self._write(self._lease_path(lease["id"]), lease)
                    out.append(dict(lease))
            return out

    def resume_pending(self) -> bool:
        """Ended leases left a resume to do that has not run yet (no lease holds now)."""
        with self._lock:
            return self._resume_pending and not self._live()

    def restore_resume(self) -> None:
        """A resume that could not run (the memory was not free): put the pending flag back for the next pass."""
        with self._lock:
            if not self._live():
                self._resume_pending = True
                self._write_state()

    def clear_resume(self) -> None:
        """A deliberate hibernate (the web page's, not a job's): a resume still pending from ended leases (the memory
        was not free) must not bring the model back later."""
        with self._lock:
            if not self._live() and self._resume_pending:
                self._resume_pending = False
                self._write_state()

    def override_all(self) -> list[dict]:
        with self._lock:
            over = []
            for lease in self._leases.values():
                if lease["ended"] is None:
                    lease["ended"] = "overridden"
                    lease["version"] += 1
                    self._write(self._lease_path(lease["id"]), lease)
                    over.append(dict(lease))
            self._resume_pending = False                 # a forced load cancels the pending resume
            self._write_state()
            return over

    def get(self, lease_id: str) -> dict:
        with self._lock:
            lease = self._leases.get(lease_id)
            return dict(lease) if lease is not None else None

    def live(self) -> list[dict]:
        with self._lock:
            return [dict(l) for l in sorted(self._live(), key=lambda l: (l["created"], l["id"]))]

    def safe_to_load(self) -> bool:
        with self._lock:
            if self._live():
                return False
            now = self.clock()
            return not any(now - since < self.expire_s for since in self._unknown.values())

    def take_resume(self) -> bool:
        with self._lock:
            if self._live():                               # never load the model over a job that holds the lock
                return False
            if self._resume_pending:
                self._resume_pending = False
                self._write_state()
                return True
            return False

    def to_wake(self) -> list[dict]:
        with self._lock:
            out = [dict(l) for l in sorted(self._leases.values(), key=lambda l: (l["created"], l["id"]))
                   if l["ended"] == "expired" and l.get("session") and not l.get("woken")]
            return out

    def mark_woken(self, lease_id: str) -> dict:
        """The waker told this lease's session: mark it woken on disk at once.  Twice is fine; unknown id: KeyError."""
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise KeyError(f"no lease {lease_id}")
            if not lease.get("woken"):
                lease["woken"] = True
                lease["version"] += 1
                self._write(self._lease_path(lease_id), lease)
            return dict(lease)

    def checkpoint(self, register: dict, files: list, **extra) -> None:
        """One atomic write under the lock; extra keys (e.g. shutting_down=True) ride in the same write, so a
        concurrent beat can never overwrite them."""
        with self._lock:
            cp = {"time": self.clock(), "leases": self.live(), "register": register, "files": files, "woken": []}
            cp.update(extra)
            self._write(self.dir / "checkpoint.json", cp)

    def mark_session_woken(self, agent: str, session_id: str) -> None:
        """A session of the previous run's register was told: add [agent, id] to checkpoint.prev.json's "woken" list
        (where the waker reads the register), so a session is told once.  This table is the only writer, under its
        lock, the usual atomic write."""
        with self._lock:
            path = self.dir / "checkpoint.prev.json"
            try:
                cp = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(cp, dict):
                    cp = None
            except (OSError, ValueError):
                cp = None
            if cp is None:
                cp = {"time": self.clock(), "leases": [], "register": {}, "files": []}
            woken = cp.get("woken")
            woken = [w for w in woken if isinstance(w, list) and len(w) == 2] if isinstance(woken, list) else []
            if [agent, session_id] not in woken:
                woken.append([agent, session_id])
            cp["woken"] = woken
            self._write(path, cp)

    def read_checkpoint(self) -> dict | None:
        with self._lock:
            try:
                return json.loads((self.dir / "checkpoint.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
