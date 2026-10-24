"""local: the web app's chat kept on the server (the run config's "chat_dir"), so a chat started on one device goes on
on another: every page loads it, saves to it, and looks for a newer version while it is idle.

One current chat, with a version that goes up on every save.  A save names the version it was made from; if another
device saved in between, it is refused (409) with the current chat, and that page takes it.  "archive": true (the
page's New chat) keeps the chat it replaces in archive/<time>.json.  Writes go to a temporary file first, then replace
current.json, so a crash never leaves half a chat."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path


class ChatStore:
    def __init__(self, folder):
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "current.json"
        self.lock = threading.Lock()
        self.version, self.messages, self.updated = 0, [], 0.0
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            self.version = int(saved.get("version") or 0)
            self.messages = saved.get("messages") if isinstance(saved.get("messages"), list) else []
            self.updated = float(saved.get("updated") or 0)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError) as e:          # unreadable: keep it aside, start empty
            print(f"[strata] chat_dir: {self.path} cannot be read ({e}); starting with an empty chat", flush=True)
            try:
                self.path.replace(self.path.with_name(f"current.unreadable-{int(time.time())}.json"))
            except OSError:
                pass

    def state(self, messages=True) -> dict:
        with self.lock:
            out = {"version": self.version, "updated": self.updated}
            if messages:
                out["messages"] = self.messages
            return out

    def _write(self, path: Path, data: dict):
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)

    def save(self, base: int, messages: list, archive: bool = False) -> tuple[bool, dict]:
        """(True, {"version"}) or (False, the current state) when another device saved since `base`."""
        with self.lock:
            if base != self.version:
                return False, {"version": self.version, "updated": self.updated, "messages": self.messages}
            if archive and self.messages:
                folder = self.dir / "archive"
                folder.mkdir(exist_ok=True)
                self._write(folder / f"{time.strftime('%Y-%m-%d-%H%M%S')}-v{self.version}.json",
                            {"version": self.version, "updated": self.updated, "messages": self.messages})
            version, updated = self.version + 1, time.time()
            self._write(self.path, {"version": version, "updated": updated, "messages": messages})
            self.version, self.messages, self.updated = version, messages, updated
            return True, {"version": version, "updated": updated}
