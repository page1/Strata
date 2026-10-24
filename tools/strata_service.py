"""tools/strata_service.py - local: run the Strata server and start it again when it asks (POST /restart).

Run this (for example from a Windows boot task or a service) instead of serve/server.py:

    python tools/strata_service.py -- serve/server.py --engine strata --config strata-iq3_s.json --port 8080

It starts the server with the same Python and arguments. When the server ends with RESTART_EXIT_CODE (75: POST
/restart saved the conversations and stopped), it is started again at once, with whatever code is now in the folder -
so a deploy (git checkout, then POST /restart) needs no administrator. Any other exit ends this process with the
same code, so the task's own restart-on-failure still applies to a crash. More than RESTART_LIMIT restarts within
RESTART_WINDOW_S end it too (a restart loop is a bug, not a deploy).

The server is in a kill-on-close job (serve/winjob.py): when the task is stopped, this process ends and the server and
its engine end with it.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESTART_EXIT_CODE = 75                       # serve/server.py's RESTART_EXIT_CODE
RESTART_LIMIT, RESTART_WINDOW_S = 5, 600


def run(argv: list[str], popen=subprocess.Popen, contain=None, now=time.monotonic) -> int:
    """Start `python <argv>` until it exits with something other than RESTART_EXIT_CODE; -> that exit code."""
    if contain is None:
        from serve import winjob
        contain = winjob.contain
    restarts: list[float] = []
    while True:
        proc = popen([sys.executable, *argv], cwd=str(ROOT))
        contain(proc)
        code = proc.wait()
        if code != RESTART_EXIT_CODE:
            return code
        t = now()
        restarts = [r for r in restarts if t - r < RESTART_WINDOW_S] + [t]
        if len(restarts) > RESTART_LIMIT:
            print(f"[strata_service] {len(restarts)} restarts in {RESTART_WINDOW_S // 60} min: stopping", flush=True)
            return 1
        print("[strata_service] the server asked for a restart: starting it again", flush=True)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        print("usage: strata_service.py -- serve/server.py <server arguments>")
        return 2
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
