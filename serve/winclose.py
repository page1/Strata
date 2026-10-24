"""serve/winclose.py - local ("save_on_close"): on Windows, act when the console window is closed.

Closing the window (its X, or the tab in Windows Terminal), logging off or shutting down sends the console's
processes a close event, and Windows ends them about 5 s later whatever they are doing - too short to write gigabytes
of conversations.  `install(on_close)` registers a console control handler that calls `on_close()` for those events
(not for Ctrl+C or Ctrl+Break, which take the server's normal stop).  The caller makes `on_close` quick: it lets the
engine outlive the server (winjob.release) and closes its stdin, and the engine, started without the console
(StrataEngine.detach_console), writes its conversations and ends on its own.  Elsewhere a no-op.
"""
from __future__ import annotations

import os

_handler = None                                      # the ctypes callback must outlive install()

CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT = 2, 5, 6


def dispatch(event: int, on_close) -> bool:
    """The handler's decision: True (handled) after on_close() for a close, log-off or shutdown; False for the rest
    (Ctrl+C 0, Ctrl+Break 1), which Python handles as before."""
    if event not in (CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT):
        return False
    try:
        on_close()
    except Exception:  # noqa: BLE001 - a handler must not raise
        pass
    return True


def install(on_close) -> bool:
    """Call on_close() when the console is closed, on log-off or at shutdown.  True if the handler is registered."""
    global _handler
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    routine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

    _handler = routine(lambda event: dispatch(event, on_close))
    k32.SetConsoleCtrlHandler.argtypes = (routine, wintypes.BOOL)
    return bool(k32.SetConsoleCtrlHandler(_handler, True))
