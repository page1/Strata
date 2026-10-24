"""tools/test_strata_service.py - local: the boot task's launcher restarts the server only when it asks (exit 75).

    python tools/test_strata_service.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_service as S  # noqa: E402


class FakeProc:
    def __init__(self, code):
        self.code = code

    def wait(self):
        return self.code


class Launcher(unittest.TestCase):
    def run_with(self, codes, times=None):
        started, contained = [], []
        codes, clock = list(codes), iter(times or range(0, 10 ** 6, 1000))

        def popen(cmd, cwd=None):
            started.append(cmd)
            return FakeProc(codes.pop(0))

        code = S.run(["serve/server.py", "--port", "8080"], popen=popen, contain=contained.append,
                     now=lambda: next(clock))
        return code, started, contained

    def test_a_restart_request_starts_it_again_with_the_same_arguments(self):
        code, started, contained = self.run_with([75, 75, 0])
        self.assertEqual(code, 0)
        self.assertEqual(len(started), 3)
        self.assertTrue(all(c == [sys.executable, "serve/server.py", "--port", "8080"] for c in started))
        self.assertEqual(len(contained), 3)                   # each run in the kill-on-close job

    def test_a_crash_ends_it_with_the_servers_code(self):
        self.assertEqual(self.run_with([3])[0:2], (3, [[sys.executable, "serve/server.py", "--port", "8080"]]))

    def test_a_restart_loop_is_stopped(self):
        code, started, _ = self.run_with([75] * 20, times=range(0, 200, 10))    # every 10 s
        self.assertEqual(code, 1)
        self.assertEqual(len(started), S.RESTART_LIMIT + 1)

    def test_restarts_spread_out_are_fine(self):
        code, started, _ = self.run_with([75] * 8 + [0], times=range(0, 10 ** 5, 3600))   # one an hour
        self.assertEqual((code, len(started)), (0, 9))

    def test_the_exit_code_matches_the_server(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from serve import server
        self.assertEqual(server.RESTART_EXIT_CODE, S.RESTART_EXIT_CODE)


if __name__ == "__main__":
    unittest.main()
