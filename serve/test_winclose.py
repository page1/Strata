"""Local "save_on_close": serve/winclose.py's decision and serve/winjob.py's release (no real close event, no GPU)."""
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve import winclose, winjob  # noqa: E402


class Dispatch(unittest.TestCase):
    def test_close_logoff_shutdown_call_on_close(self):
        for event in (winclose.CTRL_CLOSE_EVENT, winclose.CTRL_LOGOFF_EVENT, winclose.CTRL_SHUTDOWN_EVENT):
            calls = []
            self.assertTrue(winclose.dispatch(event, lambda: calls.append(1)))
            self.assertEqual(calls, [1])

    def test_ctrl_c_and_break_left_to_python(self):
        for event in (0, 1):
            calls = []
            self.assertFalse(winclose.dispatch(event, lambda: calls.append(1)))
            self.assertEqual(calls, [])

    def test_a_raising_on_close_still_handles(self):
        def boom():
            raise RuntimeError("x")
        self.assertTrue(winclose.dispatch(winclose.CTRL_CLOSE_EVENT, boom))


@unittest.skipUnless(os.name == "nt", "Windows job objects")
class Release(unittest.TestCase):
    def test_released_child_outlives_the_job_handle(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.assertTrue(winjob.contain(child))
            self.assertTrue(winjob.release())
            handle, winjob._job = winjob._job, None           # what the OS does when this process ends
            winjob._k32.CloseHandle(handle)
            time.sleep(0.5)
            self.assertIsNone(child.poll(), "the child must survive the job's last handle after release()")
        finally:
            child.kill()
            child.wait()

    def test_without_release_the_job_kills_the_child(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.assertTrue(winjob.contain(child))
            handle, winjob._job = winjob._job, None
            winjob._k32.CloseHandle(handle)
            for _ in range(50):
                if child.poll() is not None:
                    break
                time.sleep(0.1)
            self.assertIsNotNone(child.poll(), "kill-on-close: the child ends with the job's last handle")
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()


if __name__ == "__main__":
    unittest.main()
