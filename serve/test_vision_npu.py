"""The vision section's opt-in "npu" (docs/INTEL_NPU.md, "In this fork"): the Intel NPU picture reader is tried first,
and a PC without an NPU, its driver or OpenVINO gets the section's own reader instead - no config, no change.

    python -m serve.test_vision_npu
"""
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import serve.server as server

SECTION = {"exe": "strata-vision", "mmproj": "m.gguf", "model": "t.gguf", "max_tokens": 2048}


class Args(unittest.TestCase):
    def test_no_npu_no_reader(self):
        self.assertIsNone(server.npu_reader_args(SECTION))
        self.assertIsNone(server.npu_reader_args({**SECTION, "npu": {"anysize": "x", "enabled": False}}))
        self.assertIsNone(server.npu_reader_args({**SECTION, "npu": "yes"}))

    def test_needs_an_export(self):
        with mock.patch("builtins.print") as p:
            self.assertIsNone(server.npu_reader_args({**SECTION, "npu": {"python": "py"}}))
        self.assertIn("neither", p.call_args[0][0])

    def test_anysize_command_line(self):
        args = server.npu_reader_args({**SECTION, "min_tokens": 64,
                                       "npu": {"python": "py", "anysize": "exp", "cache_dir": "cache", "device": "NPU"}})
        self.assertEqual(args[:2], ["py", "-I"])
        self.assertTrue(args[2].endswith(os.path.join("tools", "vision", "strata_vision_npu.py")))
        self.assertTrue(Path(args[2]).is_file())
        rest = args[3:]
        self.assertEqual(rest[rest.index("--anysize") + 1], os.path.abspath("exp"))
        self.assertEqual(rest[rest.index("--cache-dir") + 1], os.path.abspath("cache"))
        self.assertEqual(rest[rest.index("--device") + 1], "NPU")
        self.assertEqual(rest[rest.index("--max-tokens") + 1], "2048")      # the section's own bound
        self.assertEqual(rest[rest.index("--min-tokens") + 1], "64")
        self.assertNotIn("--gpu", rest)

    def test_fixed_ir_command_line(self):
        args = server.npu_reader_args({**SECTION, "npu": {"python": "py", "model_ir": "enc.xml"}})
        self.assertEqual(args[args.index("--model-ir") + 1], "enc.xml")
        self.assertNotIn("--anysize", args)


class Start(unittest.TestCase):
    """Vision._start with fake processes: what starts, in what order, and what is tried again."""

    def vision(self, npu_line):
        started = []

        def popen(what, args, **kw):
            started.append(args[0])
            line = npu_line if args[0] == "py" else "READY 2560\n"
            return SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(line), kill=lambda: None,
                                   wait=lambda timeout=None: 0, poll=lambda: None)

        v = server.Vision.__new__(server.Vision)
        v.spawn = (["strata-vision", "--mmproj", "m.gguf"], None, None)
        v.dir = Path(tempfile.gettempdir())
        v.npu_args = ["py", "-I", "strata_vision_npu.py", "--anysize", "exp"]
        v.npu_failed = False
        return v, started, popen

    def test_npu_ready_is_used(self):
        v, started, popen = self.vision("READY 2560\n")
        with mock.patch.object(server, "popen", popen), mock.patch.object(server, "contain"), \
                mock.patch("builtins.print") as p:
            v._start()
        self.assertEqual(started, ["py"])
        self.assertFalse(v.stopped)
        self.assertIn("Intel NPU", p.call_args[0][0])

    def test_no_npu_falls_back_and_is_not_tried_again(self):
        v, started, popen = self.vision("ERR this PC has no NPU that OpenVINO can see (it sees: CPU)\n")
        with mock.patch.object(server, "popen", popen), mock.patch.object(server, "contain"), \
                mock.patch("builtins.print") as p:
            v._start()
            self.assertEqual(started, ["py", "strata-vision"])
            self.assertTrue(v.npu_failed and not v.stopped)
            self.assertIn("own reader", p.call_args[0][0])
            v._start()                                   # a restart after an unload: straight to the own reader
        self.assertEqual(started, ["py", "strata-vision", "strata-vision"])

    def test_without_npu_nothing_changes(self):
        v, started, popen = self.vision("READY 2560\n")
        v.npu_args = None
        with mock.patch.object(server, "popen", popen), mock.patch.object(server, "contain"):
            v._start()
        self.assertEqual(started, ["strata-vision"])

    def test_own_reader_failure_still_raises(self):
        v, started, popen = self.vision("ERR no NPU\n")

        def popen2(what, args, **kw):
            started.append(args[0])
            return SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO("ERR no mmproj\n"), kill=lambda: None,
                                   wait=lambda timeout=None: 0, poll=lambda: None)

        with mock.patch.object(server, "popen", popen2), mock.patch.object(server, "contain"), \
                mock.patch("builtins.print"):
            with self.assertRaises(RuntimeError) as cm:
                v._start()
        self.assertIn("no mmproj", str(cm.exception))
        self.assertEqual(started, ["py", "strata-vision"])


if __name__ == "__main__":
    unittest.main()
