"""tools/test_strata_vision_npu.py - the Intel NPU picture reader, tested without an NPU.

Checks tools/vision/strata_vision_npu.py the way serve/server.py (Vision) drives it: the READY handshake, the
`ENC <image> <out>` / `QUIT` protocol, the SVE1 record the engine reads in --serve GENI, the errors that must come
back as one ERR line instead of a dead process, and the honest failures (no model, no OpenVINO, no NPU: ERR and a
non-zero exit, never a quiet stand-in).  The mock engine keeps every test runnable with no driver and no hardware.

    python -m unittest tools.test_strata_vision_npu        (or: python tools/test_strata_vision_npu.py)
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from tools.vision.strata_vision_npu import (
    MEAN,
    STD,
    SVE1_MAGIC,
    ReaderError,
    grid_of,
    parse_enc,
    pick_size,
    ir_paths,
    preprocess,
    triple_of,
)

CLI = ROOT / "tools" / "vision" / "strata_vision_npu.py"


def session(args, commands, cwd, env=None, timeout=120):
    """Run the reader with a scripted stdin: send every command, close the pipe, read everything back."""
    proc = subprocess.Popen(
        [sys.executable, str(CLI), *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
        cwd=str(cwd),
        env=env,
    )
    out, err = proc.communicate("\n".join(commands) + "\n", timeout=timeout)
    return proc.returncode, out.splitlines(), err


class TempCase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="strata-npu-test-"))
        self.png = self.dir / "picture.png"
        # a two-colour picture, so the mock's tokens differ across the grid
        im = Image.new("RGB", (200, 120))
        for x in range(im.width):
            for y in range(im.height):
                im.putpixel(
                    (x, y), (240, 30, 30) if x < im.width // 2 else (20, 20, 220)
                )
        im.save(self.png)
        self.addCleanup(shutil.rmtree, self.dir, True)

    def clean_env(self, **extra):
        """The server's environment minus this reader's own settings, so a developer's shell cannot bend the tests."""
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("STRATA_VISION_NPU_")
        }
        env.update(extra)
        return env


class TestProtocol(TempCase):
    def test_version_build_line(self):
        rc, out, _ = session(["--version"], [], self.dir, env=self.clean_env())
        self.assertEqual(rc, 0)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].startswith("strata-vision-npu"), out)

    def test_ready_encode_sve1_quit(self):
        """READY, one picture, the exact SVE1 bytes, clean QUIT - the whole resident cycle."""
        rc, out, _ = session(
            [
                "--mock",
                "--embd",
                "64",
                "--image-size",
                "448",
                "--patch-size",
                "16",
                "--merge-size",
                "2",
            ],
            [f"ENC {self.png} {self.dir / 'one.sve'}", "QUIT"],
            self.dir,
            env=self.clean_env(),
        )
        self.assertEqual(rc, 0, out)
        self.assertEqual(out[0], "READY 64")
        parts = out[1].split()
        self.assertEqual(parts[0], "OK")
        n, nx, ny = int(parts[1]), int(parts[2]), int(parts[3])
        self.assertEqual((nx, ny), (14, 14))  # 448 / (16 x 2)
        self.assertEqual(n, nx * ny)
        data = (self.dir / "one.sve").read_bytes()
        self.assertEqual(
            len(data), 20 + n * 64 * 4
        )  # header + float32 [n][n_embd], nothing else
        header = struct.unpack("<5i", data[:20])
        self.assertEqual(header, (SVE1_MAGIC, n, nx, ny, 64))
        embd = np.frombuffer(data[20:], dtype=np.float32).reshape(n, 64)
        self.assertFalse(np.isnan(embd).any() or np.isinf(embd).any())
        self.assertGreater(float(embd.std()), 1e-4)  # not one constant row repeated

    def test_rows_run_through_the_grid_row_major(self):
        """The two-colour picture: the left half's tokens must differ from the right half's, with x fastest."""
        session(
            ["--mock", "--embd", "32", "--image-size", "64"],
            [f"ENC {self.png} {self.dir / 'g.sve'}", "QUIT"],
            self.dir,
            env=self.clean_env(),
        )
        data = (self.dir / "g.sve").read_bytes()
        _, n, nx, ny, d = struct.unpack("<5i", data[:20])
        embd = np.frombuffer(data[20:], dtype=np.float32).reshape(n, d)
        self.assertEqual((nx, ny), (2, 2))
        left = [embd[y * nx + 0] for y in range(ny)]  # row i sits at x = i % nx
        right = [embd[y * nx + 1] for y in range(ny)]
        self.assertLess(
            float(np.linalg.norm(left[0] - left[1])), 1e-4
        )  # same column: the mock pools equal colour
        self.assertLess(float(np.linalg.norm(right[0] - right[1])), 1e-4)
        self.assertGreater(
            float(np.linalg.norm(left[0] - right[0])), 1e-3
        )  # red half against blue half

    def test_mock_is_deterministic(self):
        for run in ("a", "b"):
            session(
                ["--mock", "--embd", "32"],
                [f"ENC {self.png} {self.dir / f'{run}.sve'}", "QUIT"],
                self.dir,
                env=self.clean_env(),
            )
        self.assertEqual(
            (self.dir / "a.sve").read_bytes(), (self.dir / "b.sve").read_bytes()
        )

    def test_errors_answer_and_the_reader_lives(self):
        """A bad command, a missing picture and a file that is not one each get one ERR line; the next good
        picture still encodes, QUIT still exits 0 - one bad picture must not end the resident reader."""
        bad = self.dir / "not-an-image.png"
        bad.write_bytes(b"this is not a PNG")
        rc, out, _ = session(
            ["--mock", "--embd", "16"],
            [
                "DROP TABLE",
                "ENC",
                "ENC only-one-path",
                f"ENC {self.dir / 'gone.png'} x.sve",
                f"ENC {bad} {self.dir / 'bad.sve'}",
                f"ENC {self.png} {self.dir / 'ok.sve'}",
                "QUIT",
            ],
            self.dir,
            env=self.clean_env(),
        )
        self.assertEqual(rc, 0)
        self.assertEqual(out[0], "READY 16")
        for line in out[1:-1]:
            self.assertTrue(line.startswith("ERR "), line)
        self.assertTrue(
            out[-1].startswith("OK 256 "), out[-1]
        )  # the default 512 px: 16 x 16 tokens
        self.assertFalse((self.dir / "bad.sve").exists())

    def test_image_paths_may_hold_spaces(self):
        """The server never names a file with a space (#480), but a person at a terminal does: the image path ends
        at the last space, the way the C++ readers cut the line (which is why the output path stays plain)."""
        spaced = self.dir / "my picture.png"
        shutil.copy(self.png, spaced)
        rc, out, _ = session(
            ["--mock", "--embd", "16"],
            [f"ENC {spaced} {self.dir / 'spacious_out.sve'}", "QUIT"],
            self.dir,
            env=self.clean_env(),
        )
        self.assertEqual(rc, 0)
        self.assertTrue(out[1].startswith("OK "), out)
        self.assertTrue((self.dir / "spacious_out.sve").is_file())

    def test_crlf_lines(self):
        """A hand-typed line on Windows arrives with \\r\\n; the C++ readers strip it, so this one must too."""
        proc = subprocess.Popen(
            [sys.executable, str(CLI), "--mock", "--embd", "16"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            cwd=str(self.dir),
            env=self.clean_env(),
        )
        proc.stdin.write(f"ENC {self.png} {self.dir / 'crlf.sve'}\r\nQUIT\r\n")
        proc.stdin.close()
        try:
            out = proc.stdout.read().splitlines()
            self.assertEqual(proc.wait(timeout=30), 0)
        finally:
            proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        self.assertEqual(out[0], "READY 16")
        self.assertTrue(out[1].startswith("OK "), out)


class TestHonestFailures(TempCase):
    """No NPU, no OpenVINO or no model: ERR and a non-zero exit (the server's CPU-fallback path), never a mock
    the caller did not ask for - a stand-in answering for a real encoder would put noise where a picture belongs."""

    def test_no_model_and_no_mock(self):
        rc, out, _ = session([], [], self.dir, env=self.clean_env())
        self.assertNotEqual(rc, 0)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].startswith("ERR "), out)
        self.assertIn("--model-ir", out[0])

    def test_unreadable_model(self):
        rc, out, _ = session(
            ["--model-ir", str(self.dir / "nope.xml")],
            [],
            self.dir,
            env=self.clean_env(),
        )
        self.assertNotIn(rc, (0,))
        self.assertTrue(out[0].startswith("ERR "), out)
        self.assertFalse(any(line.startswith("READY") for line in out))

    def test_mock_arrives_through_the_env_too(self):
        """A run config names an exe, never flags: STRATA_VISION_NPU_MOCK=1 is the documented way in."""
        env = self.clean_env(STRATA_VISION_NPU_MOCK="1")
        rc, out, _ = session([], ["QUIT"], self.dir, env=env)
        self.assertEqual(rc, 0, out)
        self.assertEqual(out[0], "READY 2048")  # the documented mock width


class TestServerVision(TempCase):
    """The reader as serve/server.py starts it: one exe from the vision section, the reader's own settings
    through the process environment, the work dir as its cwd."""

    @unittest.skipUnless(
        os.name == "posix",
        "the .py exe needs the shebang; a Windows run config names strata-vision-npu.cmd",
    )
    def test_vision_class_encodes_and_caches(self):
        from serve import server

        cfg = {
            "exe": str(CLI),
            "mmproj": "mmproj.gguf",
            "model": "model.gguf",
            "max_tokens": 1024,
        }
        env = self.clean_env(STRATA_VISION_NPU_MOCK="1", STRATA_VISION_NPU_DEVICE="NPU")
        v = server.Vision(cfg, env=env)
        try:
            self.assertTrue(v.alive())
            sve, n = v.encode(str(self.png))
            self.assertTrue(Path(sve).is_file())
            self.assertEqual(n, 256)  # 512 / (16 x 2), squared
            header = struct.unpack("<5i", Path(sve).read_bytes()[:20])
            self.assertEqual(header[0], SVE1_MAGIC)
            again, n2 = v.encode(str(self.png))
            self.assertEqual(
                (str(again), n2), (str(sve), n)
            )  # the hash cache: the same file, not a re-encode
        finally:
            v.close()
        self.assertFalse(v.alive())


class TestShapeRules(unittest.TestCase):
    """The pure pieces: the ENC cut, the token grid, the size an IR shape implies."""

    def test_parse_enc(self):
        self.assertEqual(parse_enc("ENC a.png b.sve"), ("a.png", "b.sve"))
        self.assertEqual(
            parse_enc("ENC two words.png out .sve"), ("two words.png out", ".sve")
        )
        self.assertIsNone(parse_enc("ENC alone"))
        self.assertIsNone(parse_enc("ENC  out.sve"))  # an empty image path
        self.assertIsNone(parse_enc("QUIT"))

    def test_ir_paths(self):
        import os

        self.assertEqual(ir_paths("one.xml"), ["one.xml"])
        joined = os.pathsep.join(["a.xml", "b.xml"])
        self.assertEqual(ir_paths(joined), ["a.xml", "b.xml"])
        self.assertEqual(ir_paths(f" {joined} "), ["a.xml", "b.xml"])  # typed spaces
        with self.assertRaises(Exception):
            ir_paths("  ")
        with self.assertRaises(Exception):
            ir_paths(os.pathsep.join(["", " "]))

    def test_grid_of(self):
        self.assertEqual(grid_of(512, 512, 16, 2), (16, 16))
        self.assertEqual(grid_of(448, 224, 16, 2), (14, 7))
        with self.assertRaises(Exception):
            grid_of(500, 512, 16, 2)  # not a multiple of the merged patch

    def test_pick_size_static_wins_and_bounds_are_checked(self):
        self.assertEqual(pick_size([1, 3, 448, 448], 512, 16, 2, 0, 0), (448, 448))
        self.assertEqual(pick_size([1, 3, 512, 512], 512, 16, 2, 256, 256), (512, 512))
        with self.assertRaises(Exception):  # 256 tokens > --max-tokens 64
            pick_size([1, 3, 512, 512], 512, 16, 2, 64, 0)
        with self.assertRaises(Exception):  # 256 tokens < --min-tokens 512
            pick_size([1, 3, 512, 512], 512, 16, 2, 0, 512)

    def test_pick_size_dynamic(self):
        self.assertEqual(pick_size([1, 3, -1, -1], 512, 16, 2, 0, 0), (512, 512))
        self.assertEqual(
            pick_size([1, 3, -1, -1], 512, 16, 2, 256, 0), (512, 512)
        )  # isqrt(256) x 32
        self.assertEqual(
            pick_size([1, 3, -1, -1], 512, 16, 2, 100, 0), (320, 320)
        )  # isqrt(100)=10 sides of 32 px
        self.assertEqual(
            pick_size([1, 3, -1, -1], 500, 16, 3, 0, 0), (480, 480)
        )  # 500 snapped down to a multiple of 16 x 3
        self.assertEqual(
            pick_size([1, 3, 384, -1], 512, 16, 2, 256, 0), (512, 384)
        )  # the static side stays
        self.assertEqual(
            pick_size([1, 3, 384, -1], 512, 16, 2, 0, 64), (256, 384)
        )  # ceil(sqrt(64)) x 32 for the dynamic W
        with self.assertRaises(Exception):
            pick_size([3, 512, 512], 512, 16, 2, 0, 0)  # 3-dimensional


class TestNormalization(unittest.TestCase):
    """The (byte/255 - mean) / std the exported IR expects: 0.5/0.5 for the Qwen3.8-Flash-Next encoder, with
    --image-mean/--image-std and STRATA_VISION_NPU_MEAN/STD for an IR exported from another tower."""

    def test_defaults_are_the_flash_next_constants(self):
        self.assertEqual(MEAN, (0.5, 0.5, 0.5))
        self.assertEqual(STD, (0.5, 0.5, 0.5))

    def test_triple_of(self):
        self.assertEqual(triple_of("", (1.0, 2.0, 3.0), "mean"), (1.0, 2.0, 3.0))
        self.assertEqual(triple_of("0.3", (0.5, 0.5, 0.5), "mean"), (0.3, 0.3, 0.3))
        self.assertEqual(
            triple_of("0.1, 0.2, 0.3", (0.0, 0.0, 0.0), "std"), (0.1, 0.2, 0.3)
        )
        for bad in ("a,b,c", "1,2", "1,2,3,4", "1,,2,,"):
            with self.assertRaises(ReaderError, msg=bad):
                triple_of(bad, (0.5, 0.5, 0.5), "mean")
        for bad in ("1,0,1", "1,-2,1"):  # a zero or negative std divides by zero
            with self.assertRaises(ReaderError, msg=bad):
                triple_of(bad, (0.5, 0.5, 0.5), "std")
        # a mean may be negative: it is only shifted, never divided
        self.assertEqual(
            triple_of("-1,0.5,2", (0.5, 0.5, 0.5), "mean"), (-1.0, 0.5, 2.0)
        )

    def test_preprocess_applies_the_triples(self):
        with tempfile.TemporaryDirectory() as td:
            png = Path(td) / "flat.png"
            Image.new("RGB", (64, 48), (255, 128, 0)).save(png)
            got = preprocess(str(png), 64, 48)  # the 0.5/0.5 default
            self.assertEqual(got.shape, (1, 3, 48, 64))  # CHW, batch 1
            want = np.array([1.0, (128 / 255 - 0.5) / 0.5, -1.0], np.float32)
            for c in range(
                3
            ):  # solid colour: every pixel of every channel is the same value
                self.assertTrue(
                    np.allclose(got[0, c], want[c], atol=1e-5),
                    f"channel {c} averaged {got[0, c].mean()}",
                )
            clip = preprocess(
                str(png),
                64,
                48,
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            )  # a CLIP tower's constants: same code path, different numbers
            self.assertFalse(np.allclose(clip, got))


class TestNormalizationFlags(TempCase):
    def test_bad_triples_are_one_err_line(self):
        rc, out, _ = session(
            ["--mock", "--embd", "16", "--image-std", "0,1,1"],
            [],
            self.dir,
            env=self.clean_env(),
        )
        self.assertNotEqual(rc, 0)
        self.assertTrue(out and out[0].startswith("ERR "), out)
        rc, out, _ = session(
            ["--mock", "--embd", "16"],
            [],
            self.dir,
            env=self.clean_env(STRATA_VISION_NPU_MEAN="wide"),
        )
        self.assertNotEqual(rc, 0)
        self.assertTrue(out and out[0].startswith("ERR "), out)

    def test_flags_and_env_still_encode(self):
        rc, out, _ = session(
            [
                "--mock",
                "--embd",
                "16",
                "--image-mean",
                "0.48145466",
                "--image-std",
                "0.26862954",
            ],
            [f"ENC {self.png} {self.dir / 'clip.sve'}", "QUIT"],
            self.dir,
            env=self.clean_env(),
        )
        self.assertEqual(rc, 0, out)
        self.assertEqual(out[0], "READY 16")
        self.assertTrue(out[1].startswith("OK "), out)


class CacheDir(unittest.TestCase):
    """--cache-dir / STRATA_VISION_NPU_CACHE_DIR: OpenVINO's CACHE_DIR is set on the chosen device,
    so the slow NPU compile (measured 170 s + 190 s for the chain, 1.2 s + 1.3 s from a warm cache)
    is paid once.  A fake openvino stands in; no driver and no hardware here."""

    def build(self, cache_dir):
        import importlib.util
        import types

        calls = []

        class Side:
            shape = [1, 1536, 768, 768]

        class Comp:
            def input(self, _i):
                return Side()

            def output(self, _i):
                return Side()

            def create_infer_request(self):
                return object()

        ov = types.ModuleType("openvino")

        class Core:
            available_devices = ["CPU", "NPU"]

            def set_property(self, device, cfg):
                calls.append((device, dict(cfg)))

            def read_model(self, _path):
                return object()

            def compile_model(self, _model, _device, _cfg):
                return Comp()

        ov.Core = Core
        ov.get_version = lambda: "fake"
        spec = importlib.util.spec_from_file_location(
            "strata_vision_npu_cache_test",
            ROOT / "tools" / "vision" / "strata_vision_npu.py",
        )
        mod = importlib.util.module_from_spec(spec)
        saved = sys.modules.get("openvino")
        sys.modules["openvino"] = ov
        try:
            spec.loader.exec_module(mod)
            mod.OpenVinoEngine("one.xml", "NPU", cache_dir)
        finally:
            if saved is not None:
                sys.modules["openvino"] = saved
            else:
                sys.modules.pop("openvino", None)
        return calls

    def test_cache_dir_is_set_on_the_chosen_device_and_created(self):
        d = Path(tempfile.mkdtemp(prefix="cache-dir-")) / "nested" / "ov-cache"
        self.assertEqual(self.build(str(d)), [("NPU", {"CACHE_DIR": str(d)})])
        self.assertTrue(d.is_dir())

    def test_no_cache_dir_means_no_set_property(self):
        self.assertEqual(self.build(""), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
