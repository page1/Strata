"""tools/test_npu_anysize.py - the any-size NPU encoder (tools/vision/npu_anysize.py and strata_vision_npu.py
--anysize), tested without an NPU or a model file.

The numpy half runs anywhere: mtmd's picture sizing and letterbox, the merged token order against
export_npu_encoder.C_idx, the resized position table, the rope tables against export_npu_encoder's own, the padding,
the budget choice, and the per-picture run with stand-in graphs.  The OpenVINO half (skipped without the wheel)
exports a tiny synthetic mmproj, runs it on OpenVINO's CPU device and checks it against export_npu_encoder.py's
fixed-size graph of the same file, against itself padded to a larger budget and with chunked attention, and through
the reader's protocol.

    python -m unittest tools.test_npu_anysize        (or: python tools/test_npu_anysize.py)
"""

from __future__ import annotations

import importlib.util
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "vision"))
sys.path.insert(0, str(ROOT / "tools"))

import npu_anysize as A  # noqa: E402
import export_npu_encoder as ex  # noqa: E402
from test_export_npu_encoder import Tiny, build  # noqa: E402

HAVE_OV = importlib.util.find_spec("openvino") is not None
READER = ROOT / "tools" / "vision" / "strata_vision_npu.py"


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


class TestSizing(unittest.TestCase):
    def test_target_size_hand_numbers(self):
        """mtmd's calc_size_preserved_ratio, checked against what strata-vision reported for the same sizes
        (a 2000x920 screenshot: 987 tokens 47x21 at max 1024, 1827 tokens 63x29 at max 2048)."""
        self.assertEqual(A.target_size(2000, 920, 1024), (1504, 672))
        self.assertEqual(A.target_size(2000, 920, 2048), (2016, 928))
        self.assertEqual(A.target_size(768, 768, 1024), (768, 768))  # fits: only rounded to 32
        self.assertEqual(A.target_size(3200, 1910, 2048), (1856, 1088))  # 58 x 34 = 1972 tokens
        self.assertEqual(A.target_size(20, 20, 1024, 8), (96, 96))  # raised to the 8-token floor

    def test_fit_letterboxes_instead_of_stretching(self):
        """PAD_CEIL: a 100x50 red picture in a 64x64 box is scaled to 64x32, centred, black above and below."""
        out = np.asarray(A.fit(Image.new("RGB", (100, 50), (255, 0, 0)), 64, 64))
        self.assertEqual(out.shape, (64, 64, 3))
        self.assertTrue((out[:15] == 0).all() and (out[-15:] == 0).all())  # the bands
        self.assertTrue((out[20:44, :, 0] > 200).all())  # the picture, full width

    def test_plan_picks_the_smallest_budget(self):
        side = {"budgets": [576, 1024, 2048]}
        self.assertEqual(A.plan(768, 768, side, 2048, 0), (768, 768, 576))
        self.assertEqual(A.plan(2000, 920, side, 1024, 0), (1504, 672, 1024))
        self.assertEqual(A.plan(2000, 920, side, 2048, 0), (2016, 928, 2048))
        with self.assertRaises(ValueError):
            A.plan(2000, 920, {"budgets": [576, 1024]}, 2048, 0)


class TestTables(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="strata-anysize-test-"))
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_merged_order_is_export_npu_encoders(self):
        """The merged order computed per picture equals the fixed export's C_idx at a square grid."""
        for g in (4, 48):
            x, y = A.merged_xy(g, g)
            self.assertTrue(np.array_equal(y * g + x, ex.C_idx({"g": g, "mg": g // 2, "npt": g * g})))

    def test_merged_order_non_square(self):
        """A 6 x 4 patch grid: token 0 is the top-left 2x2 block, token 1 the next two columns, and every patch
        appears once."""
        x, y = A.merged_xy(6, 4)
        self.assertEqual(list(zip(x[:8], y[:8])), [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 0), (2, 1), (3, 1)])
        self.assertEqual(sorted((y * 6 + x).tolist()), list(range(24)))

    def test_position_table_resize(self):
        """Native grid: the table itself.  Other grids: bilinear with align corners - the corners are the
        table's corners, and a doubled axis puts the average of two neighbours between them."""
        t = np.arange(16 * 2, dtype=np.float32).reshape(16, 2)  # a 4x4 grid of 2-wide rows
        self.assertIs(A.interp_pos(t, 4, 4), t)
        r = A.interp_pos(t, 7, 4).reshape(4, 7, 2)
        g = t.reshape(4, 4, 2)
        for yy, xx, sy, sx in ((0, 0, 0, 0), (0, 6, 0, 3), (3, 0, 3, 0), (3, 6, 3, 3)):
            self.assertTrue(np.allclose(r[yy, xx], g[sy, sx]))
        self.assertTrue(np.allclose(r[0, 1], (g[0, 0] + g[0, 1]) / 2))

    def test_rope_tables_match_the_fixed_export(self):
        _, _, C = build(Tiny(side=64), self.dir)
        x, y = A.merged_xy(C["g"], C["g"])
        cos, sin = A.rope_tables(x, y, C["dh"])
        self.assertTrue(np.allclose(cos, C["cos"]) and np.allclose(sin, C["sin"]))

    def test_inputs_pad_with_masked_rows(self):
        side = {"pos": np.zeros((16, 8), np.float32), "bias": np.zeros(8, np.float32), "dh": 8}
        inp = A.Inputs(np.zeros((3, 64, 32), np.float32), side, 4)  # 2x4 patches = 2 tokens, padded to 4
        self.assertEqual((inp.nx, inp.ny, inp.n_tokens), (1, 2, 2))
        self.assertEqual(inp.feed["patches"].shape, (16, 768))
        self.assertTrue((inp.feed["mask"][0, 0, :8] == 0).all() and (inp.feed["mask"][0, 0, 8:] == A.MASK_NEG).all())
        self.assertTrue((inp.feed["cos"][8:] == 1).all() and (inp.feed["sin"][8:] == 0).all())
        with self.assertRaises(ValueError):
            A.Inputs(np.zeros((3, 64, 64), np.float32), side, 2)


class TestRun(unittest.TestCase):
    """AnySizeEncoder with numpy stand-ins for the three graphs: what it feeds them, in what order, and what it
    keeps of the answer."""

    def side(self):
        return {"budgets": [4, 8], "n_layers": 3, "n_embd": 5, "dh": 8, "mean": [0.5] * 3, "std": [0.5] * 3,
                "pos": np.zeros((16, 8), np.float32), "bias": np.zeros(8, np.float32),
                "blocks": [{k: np.full(2, i, np.float16) for k in A.BLOCK_WEIGHTS} for i in range(3)]}

    def test_blocks_run_in_order_and_padding_is_dropped(self):
        calls = []

        def head(feed):
            return np.zeros((feed["patches"].shape[0], 8), np.float32)

        def block(feed):
            calls.append(int(feed["w_ob"][0]))
            self.assertEqual(set(feed), {"hidden", "cos", "sin", "mask"} | {"w_" + k for k in A.BLOCK_WEIGHTS})
            return feed["hidden"] + 1

        def tail(feed):
            h = feed["hidden"]
            return np.repeat(h[::4, :1], 5, axis=1)  # one row per token, value = blocks run

        side = self.side()
        enc = A.AnySizeEncoder(side, {4: (head, block, tail), 8: (head, block, tail)})
        inp = A.Inputs(np.zeros((3, 64, 32), np.float32), side, 8)  # 2 tokens in an 8-token budget
        out = enc.run(inp, 8)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(out.shape, (2, 5))
        self.assertTrue((out == 3).all())

    def test_max_tokens_above_the_export_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            A.AnySizeEncoder(self.side(), {}, max_tokens=16)
        self.assertIn("largest budget 8", str(cm.exception))


@unittest.skipUnless(HAVE_OV, "openvino is not installed")
class TestOpenVino(unittest.TestCase):
    """The exported graphs on OpenVINO's CPU device, against export_npu_encoder.py's fixed graph."""

    @classmethod
    def setUpClass(cls):
        import openvino as ov
        from openvino import opset13 as ops

        cls.dir = Path(tempfile.mkdtemp(prefix="strata-anysize-ov-"))
        cls.tiny = Tiny(side=128, layers=2)  # an 8x8 patch grid, 16 tokens at the native size
        mmproj = cls.tiny.file(cls.dir)
        cls.whole = cls.dir / "whole"
        cls.chunked = cls.dir / "chunked"
        A.export(str(mmproj), str(cls.whole), [16, 32], q_chunk=0)
        A.export(str(mmproj), str(cls.chunked), [16, 32], q_chunk=24)  # 64 rows -> 24, 24, 16: an uneven last chunk
        cls.core = ov.Core()
        meta, T = ex.load_gguf(str(mmproj))
        C = ex.build_constants(meta, T)
        cls.fixed = cls.core.compile_model(ex.build_model(ops, C, ln_scale="max"), "CPU")
        cls.png = cls.dir / "square.png"
        rng = np.random.default_rng(3)
        Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)).save(cls.png)
        cls.wide = cls.dir / "wide.png"
        Image.fromarray(rng.integers(0, 255, (40, 150, 3), dtype=np.uint8)).save(cls.wide)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, True)

    def pixels(self):
        with Image.open(self.png) as im:
            return (np.asarray(im.convert("RGB"), np.float32) / 255.0 - 0.5) / 0.5

    def test_native_size_equals_the_fixed_export(self):
        """At the export's own size the any-size graphs compute export_npu_encoder.py's function (fp16 weights
        here against fp32 there: cosine per row, not bit equality)."""
        enc = A.compile_encoder(str(self.whole), self.core, "CPU")
        got, nx, ny = enc.encode(str(self.png), (0.5,) * 3, (0.5,) * 3)
        want = np.asarray(self.fixed(self.pixels().transpose(2, 0, 1)[None])[self.fixed.output(0)], np.float32)[0]
        self.assertEqual((nx, ny), (4, 4))
        self.assertGreater(cos_rows(got, want).min(), 0.999)

    def test_padding_and_chunks_change_nothing(self):
        """The same picture in the 16-token graph, padded into the 32-token graph, and with chunked attention."""
        side = A.load_side(str(self.whole))
        inp4 = A.Inputs(self.pixels().transpose(2, 0, 1).copy(), side, 16)
        inp8 = A.Inputs(self.pixels().transpose(2, 0, 1).copy(), side, 32)
        whole = A.compile_encoder(str(self.whole), self.core, "CPU")
        chunked = A.compile_encoder(str(self.chunked), self.core, "CPU")
        base = whole.run(inp4, 16)
        self.assertTrue(np.allclose(whole.run(inp8, 32), base, atol=1e-4))
        self.assertTrue(np.allclose(chunked.run(inp4, 16), base, atol=1e-4))
        self.assertTrue(np.allclose(chunked.run(inp8, 32), base, atol=1e-4))

    def test_reader_protocol(self):
        """strata_vision_npu.py --anysize on the CPU device: READY with the export's width, a wide picture keeps
        its shape (150x40, raised to the 8-token floor: 6 x 2), the SVE1 record matches, and too many tokens is
        one ERR."""
        out = self.dir / "wide.sve"
        p = subprocess.run([sys.executable, str(READER), "--anysize", str(self.whole), "--device", "CPU",
                            "--max-tokens", "32"], input=f"ENC {self.wide} {out}\nQUIT\n", capture_output=True,
                           text=True, timeout=300)
        lines = p.stdout.splitlines()
        self.assertEqual(lines[0], "READY 3", p.stderr)
        tw, th, _ = A.plan(150, 40, {"budgets": [16, 32]}, 32, 0)
        self.assertEqual((tw, th), (192, 64))
        nx, ny = tw // 32, th // 32
        self.assertTrue(lines[1].startswith(f"OK {nx * ny} {nx} {ny} "), lines[1])
        data = out.read_bytes()
        magic, n, gx, gy, d = struct.unpack("<5i", data[:20])
        self.assertEqual((magic, n, gx, gy, d), (0x31455653, nx * ny, nx, ny, 3))
        self.assertEqual(len(data), 20 + n * d * 4)
        p = subprocess.run([sys.executable, str(READER), "--anysize", str(self.whole), "--device", "CPU",
                            "--max-tokens", "64"], input="QUIT\n", capture_output=True, text=True, timeout=300)
        self.assertEqual(len(p.stdout.splitlines()), 1)
        self.assertTrue(p.stdout.startswith("ERR ") and "largest budget 32" in p.stdout, p.stdout)
        self.assertNotEqual(p.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
