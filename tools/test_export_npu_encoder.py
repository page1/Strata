"""tools/test_export_npu_encoder.py - the GGUF -> OpenVINO IR exporter's math, tested without a GPU,
an NPU, OpenVINO or any download.

Checks tools/vision/export_npu_encoder.py's pieces that decide whether the exported encoder is the same
function llama.cpp's mtmd builds: the merged token order (the layout chain against the index formula, and
both against hand numbers), the m-rope cos/sin tables, the folded patch-embed weight and the position/bias
addend, the transposed layer weights, and the meta checks that must refuse projectors this tool does not
build.  A tiny synthetic mmproj GGUF (patch 16, merge 2, image 32, one block) is written into a temp dir,
so no model file is needed.

    python -m unittest tools.test_export_npu_encoder        (or: python tools/test_export_npu_encoder.py)
"""

from __future__ import annotations

import importlib.util
import io
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "export_npu_encoder", ROOT / "tools" / "vision" / "export_npu_encoder.py"
)
ex = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(
    ex
)  # the exporter imports no openvino at top level, on purpose


def write_gguf(path: Path, cfg: dict, tensors: dict):
    """A minimal GGUF v3: metadata, the tensor directory, 32-byte padding, f32 tensor data.  Shapes are
    given torch-like and reversed into ggml order (fastest dim first), the way a real mmproj stores them."""
    kv = {
        "general.architecture": ("s", cfg["architecture"]),
        "clip.projector_type": ("s", cfg["projector"]),
        "clip.vision.patch_size": ("I", cfg["patch"]),
        "clip.vision.spatial_merge_size": ("I", cfg["merge"]),
        "clip.vision.image_size": ("I", cfg["side"]),
        "clip.vision.embedding_length": ("I", cfg["d"]),
        "clip.vision.attention.head_count": ("I", cfg["heads"]),
        "clip.vision.attention.layer_norm_epsilon": ("f", 1e-6),
        "clip.vision.block_count": ("I", cfg["layers"]),
        "clip.vision.projection_dim": ("I", cfg["proj"]),
        "clip.vision.ffn_dim": ("I", cfg["ffn"]),
        "clip.vision.image_mean": ("af", [0.5, 0.5, 0.5]),
        "clip.vision.image_std": ("af", [0.5, 0.5, 0.5]),
    }
    kv.update(cfg.get("extra_kv", {}))
    head = io.BytesIO()

    def ws(v: str):
        b = v.encode()
        head.write(struct.pack("<Q", len(b)) + b)

    def wv(tag: str, v):  # the GGUF type id first, the value after, as the loader reads
        if tag == "s":  # 8 = STRING
            head.write(struct.pack("<I", 8))
            ws(v)
        elif tag == "I":  # 4 = UINT32
            head.write(struct.pack("<II", 4, v))
        elif tag == "f":  # 6 = FLOAT32
            head.write(struct.pack("<If", 6, v))
        elif tag == "ab":  # 9 = ARRAY of 7 = BOOL
            head.write(struct.pack("<IIQ", 9, 7, len(v)))
            head.write(b"".join(struct.pack("<?", x) for x in v))
        elif tag == "af":  # 9 = ARRAY of 6 = FLOAT32
            head.write(struct.pack("<IIQ", 9, 6, len(v)))
            head.write(struct.pack(f"<{len(v)}f", *v))
        else:
            raise AssertionError(tag)

    body = b""
    infos = []
    for name, arr in tensors.items():
        infos.append((name, list(reversed(np.asarray(arr).shape)), len(body)))
        body += np.ascontiguousarray(arr, np.float32).tobytes()
    head.write(b"GGUF" + struct.pack("<I", 3))
    head.write(struct.pack("<QQ", len(tensors), len(kv)))
    for k, (tag, v) in kv.items():
        ws(k)
        wv(tag, v)
    for name, dims, off in infos:
        ws(name)
        head.write(struct.pack("<I", len(dims)))
        for d in dims:
            head.write(struct.pack("<Q", d))
        head.write(struct.pack("<I", 0))  # f32
        head.write(struct.pack("<Q", off))
    data = head.getvalue()
    with open(path, "wb") as fh:
        fh.write(data + b"\x00" * (-len(data) % 32) + body)


class Tiny:
    """One synthetic mmproj in memory: the tool's contract at a scale a test can run instantly."""

    def __init__(self, **over):
        self.cfg = dict(
            architecture="clip",
            projector="qwen3vl_merger",
            patch=16,
            merge=2,
            side=32,
            d=8,
            heads=1,
            layers=1,
            ffn=5,
            proj=3,
        )
        self.cfg.update(over)
        c = self.cfg
        g, d = c["side"] // c["patch"], c["d"]
        rng = np.random.default_rng(7)
        t = {
            "v.patch_embd.weight": rng.standard_normal((d, 3, 16, 16)),
            "v.patch_embd.weight.1": rng.standard_normal((d, 3, 16, 16)),
            "v.patch_embd.bias": np.arange(d, dtype=np.float32) * 0.01,
            "v.position_embd.weight": rng.standard_normal((g * g, d)),
            "v.post_ln.weight": rng.standard_normal(d),
            "v.post_ln.bias": rng.standard_normal(d),
            "mm.0.weight": rng.standard_normal(
                (4 * d, 4 * d)
            ),  # (out, in) like the real file
            "mm.0.bias": rng.standard_normal(4 * d),
            "mm.2.weight": rng.standard_normal((c["proj"], 4 * d)),
            "mm.2.bias": rng.standard_normal(c["proj"]),
        }
        for i in range(c["layers"]):
            L = f"v.blk.{i}."
            t[L + "ln1.weight"] = rng.standard_normal(d)
            t[L + "ln1.bias"] = rng.standard_normal(d)
            t[L + "attn_qkv.weight"] = rng.standard_normal((3 * d, d))
            t[L + "attn_qkv.bias"] = rng.standard_normal(3 * d)
            t[L + "attn_out.weight"] = rng.standard_normal((d, d))
            t[L + "attn_out.bias"] = rng.standard_normal(d)
            t[L + "ln2.weight"] = rng.standard_normal(d)
            t[L + "ln2.bias"] = rng.standard_normal(d)
            t[L + "ffn_up.weight"] = rng.standard_normal((c["ffn"], d))
            t[L + "ffn_up.bias"] = rng.standard_normal(c["ffn"])
            t[L + "ffn_down.weight"] = rng.standard_normal((d, c["ffn"]))
            t[L + "ffn_down.bias"] = rng.standard_normal(d)
        self.tensors = t

    def file(self, dirpath: Path) -> Path:
        path = dirpath / "mmproj-tiny.gguf"
        write_gguf(path, self.cfg, self.tensors)
        return path


def build(tiny: Tiny, dirpath: Path):
    meta, T = ex.load_gguf(str(tiny.file(dirpath)))
    ex.check_meta(meta)
    return meta, T, ex.build_constants(meta, T)


class TestMath(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="strata-export-test-"))
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_layout_chain_equals_the_index(self):
        """The reshape+transpose the graph uses for the merged order equals a Gather with C_idx - at the
        real 48x48 grid, on patch positions alone (no weights, no OpenVINO)."""
        g, mg, npt = 48, 24, 2304
        idx = ex.C_idx({"g": g, "mg": mg, "npt": npt})
        self.assertEqual(sorted(idx.tolist()), list(range(npt)))  # a true permutation
        raster = np.arange(
            npt, dtype=np.int64
        )  # value = (y, x) raster index of each patch
        chain = (
            raster.reshape(g, g)
            .reshape(mg, 2, mg, 2)
            .transpose(0, 2, 1, 3)
            .reshape(npt)
        )
        self.assertTrue((chain == raster[idx]).all())

    def test_index_hand_numbers(self):
        """Merged token 0 holds the top-left 2x2 block (patches 0,1,48,49 at g=48); token 1 the next two
        columns; the last token the bottom-right block."""
        idx = ex.C_idx({"g": 48, "mg": 24, "npt": 2304})
        self.assertEqual(idx[:4].tolist(), [0, 1, 48, 49])  # TL, TR, BL, BR
        self.assertEqual(idx[4:8].tolist(), [2, 3, 50, 51])
        self.assertEqual(idx[8:12].tolist(), [4, 5, 52, 53])  # ... and token 23 ends row 0
        self.assertEqual(
            idx[96:100].tolist(), [96, 97, 144, 145]
        )  # token 24: block row 1, column 0
        self.assertEqual(idx[-4:].tolist(), [2254, 2255, 2302, 2303])

    def test_rope_tables(self):
        """theta_scale is base^(-2/n_dims) with n_dims = dh/2 - the exponent the ggml rope kernel uses -
        rows (y) rotate the first half of each head's pairs, columns (x) the second half, and the x half
        restarts its frequencies at 1."""
        _, _, C = build(Tiny(), self.dir)
        dh, n2 = (
            C["dh"],
            C["dh"] // 2,
        )  # the tiny file: dh 8, pairs 4, two sections of 2
        self.assertEqual((dh, n2), (8, 4))
        theta = 10000.0 ** (-2.0 * np.arange(n2 // 2) / n2)
        self.assertAlmostEqual(float(theta[0]), 1.0)
        self.assertAlmostEqual(float(theta[1]), 0.01)  # 10000^(-2/4) at this tiny dh
        g, mg = C["g"], C["mg"]
        p = np.arange(C["npt"])
        x = 2 * ((p // 4) % mg) + p % 2
        y = 2 * (p // (mg * 4)) + (p // 2) % 2
        ang = np.concatenate(
            [y[:, None] * theta[None, :], x[:, None] * theta[None, :]], 1
        )
        self.assertTrue(
            np.array_equal(C["cos"][:, 0, :], np.cos(ang.astype(np.float32)))
        )
        self.assertTrue(
            np.array_equal(C["sin"][:, 0, :], np.sin(ang.astype(np.float32)))
        )

    def test_folded_patch_embed_and_position_addend(self):
        """conv(x, w0) + conv(x, w1) folds into one weight laid out like the flattened patch (channel,
        row, col with col fastest), and every token's added constant is position[its patch] + patch bias."""
        _, T, C = build(Tiny(), self.dir)
        wsum = T["v.patch_embd.weight"][0] + T["v.patch_embd.weight.1"][0]
        want = np.ascontiguousarray(wsum.transpose(1, 2, 3, 0).reshape(16 * 16 * 3, 8))
        self.assertTrue(np.array_equal(C["wsum"], want))
        pos = T["v.position_embd.weight"][0]
        want = pos[ex.C_idx(C)] + T["v.patch_embd.bias"][0][None, :]
        self.assertTrue(np.array_equal(C["biaspos"], want))

    def test_layer_weights_are_the_file_transposed(self):
        """The IR multiplies x @ W, the file stores W transposed (and the merger too)."""
        _, T, C = build(Tiny(), self.dir)
        L = C["layers"][0]
        self.assertTrue(np.array_equal(L["qkvw"], T["v.blk.0.attn_qkv.weight"][0].T))
        self.assertTrue(np.array_equal(L["ow"], T["v.blk.0.attn_out.weight"][0].T))
        self.assertTrue(np.array_equal(L["upw"], T["v.blk.0.ffn_up.weight"][0].T))
        self.assertTrue(np.array_equal(L["dnw"], T["v.blk.0.ffn_down.weight"][0].T))
        self.assertTrue(np.array_equal(C["mm0w"], T["mm.0.weight"][0].T))
        self.assertTrue(np.array_equal(C["mm2w"], T["mm.2.weight"][0].T))


class TestRefusals(unittest.TestCase):
    """An exporter that quietly built a wrong graph would be worse than none: every file outside its one
    contract must end in a plain SystemExit line, not in a plausible-looking encoder."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="strata-export-refuse-"))
        self.addCleanup(shutil.rmtree, self.dir, True)

    def refuse(self, tiny: Tiny, needle: str):
        with self.assertRaises(SystemExit) as cm:
            meta, T = ex.load_gguf(str(tiny.file(self.dir)))
            ex.check_meta(meta)
            ex.build_constants(meta, T)
        self.assertIn(needle, str(cm.exception))

    def test_wrong_architecture(self):
        self.refuse(Tiny(architecture="llama"), "qwen3vl_merger")

    def test_wrong_projector(self):
        self.refuse(Tiny(projector="mlp"), "qwen3vl_merger")

    def test_deepstack(self):
        self.refuse(
            Tiny(extra_kv={"clip.vision.deepstack_layers": ("ab", [True, False])}),
            "deepstack",
        )

    def test_other_patch_size(self):
        self.refuse(Tiny(patch=14), "patch 16")

    def test_other_merge(self):
        self.refuse(Tiny(merge=4), "merge 2")

    def test_position_table_of_the_wrong_size(self):
        t = Tiny()
        t.tensors["v.position_embd.weight"] = np.zeros(
            (3, 8), np.float32
        )  # needs 4 rows
        self.refuse(t, "position table")


if __name__ == "__main__":
    unittest.main(verbosity=2)
