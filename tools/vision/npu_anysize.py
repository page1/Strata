#!/usr/bin/env python3
"""tools/vision/npu_anysize.py - the Qwen3.8 picture encoder on the Intel NPU at any picture size.

export_npu_encoder.py bakes one picture size (768x768) into the graph: the patch order, the learned positions and the
rope tables are constants of that one grid, so every other picture is squeezed into it (a 2000x920 screenshot of
small text loses most of its letters).  Here the graph knows only a patch COUNT - a token budget - and the reader
hands it everything that depends on the picture's shape:

    in  "patches"   float32 [N, 768]     16x16x3 patches, (c, ky, kx) flattened, in merged order
                                         (each 2x2 merge block as four adjacent rows TL, TR, BL, BR)
    in  "pos"       float32 [N, 1152]    learned positions resized to the grid (bilinear, align corners, as
                                         llama.cpp's mtmd) + the patch-embed bias
    in  "cos","sin" float32 [N, 1, 36]   M-RoPE tables of each patch's (y, x)
    in  "mask"      float32 [1, 1, N]    0 for real patches, -30000 for padding

A picture is sized the way mtmd sizes it for these readers (aspect kept, sides multiples of 32, at most --max-tokens,
letterboxed on black), and its patches are padded with masked rows up to the smallest exported budget that holds
them: a few compiled graphs cover every shape, and the masked rows change nothing in the real ones (attention never
looks at them; every other op is per row).  The padded rows are dropped from the answer.

Two things make it compile on the NPU (measured on an Intel AI Boost, NPU 3720, Windows 11 driver 32.0.100.5540,
OpenVINO 2026.4.1, 2026-10-27):

  - attention a chunk of query rows at a time (--q-chunk, default 512).  One block's attention as a single
    4096 x 4096 score matrix per head did not compile in 10 minutes, and 8192 x 8192 not in 9 hours (85 GB of RAM);
    the same block in chunks of 1024 x 4096 compiled in 9.6 s, 512 x 8192 in 29 s.  The answers are the same
    numbers: softmax is per query row.
  - one graph for every block (the 27 blocks have one shape): the block's weights are fp16 inputs, kept in
    blocks.npz, and the reader runs that graph 27 times.  A 2048-token budget compiles in about 47 s and 2.6 GB of RAM
    (27 baked one-block graphs: 18 min, 17.6 GB), at about 7% of the encode time.

The files of an export (--out-dir):  anysize.json (budgets, graphs, sizes, normalization), pos.npy, bias.npy,
blocks.npz, and per budget enc-<budget>-head.xml (patch embedding), enc-<budget>-block.xml (one block, weights as
inputs) and enc-<budget>-tail.xml (post-LayerNorm and merger), each with its .bin.

    python tools/vision/npu_anysize.py export <mmproj.gguf> --out-dir DIR [--budgets 576,1024,1536,2048]
                                              [--q-chunk 512] [--ln max|plain]

The reader is strata_vision_npu.py with --anysize DIR (or STRATA_VISION_NPU_ANYSIZE); see docs/INTEL_NPU.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PATCH, MERGE, ALIGN = 16, 2, 32
BLOCK_WEIGHTS = ("ln1w", "ln1b", "qkvw", "qkvb", "ow", "ob", "ln2w", "ln2b", "upw", "upb", "dnw", "dnb")
MASK_NEG = -30000.0
PARTS = ("head", "block", "tail")


# ---------------------------------------------------------------- what the reader computes per picture

def target_size(w: int, h: int, max_tokens: int, min_tokens: int = 8) -> tuple[int, int]:
    """mtmd's calc_size_preserved_ratio for the Qwen-VL readers: sides aligned to 32, area within the token bounds."""
    max_px, min_px = max_tokens * ALIGN * ALIGN, min_tokens * ALIGN * ALIGN
    # std::round (halves away from zero; the sides are positive)
    w_bar = max(ALIGN, int(math.floor(w / ALIGN + 0.5)) * ALIGN)
    h_bar = max(ALIGN, int(math.floor(h / ALIGN + 0.5)) * ALIGN)
    if w_bar * h_bar > max_px:
        beta = math.sqrt(h * w / max_px)
        h_bar = max(ALIGN, int(math.floor(h / beta / ALIGN)) * ALIGN)
        w_bar = max(ALIGN, int(math.floor(w / beta / ALIGN)) * ALIGN)
    elif w_bar * h_bar < min_px:
        beta = math.sqrt(min_px / (h * w))
        h_bar = int(math.ceil(h * beta / ALIGN)) * ALIGN
        w_bar = int(math.ceil(w * beta / ALIGN)) * ALIGN
    return w_bar, h_bar


def fit(im, w: int, h: int):
    """mtmd's resize for these readers (PAD_CEIL, the default): scale by the smaller ratio with Pillow's bicubic,
    sides rounded up, then centre on a black w x h canvas.  `im` is an RGB PIL image."""
    from PIL import Image
    if im.size == (w, h):
        return im
    s = min(w / im.width, h / im.height)
    nw, nh = min(int(math.ceil(im.width * s)), w), min(int(math.ceil(im.height * s)), h)
    out = Image.new("RGB", (w, h), (0, 0, 0))
    out.paste(im.resize((nw, nh), Image.Resampling.BICUBIC), ((w - nw) // 2, (h - nh) // 2))
    return out


def merged_xy(gx: int, gy: int) -> tuple[np.ndarray, np.ndarray]:
    """Patch (x, y) of each row in merged order: blocks row-major, sub-patches TL, TR, BL, BR."""
    mgx = gx // 2
    p = np.arange(gx * gy)
    blk, sub = p // 4, p % 4
    return 2 * (blk % mgx) + sub % 2, 2 * (blk // mgx) + sub // 2


def interp_pos(table: np.ndarray, gx: int, gy: int) -> np.ndarray:
    """The learned [n*n, d] position table resized to gx x gy, bilinear with align corners (ggml_interpolate with
    GGML_SCALE_FLAG_ALIGN_CORNERS), raster rows."""
    n = int(round(math.sqrt(table.shape[0])))
    t = table.reshape(n, n, -1)
    if gx == n and gy == n:
        return table

    def axis(out):
        src = np.zeros(1) if out == 1 else np.arange(out) * (n - 1) / (out - 1)
        i0 = np.clip(np.floor(src).astype(int), 0, n - 1)
        i1 = np.clip(i0 + 1, 0, n - 1)
        return i0, i1, (src - i0).astype(np.float32)

    y0, y1, fy = axis(gy)
    x0, x1, fx = axis(gx)
    top = t[y0][:, x0] * (1 - fx)[None, :, None] + t[y0][:, x1] * fx[None, :, None]
    bot = t[y1][:, x0] * (1 - fx)[None, :, None] + t[y1][:, x1] * fx[None, :, None]
    out = top * (1 - fy)[:, None, None] + bot * fy[:, None, None]
    return out.reshape(gx * gy, -1).astype(np.float32)


def rope_tables(x: np.ndarray, y: np.ndarray, dh: int) -> tuple[np.ndarray, np.ndarray]:
    """M-RoPE cos/sin of each patch: rows (y) rotate the first half of each head's pairs, columns (x) the second,
    with theta = 10000^(-2j / (dh/2)) - the tables export_npu_encoder.py bakes for its one grid."""
    n2 = dh // 2
    theta = 10000.0 ** (-2.0 * np.arange(n2 // 2) / n2)
    ang = np.concatenate([y[:, None] * theta[None, :], x[:, None] * theta[None, :]], axis=1).astype(np.float32)
    return np.cos(ang)[:, None, :].astype(np.float32), np.sin(ang)[:, None, :].astype(np.float32)


def patchify(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """[3, H, W] normalized pixels -> [N, 768] rows in merged order, each patch (c, ky, kx) flattened."""
    c, h, w = img.shape
    g = img.reshape(c, h // PATCH, PATCH, w // PATCH, PATCH).transpose(1, 3, 0, 2, 4)  # gy, gx, c, ky, kx
    return np.ascontiguousarray(g[y, x].reshape(len(x), c * PATCH * PATCH), np.float32)


class Inputs:
    """Everything one picture hands the graphs, padded to `budget` tokens."""

    def __init__(self, pixels: np.ndarray, side_info: dict, budget_tokens: int):
        _, h, w = pixels.shape
        gx, gy = w // PATCH, h // PATCH
        x, y = merged_xy(gx, gy)
        n, npad = gx * gy, budget_tokens * 4
        if n > npad:
            raise ValueError(f"{n // 4} tokens do not fit the {budget_tokens}-token graph")
        self.nx, self.ny, self.n_tokens = gx // 2, gy // 2, n // 4
        pos = interp_pos(side_info["pos"], gx, gy)[y * gx + x] + side_info["bias"][None, :]
        cos, sin = rope_tables(x, y, side_info["dh"])
        pad = npad - n

        def padded(a, fill=0.0):
            if not pad:
                return np.ascontiguousarray(a)
            return np.ascontiguousarray(np.concatenate([a, np.full((pad,) + a.shape[1:], fill, np.float32)]))

        self.feed = {
            "patches": padded(patchify(pixels, x, y)),
            "pos": padded(pos),
            "cos": padded(cos, 1.0),
            "sin": padded(sin),
            "mask": np.concatenate([np.zeros(n, np.float32), np.full(pad, MASK_NEG, np.float32)])[None, None, :],
        }


def load_side(out_dir: str) -> dict:
    """An export's side files: anysize.json, the position table, the patch bias and the blocks' weights."""
    with open(os.path.join(out_dir, "anysize.json"), encoding="utf-8") as f:
        info = json.load(f)
    info["pos"] = np.load(os.path.join(out_dir, "pos.npy"))
    info["bias"] = np.load(os.path.join(out_dir, "bias.npy"))
    with np.load(os.path.join(out_dir, "blocks.npz")) as z:
        info["blocks"] = [{k: np.ascontiguousarray(z[f"L{i}_{k}"]) for k in BLOCK_WEIGHTS}
                          for i in range(info["n_layers"])]
    return info


def plan(w: int, h: int, side_info: dict, max_tokens: int, min_tokens: int) -> tuple[int, int, int]:
    """(target w, target h, budget) for a w x h picture: mtmd's size within max_tokens, then the smallest budget
    that holds it."""
    tw, th = target_size(w, h, max_tokens, min_tokens or 8)
    n = (tw // ALIGN) * (th // ALIGN)
    fits = [b for b in side_info["budgets"] if b >= n]
    if not fits:
        raise ValueError(f"{n} tokens are above the export's largest budget {max(side_info['budgets'])}")
    return tw, th, min(fits)


class AnySizeEncoder:
    """The export's graphs, compiled once per budget, and the per-picture run.

    `graphs[budget]` is (head, block, tail): callables that take a feed dict and answer one float32 array.  The
    OpenVINO wrapper in compile() builds them; tests may hand in numpy stand-ins."""

    def __init__(self, side_info: dict, graphs: dict, max_tokens: int = 0, min_tokens: int = 0):
        self.side = side_info
        self.graphs = graphs
        top = max(side_info["budgets"])
        if max_tokens and max_tokens > top:
            raise ValueError(f"--max-tokens {max_tokens} is above the export's largest budget {top}: lower it, "
                             f"or export a budget of {max_tokens} (npu_anysize.py export --budgets ...)")
        self.max_tokens = max_tokens or top
        self.min_tokens = min_tokens

    @property
    def n_embd(self) -> int:
        return int(self.side["n_embd"])

    def run(self, inp: Inputs, budget: int) -> np.ndarray:
        head, block, tail = self.graphs[budget]
        x = head({k: inp.feed[k] for k in ("patches", "pos", "cos", "sin", "mask")})
        rest = {k: inp.feed[k] for k in ("cos", "sin", "mask")}
        for weights in self.side["blocks"]:
            x = block({"hidden": x, **rest, **{"w_" + k: v for k, v in weights.items()}})
        x = tail({"hidden": x, **rest})
        return np.ascontiguousarray(x[: inp.n_tokens], np.float32)

    def encode(self, path: str, mean, std) -> tuple[np.ndarray, int, int]:
        """One picture file -> (embeddings [n_tokens, n_embd], nx, ny)."""
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            tw, th, budget = plan(im.width, im.height, self.side, self.max_tokens, self.min_tokens)
            im = fit(im, tw, th)
            px = (np.asarray(im, np.float32) / 255.0 - np.array(mean, np.float32)) / np.array(std, np.float32)
        inp = Inputs(np.ascontiguousarray(px.transpose(2, 0, 1)), self.side, budget)
        return self.run(inp, budget), inp.nx, inp.ny

    def warm(self) -> None:
        """One zero picture through the smallest budget, so the first real one does not pay for buffer setup."""
        b = min(self.side["budgets"])
        side = int(math.isqrt(b)) * ALIGN
        self.run(Inputs(np.zeros((3, side, side), np.float32), self.side, b), b)


def compile_encoder(out_dir: str, core, device: str, max_tokens: int = 0, min_tokens: int = 0) -> AnySizeEncoder:
    """Read an export and compile every budget's three graphs on `device` with OpenVINO (`core` carries the
    CACHE_DIR, if any).  Never set NPU_TILES=1 for these graphs: on the 2026-10 Windows driver it compiled and then
    hung the NPU at the first encode (ZE_RESULT_ERROR_DEVICE_LOST)."""
    side = load_side(out_dir)

    def wrap(path):
        comp = core.compile_model(core.read_model(path), device)
        req = comp.create_infer_request()
        names = [p.get_any_name() for p in comp.inputs]
        out = comp.output(0)

        def call(feed):
            return np.array(req.infer({n: feed[n] for n in names})[out], np.float32)
        return call

    graphs = {b: tuple(wrap(os.path.join(out_dir, f)) for f in side["graphs"][str(b)]) for b in side["budgets"]}
    return AnySizeEncoder(side, graphs, max_tokens, min_tokens)


# ---------------------------------------------------------------- the graphs

def build_model(ops, C, npad: int, part: str, ln: str = "max", q_chunk: int = 512):
    """One of the three graphs over npad patch rows: "head" (patches + positions -> hidden), "block" (one encoder
    block, its weights the inputs w_<name> in fp16) or "tail" (post-LayerNorm and merger -> embeddings).
    ln="max" divides each LayerNorm's input by its row's max |x| first (export_npu_encoder.py's --ln-scale max, for
    NPU builds that square in fp16); ln="plain" is the textbook LayerNorm."""
    import openvino as ov

    def con(a):
        if not isinstance(a, (np.ndarray, np.generic, float, int)):
            return a  # already a graph node (a weight input)
        return ops.constant(np.ascontiguousarray(np.asarray(a, np.float32)))

    ax = ops.constant(np.array([-1], np.int64))
    eps_c = np.float32(1e-6) if ln == "max" else C["eps"]

    def lnorm(x, wt, bs):
        if ln == "max":
            m = ops.add(ops.reduce_max(ops.abs(x), ax, keep_dims=True), con(np.float32(1e-6)))
            x = ops.divide(x, m)
        mu = ops.reduce_mean(x, ax, keep_dims=True)
        xc = ops.subtract(x, mu)
        var = ops.reduce_mean(ops.multiply(xc, xc), ax, keep_dims=True)
        return ops.add(ops.multiply(ops.divide(xc, ops.sqrt(ops.add(var, con(eps_c)))), con(wt)), con(bs))

    half, one = con(np.float32(0.5)), con(np.float32(1.0))
    gc, ga = con(np.float32(0.7978845608028654)), con(np.float32(0.044715))

    def gelu(x):
        inner = ops.multiply(gc, ops.add(x, ops.multiply(ga, ops.multiply(ops.multiply(x, x), x))))
        return ops.multiply(ops.multiply(x, half), ops.add(one, ops.tanh(inner)))

    d, heads, dh = C["d"], C["heads"], C["dh"]
    if part == "head":
        patches = ops.parameter([npad, PATCH * PATCH * 3], np.float32, name="patches")
        pos = ops.parameter([npad, d], np.float32, name="pos")
        extra = [ops.parameter(s, np.float32, name=n) for n, s in
                 (("cos", [npad, 1, dh // 2]), ("sin", [npad, 1, dh // 2]), ("mask", [1, 1, npad]))]
        tok = ops.add(ops.matmul(patches, con(C["wsum"]), False, False), pos)
        # cos/sin/mask are unused here; they are inputs so all three graphs take the same per-picture tables
        return ov.Model([tok], [patches, pos] + extra, "qwen3vl_anysize_head")

    tok = ops.parameter([npad, d], np.float32, name="hidden")
    cos = ops.parameter([npad, 1, dh // 2], np.float32, name="cos")
    sin = ops.parameter([npad, 1, dh // 2], np.float32, name="sin")
    mask = ops.parameter([1, 1, npad], np.float32, name="mask")
    params = [tok, cos, sin, mask]

    if part == "tail":
        m = lnorm(tok, C["postw"], C["postb"])
        m = ops.reshape(m, ops.constant(np.array([npad // 4, d * 4], np.int64)), False)
        z = gelu(ops.add(ops.matmul(m, con(C["mm0w"]), False, False), con(C["mm0b"])))
        z = ops.add(ops.matmul(z, con(C["mm2w"]), False, False), con(C["mm2b"]))
        return ov.Model([z], params, "qwen3vl_anysize_tail")

    assert part == "block", part
    wp = {k: ops.parameter(list(np.shape(C["layers"][0][k])), np.float16, name="w_" + k) for k in BLOCK_WEIGHTS}
    params += list(wp.values())
    L = {k: ops.convert(p, np.float32) for k, p in wp.items()}

    def rope(qk):
        sp = ops.split(qk, 2, 2)
        a, b = sp.output(0), sp.output(1)
        return ops.concat([ops.subtract(ops.multiply(a, cos), ops.multiply(b, sin)),
                           ops.add(ops.multiply(a, sin), ops.multiply(b, cos))], 2)

    t_hbt = ops.constant(np.array([1, 0, 2], np.int64))
    shp = ops.constant(np.array([npad, heads, dh], np.int64))
    h = lnorm(tok, L["ln1w"], L["ln1b"])
    qkv = ops.add(ops.matmul(h, con(L["qkvw"]), False, False), con(L["qkvb"]))
    sp = ops.split(qkv, 1, 3)
    q, k, v = (ops.reshape(sp.output(i), shp, False) for i in range(3))
    q, k = rope(q), rope(k)
    q, k, v = (ops.transpose(t, t_hbt) for t in (q, k, v))  # (heads, npad, dh)
    # the same attention, a chunk of query rows at a time: each score matrix is at most q_chunk x npad
    step = q_chunk if q_chunk and q_chunk < npad else npad
    lengths = [step] * (npad // step) + ([npad % step] if npad % step else [])
    qs = ops.variadic_split(q, ops.constant(np.int64(1)), ops.constant(np.array(lengths, np.int64)))
    outs = []
    for i in range(len(lengths)):
        sc = ops.add(ops.multiply(ops.matmul(qs.output(i), k, False, True), con(C["kq"])), mask)
        outs.append(ops.matmul(ops.softmax(sc, 2), v, False, False))
    o = ops.concat(outs, 1) if len(outs) > 1 else outs[0]
    o = ops.reshape(ops.transpose(o, t_hbt), ops.constant(np.array([npad, d], np.int64)), False)
    tok = ops.add(tok, ops.add(ops.matmul(o, con(L["ow"]), False, False), con(L["ob"])))
    h = lnorm(tok, L["ln2w"], L["ln2b"])
    f = gelu(ops.add(ops.matmul(h, con(L["upw"]), False, False), con(L["upb"])))
    tok = ops.add(tok, ops.add(ops.matmul(f, con(L["dnw"]), False, False), con(L["dnb"])))
    return ov.Model([tok], params, "qwen3vl_anysize_block")


def export(mmproj: str, out_dir: str, budgets: list[int], ln: str = "max", q_chunk: int = 512) -> dict:
    """Write an export (see the module docstring) and answer its anysize.json."""
    import openvino as ov
    from openvino import opset13 as ops

    import export_npu_encoder as E

    meta, T = E.load_gguf(mmproj)
    E.check_meta(meta)
    C = E.build_constants(meta, T)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "pos.npy"), np.ascontiguousarray(T["v.position_embd.weight"][0], np.float32))
    np.save(os.path.join(out_dir, "bias.npy"), np.ascontiguousarray(T["v.patch_embd.bias"][0], np.float32))
    np.savez(os.path.join(out_dir, "blocks.npz"), **{f"L{i}_{k}": np.asarray(L[k], np.float16)
                                                     for i, L in enumerate(C["layers"]) for k in BLOCK_WEIGHTS})
    graphs = {}
    for b in sorted(budgets):
        files = []
        for part in PARTS:
            path = os.path.join(out_dir, f"enc-{b}-{part}.xml")
            ov.save_model(build_model(ops, C, b * 4, part, ln, q_chunk), path, compress_to_fp16=True)
            files.append(os.path.basename(path))
        graphs[str(b)] = files
    info = {"format": 1, "budgets": sorted(budgets), "graphs": graphs, "n_layers": len(C["layers"]),
            "dh": C["dh"], "n_embd": C["proj"], "ln": ln, "q_chunk": q_chunk,
            "mean": [float(x) for x in C["mean"]], "std": [float(x) for x in C["std"]]}
    with open(os.path.join(out_dir, "anysize.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="write an any-size export of a Qwen3.8 mmproj GGUF")
    e.add_argument("mmproj", help="the mmproj GGUF (clip + qwen3vl_merger)")
    e.add_argument("--out-dir", required=True)
    e.add_argument("--budgets", default="576,1024,1536,2048",
                   help="token budgets to compile, comma-separated (a picture uses the smallest that holds it)")
    e.add_argument("--q-chunk", type=int, default=512,
                   help="attention in chunks of this many query rows (0: one chunk); 512 compiles a 2048 budget")
    e.add_argument("--ln", choices=("max", "plain"), default="max")
    a = ap.parse_args()
    budgets = sorted({int(b) for b in a.budgets.split(",") if b.strip()})
    if not budgets or budgets[0] < 1:
        ap.error("--budgets needs positive numbers")
    info = export(a.mmproj, a.out_dir, budgets, a.ln, a.q_chunk)
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
