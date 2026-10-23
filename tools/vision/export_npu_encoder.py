#!/usr/bin/env python3
"""export_npu_encoder.py MMPROJ.XML --out OUT.XML : turn a Strata mmproj GGUF
(clip arch with the qwen3vl_merger projector, the shape used by
Qwen3.8-Flash-Next) into a static OpenVINO IR for the NPU picture reader.

The IR is the exact graph llama.cpp's mtmd builds for ONE still image at the
native resolution of the file (clip.vision.image_size, 768x768 today):
patch embed (both temporal halves summed), reordered so each 2x2 merge block
is four adjacent tokens, learned position embeddings, 27 blocks of
[M-RoPE attention, MLP with tanh GELU], post-LayerNorm, then the merger
(4608 -> 4608 GELU -> 2560).  Positions, the patch order and the position
table are baked in as constants at the native size, so the model takes one
tensor and no index inputs:

    in  "image"      float32 [1, 3, S, S]   pixels as (byte/255 - mean) / std,
                                               mean/std = clip.vision.image_mean
                                               and clip.vision.image_std from the
                                               GGUF (0.5/0.5 for Flash-Next)
    out "embeddings" float32 [1, S/32*S/32, projection_dim]   x-fastest rows,
                                               the same order the engine's own
                                               mtmd encoder writes into SVE1

Two Intel NPU driver bugs shaped the recommended export (Core Ultra 7 265K,
Intel AI Boost, intel_vpu driver, kernel 6.8.0-147, OpenVINO 2026.4.1,
measured 2026-10-23; details and all numbers in docs/INTEL_NPU.md):

  * a single compiled graph over ~850 MB of weights hangs the device at its
    first inference (ZE_RESULT_ERROR_DEVICE_LOST): 24 blocks / 805 MB runs in
    1.91 s, the whole 27-block / 896 MB graph hangs, while two graphs totalling
    1.17 GB in one process are fine - the ceiling is per compiled graph.
    --split-after 13 writes two chained IRs that compute the same function.
  * the NPU computes some LayerNorms with fp16 accumulators and this encoder's
    deep blocks run at std ~100: the sums of squares overflow and the LayerNorm
    quietly answers a constant (cosine against CPU: 0.999948 at std 0.8,
    0.958104 at std 53, collapsed at std 105, NaN at std 210).  --ln-scale max
    divides each LayerNorm's input by its row's max |x| - the same function,
    variance near 1 at every depth.

What runs correctly end to end on that NPU (0.998390 mean cosine against the
strata-vision engine reader, 2.16 s per 768x768 picture):

    export_npu_encoder.py <mmproj.gguf> --out encoder.xml --fp16 \
        --split-after 13 --ln-scale max
    reader:  --model-ir encoder-1.xml:encoder-2.xml   (os.pathsep; "; on Windows)

Run it with the NPU venv's python (it needs openvino and numpy):

    .venv-npu/bin/python tools/vision/export_npu_encoder.py \
        ~/projects/Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf \
        --out ~/models/npu-vision/encoder.xml

The math was checked against engine/strata-vision on a 768x768 picture:
per-row cosine 1.000000 for the numpy transcription of this same graph
(see docs/INTEL_NPU.md).  Video, deepstack features and other projectors are
out of scope: the script stops with a clear error on files it does not cover.
"""

import argparse
import os
import struct
import sys

import numpy as np

DT_SIZE = {0: 4, 1: 2, 30: 2}  # f32, f16, bf16 - all this file holds
META_REQ = {
    "general.architecture": "clip",
    "clip.projector_type": "qwen3vl_merger",
}


def load_gguf(path):
    """Return (meta, {name: (float32 ndarray, ggml dims)}); torch-like shapes
    (reversed dims, so the last axis is ggml dim 0, the file-fast axis)."""
    import mmap

    with open(path, "rb") as fh:
        if fh.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        fh.read(4)
        nt, nkv = struct.unpack("<QQ", fh.read(16))
        sc = {
            0: "<B",
            1: "<b",
            2: "<H",
            3: "<h",
            4: "<I",
            5: "<i",
            6: "<f",
            7: "<?",
            10: "<Q",
            11: "<q",
            12: "<d",
        }

        def rs():
            n = struct.unpack("<Q", fh.read(8))[0]
            return fh.read(n).decode()

        def rv(t):
            if t == 8:
                return rs()
            if t == 9:
                et = struct.unpack("<I", fh.read(4))[0]
                return [rv(et) for _ in range(struct.unpack("<Q", fh.read(8))[0])]
            return struct.unpack(sc[t], fh.read(struct.calcsize(sc[t])))[0]

        meta = {}
        for _ in range(nkv):
            k = rs()
            meta[k] = rv(struct.unpack("<I", fh.read(4))[0])
        infos = []
        for _ in range(nt):
            name = rs()
            nd = struct.unpack("<I", fh.read(4))[0]
            dims = [struct.unpack("<Q", fh.read(8))[0] for _ in range(nd)]
            dt = struct.unpack("<I", fh.read(4))[0]
            off = struct.unpack("<Q", fh.read(8))[0]
            infos.append((name, dims, dt, off))
        algn = meta.get("general.alignment", 32)
        start = (fh.tell() + algn - 1) // algn * algn
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    out = {}
    for name, dims, dt, off in infos:
        count = int(np.prod(dims)) if dims else 1
        if dt not in DT_SIZE:
            raise SystemExit(f"{path}: tensor {name} has unsupported dtype {dt}")
        raw = np.frombuffer(
            mm, dtype=np.uint8, count=count * DT_SIZE[dt], offset=start + off
        )
        if dt == 0:
            arr = np.array(raw.view("<f4"))
        elif dt == 1:
            arr = raw.view("<f2").astype(np.float32)
        else:  # bf16
            arr = (raw.view("<u2").astype(np.uint32) << 16).view("<f4")
        out[name] = (arr.reshape(tuple(reversed(dims))), dims)
    return meta, out


def check_meta(meta):
    for key, want in META_REQ.items():
        got = meta.get(key)
        if got != want:
            raise SystemExit(
                f"this tool converts clip + qwen3vl_merger mmprojs; "
                f"the file says {key} = {got!r}"
            )
    for key, val in meta.items():
        if "deepstack" in key and (
            val is True or (isinstance(val, list) and any(bool(v) for v in val))
        ):
            raise SystemExit(
                f"deepstack features are not supported yet ({key} is set in the file)"
            )


def build_constants(meta, T):
    """All the numbers of the static graph as numpy arrays."""
    patch = int(meta["clip.vision.patch_size"])
    merge = int(meta.get("clip.vision.spatial_merge_size", 2))
    side = int(meta["clip.vision.image_size"])
    d = int(meta["clip.vision.embedding_length"])
    heads = int(meta["clip.vision.attention.head_count"])
    nlayer = int(meta["clip.vision.block_count"])
    proj = int(meta["clip.vision.projection_dim"])
    eps = float(meta.get("clip.vision.attention.layer_norm_epsilon", 1e-6))
    mean = np.array(meta.get("clip.vision.image_mean", [0.5] * 3), np.float32)
    std = np.array(meta.get("clip.vision.image_std", [0.5] * 3), np.float32)
    if patch != 16 or merge != 2:
        raise SystemExit(
            "this tool covers patch 16 with merge 2 "
            f"(the file says patch {patch}, merge {merge})"
        )
    if side % (patch * merge):
        raise SystemExit("image_size must divide by patch*merge")
    g, mg = side // patch, side // patch // 2
    npt, ntok, dh = g * g, mg * mg, d // heads
    if dh % 4:
        raise SystemExit("head width must split into four m-rope sections")

    def w(name):
        if name not in T:
            raise SystemExit(f"{name} is missing from the file")
        return np.ascontiguousarray(T[name][0], np.float32)

    # --- patch embed: conv(x, w0) + conv(x, w1) as one gemm over patches ---
    # file shape (out, c, ky, kx); patch rows flatten (c, ky, kx) with kx fastest.
    wsum = w("v.patch_embd.weight") + w("v.patch_embd.weight.1")
    c_patch = np.ascontiguousarray(
        wsum.transpose(1, 2, 3, 0).reshape(patch * patch * 3, d)
    )

    # --- token order: merged block-major, sub-patch TL,TR,BL,BR ---
    p = np.arange(npt)
    xpair, ypair = p % 2, (p // 2) % 2
    xm, ym = (p // 4) % mg, p // (mg * 4)
    x, y = 2 * xm + xpair, 2 * ym + ypair  # patch of each merged token

    # --- learned positions at the native grid + the patch bias ---
    pos = w("v.position_embd.weight")  # (npt, d), x-fastest rows
    if pos.shape != (npt, d):
        raise SystemExit(f"position table is {pos.shape}, expected {(npt, d)}")
    c_biaspos = pos[y * g + x] + w("v.patch_embd.bias")[None, :]

    # --- m-rope cos/sin: pairs (j, j+dh/2); rows drive j<dh/4, cols the rest;
    #     each section restarts at theta 1; theta_scale = 10000^(-2/(dh/2)) ---
    n2 = dh // 2
    j = np.arange(n2 // 2)
    theta = 10000.0 ** (-2.0 * j / n2)
    ang = np.concatenate(
        [y[:, None] * theta[None, :], x[:, None] * theta[None, :]], axis=1
    )
    c_cos = np.cos(ang.astype(np.float32))[:, None, :].astype(np.float32)
    c_sin = np.sin(ang.astype(np.float32))[:, None, :].astype(np.float32)

    layers = []
    for i in range(nlayer):
        L = f"v.blk.{i}."
        layers.append(
            {
                "ln1w": w(L + "ln1.weight"),
                "ln1b": w(L + "ln1.bias"),
                "qkvw": np.ascontiguousarray(w(L + "attn_qkv.weight").T),
                "qkvb": w(L + "attn_qkv.bias"),
                "ow": np.ascontiguousarray(w(L + "attn_out.weight").T),
                "ob": w(L + "attn_out.bias"),
                "ln2w": w(L + "ln2.weight"),
                "ln2b": w(L + "ln2.bias"),
                "upw": np.ascontiguousarray(w(L + "ffn_up.weight").T),
                "upb": w(L + "ffn_up.bias"),
                "dnw": np.ascontiguousarray(w(L + "ffn_down.weight").T),
                "dnb": w(L + "ffn_down.bias"),
            }
        )
    const = {
        "side": side,
        "patch": patch,
        "merge": merge,
        "d": d,
        "heads": heads,
        "dh": dh,
        "npt": npt,
        "ntok": ntok,
        "eps": np.float32(eps),
        "kq": np.float32(1.0 / np.sqrt(dh)),
        "nlayer": nlayer,
        "proj": proj,
        "g": g,
        "mg": mg,
        "mean": mean,
        "std": std,
        "wsum": c_patch,
        "biaspos": c_biaspos,
        "cos": c_cos,
        "sin": c_sin,
        "mm0w": np.ascontiguousarray(w("mm.0.weight").T),
        "mm0b": w("mm.0.bias"),
        "mm2w": np.ascontiguousarray(w("mm.2.weight").T),
        "mm2b": w("mm.2.bias"),
        "postw": w("v.post_ln.weight"),
        "postb": w("v.post_ln.bias"),
        "layers": layers,
    }
    return const


def build_model(ops, C, take=None, head=True, tail=True, ln_scale=1.0):
    """Assemble the ov.Model with openvino.opset13; every constant is baked.

    Whole encoder = head=True, tail=True, every layer: image in, embeddings out.
    head=False starts the graph at block input "hidden" [npt, d] instead of at the
    picture; tail=False stops after the last included block and answers "hidden".
    take slices C["layers"].  The halves exist for NPU drivers that will not
    compile all 27 blocks as one graph; chained in order they compute the exact
    same function (--split-after in main).

    ln_scale works around NPU builds that compute a LayerNorm's variance in
    fp16: this encoder's deep blocks run at std ~100, and the squares overflow
    fp16 - measured on an Intel AI Boost with OpenVINO 2026.4.1, 2026-10-23:
    a LayerNorm fed std 13 is clean, std 53 starts to drift, std 105 collapses
    to the bias, std 210 answers NaN.  No fixed divisor fits: dividing by 64
    pushes the shallow LayerNorms' variance (down to ~1e-4) under the fp16
    normal range instead (NaN the other way).  ln_scale="max" divides each
    LayerNorm's input by that row's max |x| (epsilon floored at 1e-6), so every
    row's variance sits near 1 at any depth: scale-invariance makes the
    normalised part the same number, and 1e-6 floors the true eps/(max*max)
    with an error under ~1e-5 of the variance at these magnitudes.  A plain
    number N divides by N and shrinks eps by N*N instead."""
    import openvino as ov

    def con(a):
        return ops.constant(np.ascontiguousarray(np.asarray(a, np.float32)))

    ln_dyn = isinstance(ln_scale, str) and ln_scale == "max"
    lns = 1.0 if ln_dyn else float(ln_scale)
    if ln_dyn:
        # a per-row max keeps every row's variance near 1; the epsilon is then
        # floored at 1e-6, a rounding of the true eps/M*M of at most ~1e-5 of
        # the variance at the magnitudes this encoder runs at (measured)
        eps_c = np.float32(1e-6)
    elif lns != 1.0:
        eps_c = np.float32(float(C["eps"]) / (lns * lns))
    else:
        eps_c = C["eps"]

    def ln(x, wt, bs):
        if ln_dyn:
            mx = ops.constant(np.array([-1], np.int64))
            m = ops.add(ops.reduce_max(ops.abs(x), mx, keep_dims=True), con(np.float32(1e-6)))
            x = ops.divide(x, m)
        elif lns != 1.0:
            x = ops.multiply(x, con(np.float32(1.0 / lns)))
        ax = ops.constant(np.array([-1], np.int64))
        mu = ops.reduce_mean(x, ax, keep_dims=True)
        xc = ops.subtract(x, mu)
        var = ops.reduce_mean(ops.multiply(xc, xc), ax, keep_dims=True)
        den = ops.sqrt(ops.add(var, con(eps_c)))
        return ops.add(ops.multiply(ops.divide(xc, den), con(wt)), con(bs))

    half, one = con(np.float32(0.5)), con(np.float32(1.0))
    gc, ga = con(np.float32(0.7978845608028654)), con(np.float32(0.044715))

    def gelu(x):
        x2 = ops.multiply(x, x)
        inner = ops.multiply(gc, ops.add(x, ops.multiply(ga, ops.multiply(x2, x))))
        return ops.multiply(ops.multiply(x, half), ops.add(one, ops.tanh(inner)))

    side, patch, d, heads, dh = C["side"], C["patch"], C["d"], C["heads"], C["dh"]
    npt, ntok, n2, mg = C["npt"], C["ntok"], dh // 2, C["mg"]
    kc = patch * patch * 3
    if head:
        img = ops.parameter(shape=[1, 3, side, side], dtype=np.float32, name="image")
        p = ops.reshape(
            img,
            ops.constant(
                np.array([1, 3, side // patch, patch, side // patch, patch], np.int64)
            ),
            False,
        )
        p = ops.transpose(p, np.array([0, 2, 4, 1, 3, 5], np.int64))
        # the merged token order (block-major, sub-patches TL,TR,BL,BR) is a pure
        # layout change: what a Gather with C_idx would pick, two reshapes and a
        # transpose say exactly (checked in the tests), and layout ops are what
        # every NPU compiler folds into the layers around it
        p = ops.reshape(
            p, ops.constant(np.array([1, mg, 2, mg, 2, kc], np.int64)), False
        )
        p = ops.transpose(
            p, np.array([0, 1, 3, 2, 4, 5], np.int64)
        )  # (n, ym, xm, ypair, xpair, k)
        p = ops.reshape(p, ops.constant(np.array([npt, kc], np.int64)), False)
        tok = ops.matmul(p, con(C["wsum"]), False, False)
        tok = ops.add(tok, con(C["biaspos"]))
    else:
        img = ops.parameter(shape=[npt, d], dtype=np.float32, name="hidden")
        tok = img

    cos, sin = con(C["cos"]), con(C["sin"])  # (npt, 1, n2)

    def rope(qk):  # (npt, heads, dh)
        sp = ops.split(qk, 2, 2)
        a, b = sp.output(0), sp.output(1)
        r0 = ops.subtract(ops.multiply(a, cos), ops.multiply(b, sin))
        r1 = ops.add(ops.multiply(a, sin), ops.multiply(b, cos))
        return ops.concat([r0, r1], 2)

    t_hbt = ops.constant(np.array([1, 0, 2], np.int64))
    for L in C["layers"] if take is None else C["layers"][take]:
        h = ln(tok, L["ln1w"], L["ln1b"])
        qkv = ops.add(ops.matmul(h, con(L["qkvw"]), False, False), con(L["qkvb"]))
        sp = ops.split(qkv, 1, 3)
        q, k, v = (sp.output(i) for i in (0, 1, 2))
        shp = ops.constant(np.array([npt, heads, dh], np.int64))
        q, k, v = (ops.reshape(t, shp, False) for t in (q, k, v))
        q, k = rope(q), rope(k)
        q16 = ops.transpose(q, t_hbt)  # (heads, npt, dh)
        k16 = ops.transpose(k, t_hbt)
        v16 = ops.transpose(v, t_hbt)
        sc = ops.matmul(
            q16, ops.transpose(k16, np.array([0, 2, 1], np.int64)), False, False
        )
        sm = ops.softmax(ops.multiply(sc, con(C["kq"])), 2)
        o = ops.reshape(
            ops.transpose(ops.matmul(sm, v16, False, False), t_hbt),
            ops.constant(np.array([npt, d], np.int64)),
            False,
        )
        tok = ops.add(
            tok, ops.add(ops.matmul(o, con(L["ow"]), False, False), con(L["ob"]))
        )
        h = ln(tok, L["ln2w"], L["ln2b"])
        f = gelu(ops.add(ops.matmul(h, con(L["upw"]), False, False), con(L["upb"])))
        tok = ops.add(
            tok, ops.add(ops.matmul(f, con(L["dnw"]), False, False), con(L["dnb"]))
        )

    if not tail:
        tok.set_friendly_name("hidden")
        return ov.Model([tok], [img], "mmproj_qwen3vl_head")
    m = ln(tok, C["postw"], C["postb"])
    m = ops.reshape(m, ops.constant(np.array([ntok, d * 4], np.int64)), False)
    z = gelu(ops.add(ops.matmul(m, con(C["mm0w"]), False, False), con(C["mm0b"])))
    z = ops.add(ops.matmul(z, con(C["mm2w"]), False, False), con(C["mm2b"]))
    out = ops.reshape(z, ops.constant(np.array([1, ntok, C["proj"]], np.int64)), False)
    out.set_friendly_name("embeddings")
    return ov.Model([out], [img], "mmproj_qwen3vl")


def C_idx(C):
    """Raster patch index x + g*y of each merged token p: merged block p//4
    holds sub-patches TL,TR,BL,BR at p%4.  The graph reaches this order with
    reshape+transpose (same numbers, friendlier to the NPU compiler); the
    test checks the layout chain against these indices."""
    g, mg = C["g"], C["mg"]
    p = np.arange(C["npt"])
    xm, ym = (p // 4) % mg, p // (mg * 4)
    x, y = 2 * xm + p % 2, 2 * ym + (p // 2) % 2
    return (x + g * y).astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mmproj", help="the mmproj GGUF (clip + qwen3vl_merger)")
    ap.add_argument(
        "--out", required=True, help="encoder .xml to write (the .bin lands next to it)"
    )
    ap.add_argument(
        "--fp16",
        action="store_true",
        help="write the weights as fp16: half the .bin, what the NPU compiles "
        "best, and lossless for a bf16 source file",
    )
    ap.add_argument(
        "--split-after",
        type=int,
        default=0,
        help="write two IRs instead of one, the first with this many blocks: for "
        "NPU drivers that will not run the whole 27-block graph (docs/INTEL_NPU."
        "md); the reader chains them with --model-ir "
        "\"first.xml" + os.pathsep + "second.xml\"",
    )
    ap.add_argument(
        "--ln-scale",
        default="1",
        help='LayerNorm magnitude workaround for NPU builds with fp16 '
        'variance: "max" divides each LayerNorm input by its row max|x| '
        "(the same function, variance near 1 everywhere - what the Intel "
        'NPU here needs); a number N divides by N instead; 1 (the default) '
        "exports the graph unchanged",
    )
    ap.add_argument(
        "--no-check",
        action="store_true",
        help="skip the one-picture CPU smoke run after export",
    )
    a = ap.parse_args()
    try:
        import openvino as ov
        from openvino import opset13 as ops
    except ImportError:
        raise SystemExit(
            "openvino is missing - run this with the NPU venv's "
            "python (.venv-npu/bin/python)"
        )
    meta, T = load_gguf(a.mmproj)
    check_meta(meta)
    C = build_constants(meta, T)
    out = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if a.split_after and not 0 < a.split_after < C["nlayer"]:
        raise SystemExit(f"--split-after wants 1..{C['nlayer'] - 1} (got {a.split_after})")
    stem, ext = os.path.splitext(out)
    parts = (
        [
            (f"{stem}-1{ext}", dict(take=slice(0, a.split_after), tail=False)),
            (f"{stem}-2{ext}", dict(take=slice(a.split_after, None), head=False)),
        ]
        if a.split_after
        else [(out, {})]
    )

    def save(model, path):
        if a.fp16:
            ov.save_model(model, path, compress_to_fp16=True)
        else:
            # "as-is" f32: the exact graph, nothing compressed
            ov.serialize(model, path)

    for path, kw in parts:
        lns_arg = (
            "max"
            if str(a.ln_scale).strip().lower() == "max"
            else float(a.ln_scale)
        )
        save(build_model(ops, C, ln_scale=lns_arg, **kw), path)
        print(f"wrote {path}")
    mean = ",".join(f"{x:g}" for x in C["mean"])
    std = ",".join(f"{x:g}" for x in C["std"])
    print(
        f"  image {C['side']}x{C['side']} -> {C['ntok']} x {C['proj']} "
        f"embeddings; normalize with mean {mean} std {std}"
    )
    if a.split_after:
        print(
            f"  the reader runs the chain: --model-ir {parts[0][0]}{os.pathsep}{parts[1][0]}"
        )
    if not a.no_check:
        core = ov.Core()
        # what was written, not what was built: the smoke run reads the IRs back
        comps = [core.compile_model(path, "CPU") for path, _ in parts]
        rng = np.random.default_rng(0)
        x = rng.uniform(-1.0, 1.0, [1, 3, C["side"], C["side"]]).astype(np.float32)
        for comp in comps:
            x = np.asarray(comp(x)[comp.output(0)], np.float32)
        ok = np.isfinite(x).all()
        print(f"  CPU smoke: output {x.shape} std {x.std():.4f} finite={bool(ok)}")
        if not ok:
            sys.exit("the exported graph produced non-finite numbers")


if __name__ == "__main__":
    main()
