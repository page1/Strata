#!/usr/bin/env python3
"""tools/vision/strata_vision_npu.py - the picture reader on the Intel NPU (OpenVINO).

The standalone, resident image encoder of tools/vision/strata_vision.cpp, with the vision encoder (tower + merger)
run as an OpenVINO IR model on the Intel NPU of a Core Ultra chip (Series 1 "Meteor Lake" / Series 2 "Lunar Lake",
"Arrow Lake" / Series 3) instead of on the CPU or a graphics card.  The NPU is a separate device: reading a picture
takes neither CPU cores nor the VRAM the engine's experts use.

Resident, with the protocol every Strata picture reader speaks (serve/server.py, Vision):

    prints "READY <n_embd>" once it can encode
    stdin  "ENC <image path> <output path>"  ->  "OK <n_tokens> <nx> <ny> <ms>"  or  "ERR <message>"
    stdin  "QUIT"                            ->  exit 0

The output file is  int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd}  then float32 [n_tokens][n_embd],
row i at grid position (x = i % nx, y = i / nx) - the same record the engine reads in --serve GENI.

The model is not the mmproj GGUF (that belongs to mtmd): it is the encoder exported to OpenVINO IR (.xml + .bin)
- one IR, or the halves of a --split-after export joined with os.pathsep in --model-ir and run in order.
The IR takes float32 NCHW [1, 3, H, W], RGB, each channel divided by 255 and normalized (byte/255 - mean) / std
with the mean/std the encoder was exported with - 0.5/0.5, the constants the Qwen3.8-Flash-Next mmproj records -
by default; --image-mean/--image-std or STRATA_VISION_NPU_MEAN/STD carry another encoder's constants (a CLIP
tower trains on 0.48145466,0.4578275,0.40821073 over 0.26862954,0.26130258,0.27577711).  It answers with float32
[n_tokens, n_embd] in the row-major grid order above.  Static input shapes are what the NPU compiles best; a
dynamic H, W works on the platforms whose NPU compiler takes them, and the reader then picks a square size from
--max-tokens or --image-size.

    python tools/vision/strata_vision_npu.py --model-ir encoder.xml [--device NPU] [--cache-dir DIR] [--embd N]
    python tools/vision/strata_vision_npu.py --anysize DIR [--max-tokens N] [--device NPU] [--cache-dir DIR]
    python tools/vision/strata_vision_npu.py --mock              # deterministic stand-in, no OpenVINO or NPU needed

--anysize reads an export of tools/vision/npu_anysize.py instead of one IR: every picture keeps its aspect ratio
and is sized the way mtmd sizes it (up to --max-tokens, default the export's largest budget), as the CPU and GPU
readers do, instead of being squeezed into the one size the IR was exported for.

The flags the server hands every reader (--mmproj, --model, --gpu, --threads, --max-tokens, --min-tokens,
--flash-attn) are accepted so a run config's vision section can name this program; only --threads (on --device
CPU) and --max-tokens / --min-tokens (with a dynamic IR input shape) carry meaning here.  A run config cannot add
flags to a reader, only an exe, so --model-ir, --anysize, --device, --cache-dir and --mock also read
STRATA_VISION_NPU_MODEL_IR, STRATA_VISION_NPU_ANYSIZE, STRATA_VISION_NPU_DEVICE, STRATA_VISION_NPU_CACHE_DIR and
STRATA_VISION_NPU_MOCK from the server's environment.

Unlike the GPU readers this one never quietly becomes something else: with no NPU, no OpenVINO or an unreadable IR
it prints "ERR <what and the fix>" on stdout and exits - the server then falls back to the CPU reader by itself
(with a cpu_exe in the section), and a person running it by hand reads the reason.  --mock is the only way to get
the stand-in engine, and it says so on stderr at every start.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:  # --help and the error lines must still work without it
    np = None

try:
    from PIL import Image
except ImportError:
    Image = None

SVE1_MAGIC = 0x31455653
# The pixels as the exported IR wants them: (byte/255 - MEAN) / STD.  0.5/0.5 is what the qwen3vl_merger
# mmprojs record in clip.vision.image_mean/std and what tools/vision/export_npu_encoder.py exports for;
# --image-mean/--image-std (or the env) cover an IR built from an encoder trained with other constants.
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


class ReaderError(Exception):
    """Cannot serve pictures as configured: one plain line for stdout, the server reads it and falls back."""


class DeviceError(ReaderError):
    """The runtime or the NPU itself is not there (the fix is the driver, the wheel or --device), exit 3 like the
    GPU readers when they find no card."""


def parse_enc(line: str) -> tuple[str, str] | None:
    """Parse "ENC <image> <out>" the way the C++ readers do: the image path ends at the last " " before the
    output path, so a hand-typed path may contain spaces (the server names its own files without any)."""
    if not line.startswith("ENC "):
        return None
    rest = line[4:]
    sp = rest.rfind(" ")
    if sp == -1 or sp == 0:
        return None
    img, out = rest[:sp], rest[sp + 1 :]
    return (img, out) if img and out else None


def ir_paths(spec: str) -> list:
    """The IR files a --model-ir / STRATA_VISION_NPU_MODEL_IR value names: one path, or
    two or more joined with os.pathsep (":" on Linux and Mac, ";" on Windows) to run as
    a chain - the exported encoder split in halves for an NPU driver that will not
    compile all 27 blocks as one graph (docs/INTEL_NPU.md).  Chained, the halves compute
    the exact function of the whole."""
    paths = [p.strip() for p in spec.split(os.pathsep) if p.strip()]
    if not paths:
        raise ReaderError("no OpenVINO IR path in the model setting")
    return paths


def one_line(text: object) -> str:
    """One ERR line out of anything, traceback-free: newlines gone, length kept sane for the server's log."""
    return " ".join(str(text).split())[:300]


def triple_of(text: str, fallback: tuple, what: str) -> tuple:
    """Parse an "r,g,b" normalization triple (--image-mean/--image-std, one value repeats).  The std must be
    positive - a zero there turns the picture into infinities the IR answers with NaNs."""
    if not text:
        return fallback
    parts = [x.strip() for x in text.replace(";", ",").split(",") if x.strip()]
    try:
        vals = tuple(float(x) for x in parts)
    except ValueError:
        raise ReaderError(
            f"--image-{what} wants numbers like 0.5,0.5,0.5 (got {text!r})"
        )
    if len(vals) == 1:
        vals = vals * 3
    if len(vals) != 3:
        raise ReaderError(f"--image-{what} wants one or three numbers, got {len(vals)}")
    if what == "std" and any(v <= 0 for v in vals):
        raise ReaderError("--image-std numbers must be above 0")
    return vals


def grid_of(w: int, h: int, patch: int, merge: int) -> tuple[int, int]:
    """The token grid (nx, ny) of a w x h picture with `patch`-pixel patches merged `merge` by `merge`: what the
    merger's output rows run through, row i at (x = i % nx, y = i / nx)."""
    side = patch * merge
    if w % side or h % side:
        raise ReaderError(
            f"the picture size {w}x{h} is not a multiple of patch {patch} x merge {merge} ({side})"
        )
    return w // side, h // side


def pick_size(
    shape: list,
    image_size: int,
    patch: int,
    merge: int,
    max_tokens: int,
    min_tokens: int,
) -> tuple[int, int]:
    """The (W, H) to resize every picture to, from the IR's input shape: static H, W win as they are (the NPU
    compiles them best); with a dynamic H, W a square is picked that fits --max-tokens (--image-size without
    it), and --min-tokens raises the floor.  A static size the token bounds disagree with is reported, not fixed:
    it is the export, not the picture, that decided it."""
    if len(shape) != 4:
        raise ReaderError(
            f"the IR input is {len(shape)}-dimensional, not the 4 this reader expects [1, 3, H, W]"
        )
    h, w = int(shape[2]), int(shape[3])
    if h > 0 and w > 0:
        nx, ny = grid_of(
            w, h, patch, merge
        )  # checked here, once, so READY never promises a broken size
        n = nx * ny
        if max_tokens and n > max_tokens:
            raise ReaderError(
                f"the IR input {w}x{h} gives {n} image tokens, above --max-tokens {max_tokens}"
            )
        if min_tokens and n < min_tokens:
            raise ReaderError(
                f"the IR input {w}x{h} gives {n} image tokens, below --min-tokens {min_tokens}"
            )
        return w, h
    if shape[2] < -1 or shape[3] < -1:
        raise ReaderError(f"the IR input shape {list(shape)} has no usable H, W")
    # at least one of H, W is dynamic (-1): a square side that fits the token bounds, the static side staying as it is
    side = image_size
    if max_tokens:
        side = max(math.isqrt(max_tokens), 1) * patch * merge
    elif min_tokens:
        side = math.ceil(math.sqrt(min_tokens)) * patch * merge
    side -= side % (patch * merge)
    w, h = (w if w > 0 else side), (h if h > 0 else side)
    grid_of(w, h, patch, merge)  # the same config check as the fully static case
    return w, h


def preprocess(
    path: str, w: int, h: int, mean: tuple = MEAN, std: tuple = STD
) -> "np.ndarray":
    """The image as the IR expects it: RGB, resized w x h with bicubic, /255, (x - mean) / std, CHW, batch 1."""
    if Image is None:
        raise ReaderError("Pillow is not installed (pip install pillow)")
    if np is None:
        raise ReaderError("NumPy is not installed (pip install numpy)")
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (w, h):
            im = im.resize((w, h), Image.Resampling.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...])


class MockEngine:
    """A deterministic stand-in with the same shape rules as the real one: every output token is a fixed
    projection of its picture region's mean colour, so tests check the protocol, the grid and the numbers
    without OpenVINO or an NPU.  It is not a vision encoder - nothing should trust what it answers."""

    def __init__(self, n_embd: int = 2048):
        self.n_embd = n_embd
        self.device = "MOCK"

    def infer(self, tensor: "np.ndarray", nx: int, ny: int) -> "np.ndarray":
        _, _, h, w = tensor.shape
        cell_h, cell_w = h // ny, w // nx
        pooled = (
            tensor[0].reshape(3, ny, cell_h, nx, cell_w).mean(axis=(2, 4))
        )  # [3, ny, nx]
        flat = np.ascontiguousarray(pooled.transpose(1, 2, 0)).reshape(
            ny * nx, 3
        )  # row i at (i % nx, i // nx): x runs fastest
        proj = np.random.RandomState(42).randn(3, self.n_embd).astype(np.float32) * 0.05
        embd = flat @ proj
        return (embd / (np.linalg.norm(embd, axis=-1, keepdims=True) + 1e-6)).astype(
            np.float32
        )

    def close(self) -> None:
        pass


class OpenVinoEngine:
    """The encoder as an OpenVINO IR model on the Intel NPU (or the OpenVINO device --device asks for)."""

    def __init__(
        self, model_ir: str, device: str, cache_dir: str | None = None, threads: int = 0
    ):
        try:
            import openvino as ov
        except ImportError:
            raise DeviceError(
                "OpenVINO is not installed for this Python (pip install openvino); "
                "or run with --mock to test the reader without one"
            ) from None
        self.core = ov.Core()
        available = list(self.core.available_devices)
        if device.upper() not in [d.upper() for d in available]:
            # the plain word the server's fallback line and a person both read: which device is missing
            raise DeviceError(
                f"this PC has no {device.upper()} that OpenVINO can see (it sees: "
                f"{', '.join(available) or 'nothing'}); install the Intel NPU driver, pick another "
                f"--device, or run the CPU reader strata-vision instead"
            )
        self.device = device.upper()
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.core.set_property(
                self.device, {"CACHE_DIR": cache_dir}
            )  # the NPU compiles slowly (measured 170 s + 190 s for the chain); from a warm cache the
            # same two compiles take 1.2 s + 1.3 s (2026-10-23) - set STRATA_VISION_NPU_CACHE_DIR to use it
        cfg: dict = {}
        if threads > 0 and self.device.startswith("CPU"):
            cfg["INFERENCE_NUM_THREADS"] = threads
        # one IR, or two chained (ir_paths): every one must compile before READY
        self.compiled = []
        self.requests = []
        for path in ir_paths(model_ir):
            try:
                model = self.core.read_model(path)
                comp = self.core.compile_model(model, self.device, cfg)
            except Exception as e:
                raise ReaderError(
                    f"the {self.device} could not read or compile the encoder "
                    f"{path}: {one_line(e)}"
                ) from None
            self.compiled.append(comp)
            self.requests.append(comp.create_infer_request())
        for up, low in zip(self.compiled, self.compiled[1:]):
            if [int(d) for d in up.output(0).shape] != [
                int(d) for d in low.input(0).shape
            ]:
                raise ReaderError(
                    "the IR chain does not join: one model's output shape is not the "
                    "next model's input shape"
                )
        self.input_shape = [int(d) for d in self.compiled[0].input(0).shape]
        self.output_shape = [int(d) for d in self.compiled[-1].output(0).shape]
        self.version = ov.get_version()

    @property
    def n_embd(self) -> int:
        """The embedding width from the IR's output (0 when it is dynamic and only --embd can say)."""
        return (
            self.output_shape[-1]
            if self.output_shape and self.output_shape[-1] > 0
            else 0
        )

    def infer(self, tensor: "np.ndarray", nx: int, ny: int) -> "np.ndarray":
        x = tensor  # one compiled model after the other: each takes what the last answered
        for comp, request in zip(self.compiled, self.requests):
            x = np.asarray(
                request.infer({comp.input(0): x})[comp.output(0)], dtype=np.float32
            )
        return np.ascontiguousarray(x.reshape(-1, x.shape[-1]))

    def close(self) -> None:
        try:
            del self.requests, self.compiled, self.core
        except Exception:
            pass


class AnySizeEngine:
    """An npu_anysize.py export on the NPU: one compiled graph set per token budget, and each picture at its own
    size (mtmd's sizing, up to max_tokens), padded to the smallest budget that holds it."""

    def __init__(self, export_dir: str, device: str, cache_dir: str | None, max_tokens: int, min_tokens: int):
        try:
            import openvino as ov
        except ImportError:
            raise DeviceError(
                "OpenVINO is not installed for this Python (pip install openvino); "
                "or run with --mock to test the reader without one"
            ) from None
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import npu_anysize

        self.core = ov.Core()
        available = list(self.core.available_devices)
        if device.upper() not in [d.upper() for d in available]:
            raise DeviceError(
                f"this PC has no {device.upper()} that OpenVINO can see (it sees: "
                f"{', '.join(available) or 'nothing'}); install the Intel NPU driver, pick another "
                f"--device, or run the CPU reader strata-vision instead"
            )
        self.device = device.upper()
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.core.set_property(self.device, {"CACHE_DIR": cache_dir})
        if not os.path.isfile(os.path.join(export_dir, "anysize.json")):
            raise ReaderError(
                f"{export_dir} is not an any-size export (no anysize.json): make one with "
                "tools/vision/npu_anysize.py export"
            )
        try:
            self.enc = npu_anysize.compile_encoder(export_dir, self.core, self.device, max_tokens, min_tokens)
        except ValueError as e:
            raise ReaderError(one_line(e)) from None
        except Exception as e:
            raise ReaderError(
                f"the {self.device} could not read or compile the any-size export {export_dir}: {one_line(e)}"
            ) from None
        self.budgets = list(self.enc.side["budgets"])
        self.max_tokens = self.enc.max_tokens

    @property
    def n_embd(self) -> int:
        return self.enc.n_embd

    def encode(self, path: str, mean: tuple, std: tuple) -> tuple["np.ndarray", int, int]:
        return self.enc.encode(path, mean, std)

    def warm(self) -> None:
        self.enc.warm()

    def close(self) -> None:
        try:
            del self.enc, self.core
        except Exception:
            pass


def build_engine(args) -> tuple[object, tuple[int, int], int]:
    """(engine, (W, H), n_embd) from the flags, or a ReaderError/DeviceError with the fix in its line."""
    if args.mock:
        eng = MockEngine(n_embd=args.embd or 2048)
        size = (args.image_size, args.image_size)
        grid_of(
            *size, args.patch_size, args.merge_size
        )  # same config check as the real engine gets
        return eng, size, eng.n_embd
    anysize = args.anysize or os.environ.get("STRATA_VISION_NPU_ANYSIZE", "")
    if anysize:
        eng = AnySizeEngine(anysize, args.device, args.cache_dir, args.max_tokens, args.min_tokens)
        return eng, None, args.embd or eng.n_embd  # no fixed size: every picture is sized on its own
    ir = args.model_ir or os.environ.get("STRATA_VISION_NPU_MODEL_IR", "")
    if not ir:
        raise ReaderError(
            "no --model-ir <encoder.xml> given (or set STRATA_VISION_NPU_MODEL_IR): this reader runs "
            "an OpenVINO IR export of the vision encoder, not the mmproj GGUF; --mock tests it "
            "without a model"
        )
    # --device already takes its default from STRATA_VISION_NPU_DEVICE (see the parser)
    eng = OpenVinoEngine(ir, args.device, args.cache_dir, args.threads)
    w, h = pick_size(
        eng.input_shape,
        args.image_size,
        args.patch_size,
        args.merge_size,
        args.max_tokens,
        args.min_tokens,
    )
    n_embd = args.embd or eng.n_embd
    if not n_embd:
        raise ReaderError(
            f"the IR output shape {eng.output_shape} has no fixed width: pass --embd"
        )
    return eng, (w, h), int(n_embd)


def encode_one(
    eng,
    img: str,
    out: str,
    size: tuple[int, int],
    patch: int,
    merge: int,
    n_embd: int,
    mean: tuple = MEAN,
    std: tuple = STD,
) -> tuple[int, int, int, float]:
    """One picture -> the SVE1 file; -> (n_tokens, nx, ny, ms)."""
    t0 = time.perf_counter()
    if not os.path.isfile(img):
        raise FileNotFoundError(f"cannot read the image {img}")
    if size is None:  # an any-size engine sizes each picture itself
        embd, nx, ny = eng.encode(img, mean, std)
        n = nx * ny
        if embd.shape != (n, n_embd):
            raise RuntimeError(f"the encoder answered {embd.shape}, not the ({n}, {n_embd}) its grid promises")
        data = struct.pack("<5i", SVE1_MAGIC, n, nx, ny, n_embd) + embd.tobytes()
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            f.write(data)
        return n, nx, ny, (time.perf_counter() - t0) * 1000.0
    tensor = preprocess(img, size[0], size[1], mean, std)
    nx, ny = grid_of(size[0], size[1], patch, merge)
    embd = eng.infer(tensor, nx, ny)
    n = nx * ny
    if embd.shape != (n, n_embd):
        raise RuntimeError(
            f"the encoder answered {embd.shape}, not the ({n}, {n_embd}) its grid promises"
        )
    data = struct.pack("<5i", SVE1_MAGIC, n, nx, ny, n_embd) + embd.tobytes()
    Path(out).parent.mkdir(
        parents=True, exist_ok=True
    )  # the server names plain files in its own cwd
    with open(out, "wb") as f:
        f.write(data)
    return n, nx, ny, (time.perf_counter() - t0) * 1000.0


def main() -> int:
    p = argparse.ArgumentParser(
        prog="strata-vision-npu", description=__doc__.splitlines()[0]
    )
    p.add_argument(
        "--model-ir",
        default="",
        help="OpenVINO IR of the vision encoder (.xml; .onnx too), or a chain of "
        'two joined with "' + os.pathsep + '" (docs/INTEL_NPU.md); '
        "or STRATA_VISION_NPU_MODEL_IR",
    )
    p.add_argument(
        "--anysize",
        default="",
        help="an export of tools/vision/npu_anysize.py: pictures keep their own size (up to --max-tokens) "
        "instead of the one size of --model-ir; or STRATA_VISION_NPU_ANYSIZE",
    )
    p.add_argument(
        "--device",
        default=os.environ.get("STRATA_VISION_NPU_DEVICE", "NPU"),
        help="OpenVINO device: NPU (the default), GPU or CPU",
    )
    p.add_argument(
        "--cache-dir",
        default=os.environ.get("STRATA_VISION_NPU_CACHE_DIR", ""),
        help="OpenVINO compilation cache directory; keeps the NPU's slow compile to one start "
        "(also STRATA_VISION_NPU_CACHE_DIR); the NPU driver's own cache stays on without it",
    )
    p.add_argument(
        "--embd",
        type=int,
        default=0,
        help="embedding width when the IR does not say it (the mock's: 2048)",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="square picture side when the IR input is dynamic (default 512)",
    )
    p.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="the encoder's patch side (default 16)",
    )
    p.add_argument(
        "--merge-size",
        type=int,
        default=2,
        help="the merger's merge factor (default 2)",
    )
    p.add_argument(
        "--image-mean",
        default=os.environ.get("STRATA_VISION_NPU_MEAN", ""),
        help="normalization mean as r,g,b (default 0.5,0.5,0.5, the Flash-Next encoder's; a CLIP tower "
        "uses 0.48145466,0.4578275,0.40821073); or STRATA_VISION_NPU_MEAN",
    )
    p.add_argument(
        "--image-std",
        default=os.environ.get("STRATA_VISION_NPU_STD", ""),
        help="normalization std as r,g,b (default 0.5,0.5,0.5; a CLIP tower uses "
        "0.26862954,0.26130258,0.27577711); or STRATA_VISION_NPU_STD",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="deterministic stand-in encoder: no OpenVINO, no NPU, no model",
    )
    p.add_argument(
        "--version", action="store_true", help="print the reader's build line and exit"
    )
    # what serve/server.py Vision hands every reader; only --threads and the token bounds mean anything here
    p.add_argument(
        "--mmproj",
        default="",
        help="mmproj GGUF (belongs to the mtmd readers; ignored here)",
    )
    p.add_argument("--model", default="", help="text model GGUF (ignored here)")
    p.add_argument(
        "--gpu",
        action="store_true",
        help="the mtmd readers' GPU flag (ignored here: the NPU is its own device)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="inference threads, applied on --device CPU only",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="largest picture grid: with --anysize the most tokens a picture gets (default the export's "
        "largest budget); with --model-ir used when the IR input is dynamic",
    )
    p.add_argument(
        "--min-tokens", type=int, default=0, help="smallest picture grid, same"
    )
    p.add_argument(
        "--flash-attn",
        default="auto",
        help="the mtmd readers' flag (nothing to switch on the NPU)",
    )
    args = p.parse_args()
    if (
        os.environ.get("STRATA_VISION_NPU_MOCK") == "1"
    ):  # the run-config escape hatch for --mock (an exe, no flags)
        args.mock = True
    try:  # an unusable norm triple is the config's fault: one ERR line, no traceback
        norm = (
            triple_of(args.image_mean, MEAN, "mean"),
            triple_of(args.image_std, STD, "std"),
        )
    except ReaderError as e:
        print(f"ERR {one_line(e)}")
        return 1

    if args.version:
        try:
            import openvino

            print(f"strata-vision-npu (OpenVINO {openvino.get_version()})")
        except ImportError:
            print("strata-vision-npu (OpenVINO not installed)")
        return 0

    # the server pipes these in UTF-8 (its popen says so); Windows would otherwise read paths in the ANSI page
    for stream in (sys.stdin, sys.stdout):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    say = lambda line: (sys.stdout.write(line + "\n"), sys.stdout.flush())  # noqa: E731
    note = lambda line: (
        sys.stderr.write(f"strata-vision-npu: {line}\n"),
        sys.stderr.flush(),
    )  # noqa: E731

    if args.mock:
        note(
            "mock engine on purpose (--mock): the answers are a stand-in, not a vision encoder"
        )
    try:
        eng, size, n_embd = build_engine(args)
    except DeviceError as e:
        say(f"ERR {one_line(e)}")
        return 3
    except (
        Exception
    ) as e:  # nothing here may leave a traceback where the server reads a line
        say(f"ERR {one_line(e)}")
        return 1

    # one encode before READY, as the GPU readers warm up before it: the first picture then does not pay for
    # graph starts and buffer allocations while a request waits
    try:
        t0 = time.perf_counter()
        if size is None:
            eng.warm()
            shape = f"any size up to {eng.max_tokens} tokens (budgets {', '.join(map(str, eng.budgets))})"
        else:
            if not isinstance(eng, MockEngine):
                eng.infer(
                    np.zeros((1, 3, size[1], size[0]), dtype=np.float32),
                    *grid_of(*size, args.patch_size, args.merge_size),
                )
            shape = f"{size[0]}x{size[1]}"
        note(
            f"engine {eng.device}, input {shape}, n_embd {n_embd}, norm "
            f"{norm[0]} over {norm[1]}, warm encode {time.perf_counter() - t0:.1f} s"
        )
    except Exception as e:
        eng.close()
        say(
            f"ERR the {getattr(eng, 'device', args.device)} failed the warm-up encode: {one_line(e)}"
        )
        return 1

    say(f"READY {n_embd}")
    while True:
        try:
            line = sys.stdin.readline()
        except (
            OSError,
            ValueError,
        ):  # the server is gone: the pipe broke under the read
            break
        if (
            not line
        ):  # stdin closed (the server ended the pipe): end too, as the C++ readers
            break
        line = line.rstrip("\r\n")
        if line == "QUIT":
            break
        if not line:
            continue
        parsed = parse_enc(line)
        if not parsed:
            say("ERR expected: ENC <image> <output>")
            continue
        try:
            n, nx, ny, ms = encode_one(
                eng,
                parsed[0],
                parsed[1],
                size,
                args.patch_size,
                args.merge_size,
                n_embd,
                *norm,
            )
            say(f"OK {n} {nx} {ny} {ms:.0f}")
        except Exception as e:  # one bad picture must not end the resident reader
            say(f"ERR {one_line(e)}")
    eng.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
