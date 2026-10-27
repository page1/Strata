# Reading pictures on the Intel NPU (`strata-vision-npu`)

Strata's picture reader normally runs on the CPU (llama.cpp's mtmd) or on a graphics card.  This page is about the
opt-in third place for it: the **Intel NPU** of a Core Ultra chip, through Intel's OpenVINO runtime.  The utility is
`tools/vision/strata_vision_npu.py`.  It is standalone - one Python file, its own command line, no engine needed to
try it - and it speaks the same resident protocol as the other readers, so the server can use it as a drop-in.
Nothing installs or switches it on by itself: setup, the app and `strata launch` keep their CPU / GPU readers.

Why the NPU: it is a third device.  The CPU reader takes cores the server and the engine want; the GPU reader takes
about 1.1 GB of VRAM the experts would otherwise use (#2406).  The NPU takes neither.

## What you need

- An Intel Core Ultra with an NPU: Series 1 (Meteor Lake), Series 2 (Lunar Lake, Arrow Lake) or Series 3.
- The Intel NPU driver from Intel (Windows or Linux).  OpenVINO's own installer pages link it.
- OpenVINO itself: `pip install -r requirements-npu.txt` (one wheel; it carries the NPU plugin).  NumPy and Pillow
  are already in Strata's `requirements.txt`.  The `openvino` package is **not** added to the default install.
- A picture encoder exported to OpenVINO IR: `tools/vision/export_npu_encoder.py` builds it from the mmproj
  GGUF you already have (next section), in the venv's python.  Nothing is downloaded for it and the reader
  itself never reads the GGUF at run time.

## The model this reader runs

The official one is the Qwen3.8-Flash-Next vision encoder (27 blocks, hidden 1152, patch 16, merge 2, the
`qwen3vl_merger` projector), exported from the very mmproj GGUF the CPU reader uses:

```
.venv-npu/bin/python tools/vision/export_npu_encoder.py \
    ~/projects/Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf \
    --out ~/models/npu-vision/encoder.xml --fp16 \
    --split-after 13 --ln-scale max                       # the form this page's numbers were measured on
```

The exporter builds the graph llama.cpp's mtmd builds for one still image at the native 768x768 - both
patch-embed halves summed, tokens in merged order (each 2x2 block four adjacent tokens, TL,TR,BL,BR), learned
positions, 27 M-RoPE attention blocks, the 4608 -> 4608 -> 2560 merger - and bakes the positions, the token
order and the rope tables in as constants, so the IR is one tensor in, one tensor out.  `--fp16` halves the
`.bin` and is lossless for the bf16 file; the NPU wants it.  The contract:

- input `float32 [1, 3, 768, 768]`, RGB, each pixel `byte/255` normalized `(x - mean) / std`.  The reader's
  defaults are mean `0.5,0.5,0.5`, std `0.5,0.5,0.5` - the constants this mmproj records
  (`clip.vision.image_mean/std`) and what the CPU reader applies through mtmd.  `--image-mean` /
  `--image-std` carry another encoder's; a CLIP tower trains on mean `0.48145466 0.4578275 0.40821073`,
  std `0.26862954 0.26130258 0.27577711`;
- output `float32 [576, 2560]` - one row per merged patch, row `i` at grid position (`x = i % 24`,
  `y = i // 24`), the same record (`SVE1`) the engine reads in `--serve GENI`;
- the token grid is `(W / (patch x merge)) x (H / (patch x merge))`: pass `--patch-size` / `--merge-size` to match
  the export (defaults 16 and 2).

`--split-after` and `--ln-scale` dodge two Intel NPU driver bugs measured on our machine (Core Ultra 7 265K,
OpenVINO 2026.4.1, 2026-10-23; the numbers are in the section at the end).  A single compiled graph over about
850 MB of weights hangs that NPU at its first inference - the whole 27-block export is 896 MB, so
`--split-after 13` writes two IRs instead (blocks 1-13, then blocks 14-27 with the merger) and the reader runs
them one after the other; chained, they compute the same function.  The second bug is smaller and meaner: the
NPU computes some LayerNorms with fp16 accumulators, and this encoder's deep blocks carry std ~100, whose
squares overflow fp16 - the LayerNorm then answers a constant, quietly.  `--ln-scale max` divides each
LayerNorm's input by that row's max |x|; LayerNorm undoes any scaling, so the function is unchanged (the CPU
check moved by 5e-06) and no variance nears the fp16 edges at any depth.  A driver without these bugs runs the
same files either way; the plain export still works on CPU and Intel graphics.

A chain goes in one setting, the two IRs joined with the OS path separator (`:` on Linux and Mac, `;` on
Windows):

```
STRATA_VISION_NPU_MODEL_IR=$HOME/models/npu-vision/encoder-1.xml:$HOME/models/npu-vision/encoder-2.xml
```

Any other export that fills this contract works too (`optimum-intel` or an ONNX export converted with `ovc`,
with the normalization and grid matched); the exporter covers this one model and refuses others with a plain
line.  OpenVINO's own GGUF frontend converts `qwen3vl_merger` mmprojs natively as of OpenVINO master (it
needs a `patch_indices` / `position_ids` pair and a nightly build) - worth revisiting when it reaches a
stable release.

Static input shapes are what the NPU compiler is documented to like; the reader uses an IR's fixed `H, W` as they
are.  With a dynamic `H, W` it picks a square size that fits the server's `--max-tokens` (or `--image-size`).

## Running it

```
python tools/vision/strata_vision_npu.py --version
python tools/vision/strata_vision_npu.py --mock          # no model, no driver: a deterministic stand-in, to see the protocol
python tools/vision/strata_vision_npu.py --model-ir encoder.xml            # resident: READY, then ENC lines on stdin
```

Resident protocol (as in `tools/vision/strata_vision.cpp`): it prints `READY <n_embd>`, answers each
`ENC <image path> <output path>` line with `OK <n_tokens> <nx> <ny> <ms>` (the embeddings file) or one `ERR` line,
and ends on `QUIT`.  Everything on stdout is those lines; progress goes to stderr.  A missing NPU, a missing
OpenVINO or an unreadable IR is one `ERR` line and exit 3 or 1 - it never quietly answers with the stand-in,
because noise where a picture belongs is worse than no picture.

Flags beyond the model: `--device NPU|GPU|CPU` (default `NPU`; an Intel graphics device compiles the IR too -
the OpenVINO NVIDIA plugins do not - but the GPU readers are the better GPU answer), `--cache-dir` (OpenVINO's
compilation cache; it is what makes the slow NPU compile a once-ever cost - see "Startup compile and its
caches"), `--embd` (width when the IR does not say), `--image-size`,
`--patch-size`, `--merge-size`, `--image-mean` / `--image-std`.  `--mmproj`, `--model`, `--gpu`, `--threads`, `--max-tokens`, `--min-tokens` and
`--flash-attn` are accepted because the server hands every reader those; here only `--threads` (on `--device CPU`)
and the token bounds (with a dynamic IR) mean anything.  A run config names an exe and no flags, so `--model-ir`,
`--device`, `--cache-dir`, `--mock`, `--image-mean` and `--image-std` also read `STRATA_VISION_NPU_MODEL_IR`,
`STRATA_VISION_NPU_DEVICE`, `STRATA_VISION_NPU_CACHE_DIR`, `STRATA_VISION_NPU_MOCK`, `STRATA_VISION_NPU_MEAN` and
`STRATA_VISION_NPU_STD` from the environment.

## Startup compile and its caches

Every process that opens the IR compiles it for the NPU before `READY`: measured 170 s + 190 s of one core at
100% for the chain of two IRs.  Two caches sit under that cost, and they behave very differently (checked
2026-10-23):

- The NPU driver's own cache (`~/.cache/ze_intel_npu_cache`, or `/root/.cache/...` in a container) keeps exactly
  one compiled graph of this size: one 604 MB blob, and the second graph's write replaces the first's.  The
  driver's `ZE_INTEL_NPU_CACHE_SIZE` changed nothing observed (checked with 4 GiB).  With the chain, one of the
  two graphs therefore compiles cold at every start - about 3 minutes.
- OpenVINO's `--cache-dir` / `STRATA_VISION_NPU_CACHE_DIR` keeps one blob per model and evicts nothing here:
  after one cold start the same two compiles took 1.2 s + 1.3 s.  Point it at a directory that survives, and
  the compile is a once-ever cost; re-exporting the IR invalidates it.  The whole reader - process start, both
  compiles from cache, one 768x768 picture encoded - takes 7 s end to end (2026-10-23; 6 min 3 s from a cold
  cache).  One care: the cache keys on how the program loads the IR (`compile_model(path)` and
  `read_model(path)` + `compile_model(model)` are different keys), so the cache a helper script fills is not
  the one the reader reads; warm it with the reader itself once.

Inside one server run the reader is resident, so pictures after the first pay none of this: the encode itself
is 2.2 s with about no CPU while the NPU works.

## Pictures at any size (`--anysize`, opt-in)

The IR above has one input size, 768x768, so every picture is resized to it.  For a photo that is fine.  For a
screenshot of small text it is not: a 2000x920 screenshot squeezed to 768x768 keeps 38% of its width, its letters
end up 2-3 pixels wide, and the model guesses words instead of reading them (seen on Windows, 2026-10-26).  The CPU
and GPU readers keep the aspect ratio and give a picture up to `max_tokens` tokens; `tools/vision/npu_anysize.py`
does the same on the NPU.

```
.venv-npu/bin/python tools/vision/npu_anysize.py export \
    ~/projects/Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf --out-dir ~/models/npu-vision/anysize
.venv-npu/bin/python tools/vision/strata_vision_npu.py --anysize ~/models/npu-vision/anysize \
    --max-tokens 2048 --cache-dir ~/models/npu-vision/cache          # or STRATA_VISION_NPU_ANYSIZE
```

How it works:

- The graph knows a token **budget**, not a picture size.  The export compiles a few budgets (default 576, 1024,
  1536 and 2048 tokens).  The reader sizes each picture the way mtmd does (aspect kept, sides multiples of 32,
  letterboxed on black, at most `--max-tokens`), and pads it with masked rows to the smallest budget that holds it.
  The masked rows change nothing: attention never looks at them, and every other step is per row.  The reader
  hands the graph what depends on the picture's shape: the patches in merged order, the learned positions resized
  to the grid (bilinear, align corners, as mtmd), and the rope tables.
- **Attention runs in chunks of 512 query rows.**  A budget of 2048 tokens is 8192 patch rows.  As one score matrix
  per head (8192 x 8192), one block did not compile in 15 minutes, and the whole encoder did not compile in 9 hours
  (85 GB of RAM).  In chunks of 512 x 8192 one block compiled in 29 s.  The numbers are the same, because softmax is
  per query row.  The cliff on this driver lies between about 5.3 and 8.4 million scores in one step.
  `NPU_COMPILER_TYPE=PLUGIN`, `NPU_COMPILATION_MODE_PARAMS=optimization-level=0/1` and `NPU_TILES=1` did not move
  it.
- **One graph runs every block.**  The 27 blocks have one shape, so the export writes one block graph whose weights
  are fp16 inputs (`blocks.npz`), plus a head (patch embedding) and a tail (merger), and the reader calls the block
  27 times.  Compiling all four budgets takes about 2 minutes and 2.6 GB of RAM; 27 separate one-block graphs for
  the 2048 budget alone took 18 minutes and 17.6 GB.  The weights going in as inputs cost about 7% of encode time
  (6.0 s against 5.6 s at 1024 tokens).
- Don't compile these graphs with `NPU_TILES=1`.  On the Windows driver below it compiled, then hung the NPU at the
  first encode (`ZE_RESULT_ERROR_DEVICE_LOST`).  The device recovered, but the reset was felt across the whole PC
  as an audio glitch.

Measured on Windows (the PC under "What was checked"), 10 app and terminal screenshots from this repository's
docs, from 800x450 up to 3200x1910, at `--max-tokens 2048`, one resident reader:

| budget used | tokens | NPU per picture | the same pictures, CPU `strata-vision` (12 threads) | GPU `strata-vision --gpu` (RTX 3090) |
|---|---|---|---|---|
| 576 | 350-494 | 2.9 s | 2.8-4.2 s | 0.07-0.11 s |
| 1536 | 1410 | 12.0 s | 15.3 s | 0.50 s |
| 2048 | 1972-2035 | 24.9-25.2 s | 24.5-25.5 s | 0.90-0.96 s |
| all 10 | 16,303 | 193 s | 196 s (2 readers x 12 threads: 147 s) | 7.2 s |

(The CPU's per-picture times come from its output files' timestamps; the CPU and GPU readers don't use budgets,
so their rows are the same pictures.)

- **Accuracy:** cosine per token row against the CPU reader: 0.989868 mean, 1st percentile 0.834 (GPU reader:
  0.995824, 0.929).  At the native 768x768 the any-size graph computes the same function as the fixed IR.
- **Energy** (the CPU package power from Windows' Energy Meter, which includes the NPU, and the GPU's
  `power.draw`): the NPU added 6.7 W over idle, about 143 J per picture.  The GPU added about 293 J per picture
  (318 W for 9 s), and the CPU 2841 J (214 W).
- **Startup:** 119 s with an empty `--cache-dir` (all four budgets); 5.3 s to `READY` with it warm.
- **Parallel:** two pictures at once, from two processes or two threads of one process, gave no gain.  Each took
  twice as long; the NPU already runs one picture on all of itself.

So on screenshots the NPU is about as fast as the CPU reader with 12 threads and 25-30 times slower than an RTX
3090, but it takes no VRAM, almost no CPU, and the least energy per picture of the three.  It suits pictures read
in the background, or a PC where the GPU's memory is worth more to the model than picture speed.

## Using it as the server's picture reader (opt-in)

The server starts whatever one exe its `vision` section names.  To read pictures on the NPU:

1. Set `STRATA_VISION_NPU_MODEL_IR`, or `STRATA_VISION_NPU_ANYSIZE` for an any-size export (and, if you like,
   `STRATA_VISION_NPU_DEVICE`), in the environment that starts the server.  The server passes the vision section's
   `max_tokens` to the reader; with `--anysize` it must not be above the export's largest budget, or the reader
   answers one `ERR` line naming both.
2. In the run config, point `vision.exe` at `tools/vision/strata_vision_npu.py` (Linux, Mac) or
   `tools/vision/strata-vision-npu.cmd` (Windows), and keep `vision.cpu_exe` at the normal `strata-vision`.

When the NPU, the driver or the IR is not there, the reader's `ERR` line makes the server print one line and fall
back to the CPU reader - which needs `vision.cpu_exe` to name it, the way the AMD and Mac GPU readers do; the
custom run script names it by itself when a `strata-vision` binary sits next to the engine (`strata_control/
vision_reader.py`, `serve/server.py`, `Vision._start`).  The engine side needs nothing: it reads the `SVE1`
files exactly as it does the CPU reader's.

## Running it inside a container (podman or docker)

The reader also runs from an OCI container.  The parts that cannot be containerised - the
`intel_vpu` kernel module and the firmware - stay on the host; the container gets the device node plus the
userspace driver copied from the host, so its versions always match the running kernel module.
`container/npu-vision/` holds the image recipe: `python:3.12-slim`, the pinned `openvino==2026.4.1` wheel with
NumPy and Pillow, the reader and the exporter, and three libraries copied in: Ubuntu's `libze1` Level Zero
loader (with its layers), Intel's `intel-level-zero-npu` driver and the `intel-driver-compiler-npu` compiler.
Without the Level Zero loader the NPU simply is not enumerated and OpenVINO reports a misleading
"VCL compiler loading failed" - it is not an OpenVINO file that is missing, it is the loader that finds the
driver.

Checked 2026-10-23 on the same Core Ultra 7 265K (rootful podman, `--device /dev/accel/accel0`): the chain of
two IRs inside the container matched bare metal exactly - mean cosine 0.998390, min 0.967598 against the same
`strata-vision` reference and test picture, warm encode 2.3 s, `OK 576 24 24 2231` for the 768x768 photo.  The
startup compile is paid inside the container too, on the host CPU: 167 s + 186 s with one core at 100% (the NPU
compiler is nearly single-threaded), while waiting on the NPU during encodes costs about no CPU at all.  Two
compiles at once (one in a container, one outside) made each far slower than either alone - the first thing to
rule out when a start looks hung.

Rootless podman hits a permissions wall: `/dev/accel/accel0` is `root:render`, inside the user namespace the
node shows as `nobody:nogroup` and opening it fails (checked); `--group-add keep-groups` alone does not fix it
(checked).  Rootful works (checked), and a rootless container still runs the reader on the CPU
(`--device CPU`, `READY 2560`, checked).  The documented rootless route - the render GID in `/etc/subgid` plus
`--gidmapping` - was not checked here.  The commands, the cache volumes (a volume on `/root/.cache` holds the
driver's one-graph cache, and pointing `STRATA_VISION_NPU_CACHE_DIR` at a directory on a volume holds OpenVINO's
per-model cache - the one that makes a fresh container start in seconds), and a shim that lets the
container stand in as the server's `vision.exe` are in `container/npu-vision/README.md`.  A real `docker`
daemon was not tested: the `docker` command on this machine is podman's emulation.

## What was checked, and what was not

Checked 2026-10-22 on a Core Ultra 7 265K (Arrow Lake NPU, PCI `8086:ad1d`), Ubuntu 24.04.5, OpenVINO 2026.4.1,
with a small test encoder exported to IR (mean-pool + projection, 64x64 in, 4 tokens x 16 wide out):

- the whole resident cycle on the real NPU - `READY`, an `ENC` picture, byte-exact `SVE1`, clean `QUIT`;
- the reader's numbers against the same preprocessing computed in NumPy: cosine 1.000000 on every token row on
  NPU, GPU and CPU; the NPU's worst absolute difference 5.0e-04 against FP32 (the NPU computes in FP16), CPU 0.0;
- the honest failures: no model, an unreadable IR and an unknown `--device` each answer one `ERR` line with exit
  3 or 1, so the server's CPU fallback takes over.

Checked 2026-10-23 on the same Core Ultra 7 265K (Intel AI Boost NPU, `intel_vpu` driver, kernel 6.8.0-147,
OpenVINO 2026.4.1, Ubuntu 24.04.5), this time with the real encoder exported from the real mmproj.  The test
picture is a 768x768 photo; the reference is the `strata-vision` engine reader (SVE1, same day); cosines are
per token row:

- the exporter's IR on CPU: f32 0.999967 mean / 0.999347 min, fp16 0.999967 / 0.999428, about 3.2 s a picture;
- the fp16 IR on the machine's Intel iGPU (`GPU.0`): 0.999808 / 0.996600, 3.4 s;
- OpenVINO's NVIDIA plugin does not compile this IR on the RTX 3090 or the CMP 170HX - no consequence, those
  cards read pictures through the engine's CUDA path, not through OpenVINO;
- the NPU and the whole 27-block graph: compile succeeds, the first inference hangs the device
  (`ZE_RESULT_ERROR_DEVICE_LOST`) on f32 and fp16 alike; the device recovers and the tiny-model health check
  then passes.  The ceiling is per compiled graph and sits between 805 MB and 896 MB: 4 blocks run in 0.43 s,
  12 in 1.00 s, 20 in 1.61 s, 24 (805 MB) in 1.91 s, all computed correctly on the NPU, and two graphs
  totalling 1.17 GB are fine inside one process;
- the NPU's fp16 LayerNorm: fed at std 0.8 a merger matches CPU at 0.999948, then 0.999625 (std 3), 0.999372
  (std 13), 0.958104 (std 53), 0.723328 (std 105 - the output collapses to its constant part) and NaN (std 210).
  A max-scaled LayerNorm answers 1.000000 against CPU at std 0.8 and at std 105 while the plain one collapses on
  the same input;
- the recommended export (`--fp16 --split-after 13 --ln-scale max`), chained, on the NPU: **0.998390 mean /
  0.967598 min, 2.16 s a picture**; the same chain on CPU says 0.999962 / 0.999452, so the max-scaling left the
  function where it was; the reader's resident protocol on the NPU (READY, ENC, SVE1 file, QUIT) reproduces the
  raw-OpenVINO numbers exactly; a uniform white picture, the degenerate case for max-scaling, holds at
  0.998089 / 0.970755 against CPU, all finite.

The two NPU findings are facts about this driver build, not promises about others; both are worth reporting to
OpenVINO upstream.  They cost real work to find because the driver fails quietly: the hang is at least loud,
while the LayerNorm collapse returns plausible numbers.

A full server run on this machine answers pictures through this reader (2026-10-23): the server started it as
its `vision.exe`, a 512x512 test picture (left half red, right half blue) encoded to 576 embeddings - 637 of
the 642 prompt tokens - and the model answered "Red on the left, blue on the right."; the same picture sent
again round-tripped in 1.9 s through the server's picture-hash cache.  While resident the reader is the process
holding `/dev/accel/accel0` (`fuser /dev/accel/accel0` shows it).

Checked 2026-10-24 to 2026-10-27 on Windows 11 Pro 10.0.26300: an Intel Core Ultra 9 285K (Arrow Lake NPU, NPU
3720, Intel NPU driver 32.0.100.5540) with an RTX 3090, OpenVINO 2026.4.1, run as a standard user account:

- the recommended export on the NPU: 0.997881 mean against `strata-vision` on 7 pictures (4032 rows), 1.85 s a
  picture, about 0.02 CPU cores busy;
- neither Linux driver finding shows on this driver: the whole 27-block graph compiled (113 s) and ran (1.83 s),
  and a split export without `--ln-scale max` stayed finite and matched (0.997897);
- `strata-vision-npu.cmd` fails for an account with no `py` or `python` on its PATH (exit 9009, no `ERR` line),
  and with one there it would pick that Python, not the venv with OpenVINO;
- the any-size export and reader of the section above, with the numbers given there.

Not tested: the `.cmd` launcher's fallback path in the server; other NPU generations and driver versions;
`strata launch --vision` - no CLI flag asks for this reader yet, and no default changed.

The tests run on every PC, no driver and no hardware:

```
python -m unittest tools.test_strata_vision_npu      # the reader: protocol, SVE1 bytes, grid, errors, normalization
python -m unittest tools.test_export_npu_encoder     # the exporter: token order, rope tables, folded weights, refusals
python -m unittest tools.test_npu_anysize            # any size: mtmd sizing, tables, padding; with OpenVINO, the graphs
```

They cover the protocol, the `SVE1` bytes, the grid order, the error lines, the normalization triples, the
server's `Vision` class driving this exe, and the rule that the mock only arrives when `--mock` or
`STRATA_VISION_NPU_MOCK=1` asked for it.  The exporter tests write a tiny synthetic mmproj GGUF, so no model
file is read.
