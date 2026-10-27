# This fork: Strata for coding agents that run heavy jobs on the same PC

**Works on Windows.** This fork is developed and used every day on **Windows 11, natively** (no WSL, no Docker)
with an NVIDIA card: an RTX 3090 24 GB, a Core Ultra 9 285K and 95 GB of RAM, running Qwen3.8-Flash-Next IQ3_S at a
262K context. Everything below was measured there. Linux should work for most of it, but only one piece has been
tested on Linux so far (see "Tested").

It is an **experimental fork** of [Niko1221/Strata](https://github.com/Niko1221/Strata) at **0.1.44.1**, with the
persistent conversation disk tier of upstream PR [#1937](https://github.com/Niko1221/Strata/pull/1937) by xrip, and
the changes below on top. Upstream has moved on to 0.2.x since; for everything this fork does not add, use upstream.
The lock part is offered upstream in [#2714](https://github.com/Niko1221/Strata/issues/2714).

## What it is for

You run coding agents (opencode, Claude Code) on a local model, and the agents also start heavy jobs - a training
run, a simulation, an ML pipeline - that need the same GPU and RAM the model holds. With this fork:

- **The model gets out of the way for the job.** The job runs inside a wrapper that takes a *lease* on Strata: the
  model unloads (all of it, or only its RAM) and nothing can load it back while the job runs - not even another
  agent's request. Those requests wait instead of failing.
- **It comes back by itself, with the conversations.** When the last job ends, the model loads again (if it was
  loaded before), and the long agent conversations come back from the disk instead of being read again from token 0.
- **Nothing is left in a half state after a crash.** The wrapper renews its lease every 10 s; if it dies (killed,
  crash, power cut), the job dies with it and the lease expires 45 s later. At the next start, Strata tells the agent
  session whose job was cut off, and, after an unclean shutdown, the sessions that had a request in flight.

## What it adds

| Feature | In short | Docs |
|---|---|---|
| Persistent disk tier (#1937) | Parked conversations go to a disk folder and survive engine restarts | [LOCAL_PERFORMANCE.md](docs/LOCAL_PERFORMANCE.md) §7-8, [DETAILS.md](docs/DETAILS.md) (`--conversation-disk-*`) |
| Hibernate / resume | `POST /unload {"save": true}` saves the conversations and frees RAM and VRAM; `{"keep_gpu": true}` frees only the RAM (Windows); a request or `POST /load` brings it back | [LOCAL_PERFORMANCE.md](docs/LOCAL_PERFORMANCE.md) "Freeing the PC" and §4-5, [DETAILS.md](docs/DETAILS.md) "Sharing the GPU with other programs" |
| The lease lock + suspend wrapper | `python tools/strata_suspend.py run -- <your job>`; `status`, `stop --id`, `jobs`, `launch`, `hold --pid` | [LOCAL_JOBS.md](docs/LOCAL_JOBS.md), [LOCAL_PERFORMANCE.md](docs/LOCAL_PERFORMANCE.md) "Long jobs" |
| Waking cut-off sessions | At start, tells the opencode or Claude Code session whose job (or request) was cut off, with the job's log tail | [LOCAL_JOBS.md](docs/LOCAL_JOBS.md) |
| save_on_close (Windows) | Closing the server window saves the conversations first | [LOCAL_PERFORMANCE.md](docs/LOCAL_PERFORMANCE.md) §9 |
| Self-restart | `tools/strata_service.py` runs the server and restarts it on `POST /restart` (deploys without admin rights) | the module's docstring |
| Model-only API keys | `"api_key_file"`: `{"full": [...], "model": [...]}` - a model-only key can chat and take leases, nothing else | `serve/server.py` `api_keys_from_file` |
| Web chat additions | Attach PDF/Office files (`POST /convert`, MarkItDown), detached runs, queued messages, Compact, subagents, a chat store shared by every device | [DETAILS.md](docs/DETAILS.md) "PDFs and Office files in the chat page"; the rest is not documented yet |
| opencode plugin + commands | Continues turns that ended empty; `/strata-free-all`, `/strata-free-ram`, `/strata-resume`, `/strata-status` | [LOCAL_PERFORMANCE.md](docs/LOCAL_PERFORMANCE.md) §10 |
| Faster engine start (Windows) | `STRATA_EARLY_ARENA=1` (opt-in), plus prefaulting and threaded dense uploads (on by default); `STRATA_POOL_PIN=hard` | [LOCAL_PERFORMANCE.md](docs/LOCAL_PERFORMANCE.md) §1-2; the switches are described in the code comments |
| Pictures on an Intel NPU (opt-in) | sibnick's NPU picture reader (OpenVINO), plus pictures at any size (`--anysize`); the vision section's `"npu"` turns it on, and a PC without an NPU falls back to the normal reader | [INTEL_NPU.md](docs/INTEL_NPU.md), "In this fork" |

## Install (Windows)

1. Clone this repository: `git clone https://github.com/page1/Strata.git` and `cd Strata`.
2. Build the engine from this source: `python setup.py --build`. **Do not skip `--build`**: by default setup downloads
   upstream's ready-made 0.1.44.1 engine, which does not have this fork's engine changes (the disk tier, hibernate).
   setup installs what the build needs (the CUDA toolkit and Visual Studio's C++ tools) and picks the model for your
   RAM, as upstream's installer does.
3. In the run config setup wrote (`strata-<model>.json` in the Strata folder), turn the fork's parts on:

```json
"hibernate_dir": "C:\\Strata-data\\hibernate",
"wake_cut_off_jobs": true,
"save_on_close": true,
"args": [..., "--conversation-disk-dir", "C:\\Strata-data\\conversations",
         "--conversation-disk-persist", "on"]
```

   Optional: `"leases_dir"` (default: `leases` inside the hibernate folder), `"lease_expire_s"` (default 45) and
   `"lease_resume_wait_s"` (default 300: how long a waiting request also waits for the model to come back after a job).
4. Start Strata as upstream's README says, then run a heavy job through the wrapper:

```
python tools\strata_suspend.py run -- python train.py
python tools\strata_suspend.py status      # what holds Strata
python tools\strata_suspend.py stop --id <lease id>
```

For opencode, copy `tools/opencode-local` into `%USERPROFILE%\.config\opencode\` and edit the Python and Strata
paths in `command/strata-*.md` to your Strata folder.

### Optional: read pictures on an Intel NPU

Only for a Core Ultra PC with an NPU (Intel AI Boost) and its driver. Off unless you set it; without an NPU the
setting falls back to the normal picture reader.

1. In the Strata folder: `python -m venv .venv-npu`, then `.venv-npu\Scripts\python -m pip install -r requirements-npu.txt`.
2. Export the picture encoder once: `.venv-npu\Scripts\python tools\vision\npu_anysize.py export <your mmproj .gguf>
   --out-dir C:\Strata-data\npu-vision\anysize`.
3. Add `"npu": {"anysize": "C:\\Strata-data\\npu-vision\\anysize", "cache_dir": "C:\\Strata-data\\npu-vision\\cache"}`
   to the run config's `"vision"` section, and restart Strata. The first start compiles for about 2 minutes; later
   starts take about 5 s.

What to expect, measured on the PC above: a full-size app screenshot (about 2000 tokens) takes about 25 s on the NPU,
the same as the CPU reader with 12 threads, against under 1 s for the RTX 3090's reader. The NPU uses no VRAM,
almost no CPU, and about half the energy per picture of the GPU. It is for pictures read in the background, or when
the GPU's memory matters more than picture speed. The full numbers are in [INTEL_NPU.md](docs/INTEL_NPU.md).

## Teach your agents

Your agents won't use the wrapper unless you tell them to: without it, an agent just runs `python train.py` into
a GPU the model still holds. Paste the block from [docs/AGENT_INSTRUCTIONS.md](docs/AGENT_INSTRUCTIONS.md) into
your `CLAUDE.md` (Claude Code) or `AGENTS.md` (opencode), with your Strata folder in the paths. It tells the agent
when to wrap a job, the exact command, to give it a long timeout, not to rerun a job the user stopped, and what
the wake-up note after a crash means.

## Windows notes

- Native Windows throughout: the build (CUDA + MSVC via `setup.py --build`), the server, the wrapper.
- Windows-only pieces: freeing only the RAM (`keep_gpu`, which needs the expert arena in 4 KB pages), save_on_close,
  the CPU Sets pinning, the faster start (`STRATA_EARLY_ARENA`).
- A wrapped job lives and dies with its wrapper through a kill-on-close job object; on Linux a small reaper process
  does the same.
- `taskkill` is refused in an ssh session even on your own processes; the wrapper stops jobs through psutil instead.

## Tested

- **Windows 11 + RTX 3090** (the PC above): the server and tool test suites, a verdict test per change, and live
  tests against the running server - a job's lease unloads the model, a request sent during it waits and is
  answered after it, overlapping jobs, a killed wrapper (its job gone in 0.5 s, the lease expired after ~44 s), a
  server restart during a job, a forced load, three jobs started together.
- **Intel NPU reader (Windows 11, the PC above, NPU 3720, driver 32.0.100.5540, OpenVINO 2026.4.1)**: sibnick's
  reader and test suites, the any-size export and reader on the NPU (10 screenshots, against the CPU and GPU readers,
  with energy), and the vision section's `"npu"` and its fallback with unit tests. Not tested: the `"npu"` setting
  in a live server start (the any-size reader ran live on the NPU through the reader protocol, and sibnick's
  fixed-size reader answered pictures as a running server's `vision.exe`), other NPUs and drivers, Linux.
- **Linux**: only that a killed wrapper takes its job with it (Ubuntu under WSL).
- **Not tested**: other GPUs, AMD (HIP), Intel (SYCL), multi-GPU, macOS.

## Fixed before this release

- The MCP server's `strata_stop` now ends the real server on Windows. The venv's `python.exe` is a launcher that
  starts the real interpreter as a child; stop used to end only the launcher, so the server could keep running.
- The test suites leave the PC alone: they no longer read the real Claude transcripts or opencode database, write
  the real heartbeat file, or talk to a Strata already running on port 8080.
- Flaky tests: each one found was traced to its cause and fixed. The hold suites passed 16 of 16 runs, two at a
  time in parallel; the MCP suite 5 of 5 and its stop test 15 of 15 under that load.
- A request that waited out a job's lease could still get a 503 when the model's return after the job had to wait
  for the job's memory to show as free (seen once in a live test; the model came back by itself 5 s later). Such a
  request now waits for that too, for up to `"lease_resume_wait_s"` (default 300 s).

## Known issues

None open. (An earlier note here said `strata_start` lost track of the server if the venv launcher ended during a
start. Measured since: the launcher keeps its child in a kill-on-close job, so the server ends with it and nothing
is left on the port; a test now checks that.)

## Credits

The disk tier is xrip's work (upstream #1937). The Intel NPU picture reader, its exporter and its container are
sibnick's (`npu-vision-reader`, offered upstream in #678); this fork adds the any-size export and the `"npu"`
setting. Fixes from review by midhatn and K8nsch. Strata itself is
Niko1221's and its contributors' (MIT license, unchanged). The fork's changes were written with Claude Code and
local agents, and reviewed and tested by its author.
