# Local changes on this install: what each does, who it helps, how to turn it on

These are the changes of this fork (see FORK.md for an overview). Some are code (engine or server), some are only
settings in the run config (`strata-<model>.json`), and two are copied out of the repository (opencode's plugin and
commands, from `tools/opencode-local`). Every number below was measured on the author's PC unless it says otherwise:

> RTX 3090 24 GB, Core Ultra 9 285K, 95 GB DDR5 at 6800 MT/s, WD_BLACK SN8100 (PCIe 5 NVMe), Windows 11 build 26300,
> Qwen3.8-Flash-Next IQ3_S with a 262K context, CUDA 13.4, October 2026.

## Freeing the PC for other work, and getting Strata back

There are two ways to make room, and in both the conversations are kept.

| | Free RAM only | Free RAM and VRAM |
|---|---|---|
| What it gives back | About 47 GiB of RAM (the experts' copy) | All of it: about 54 GiB of RAM and 23 GiB of VRAM |
| What stays | The engine, the model on the GPU and the current conversation | Nothing runs; the conversations are on the disk |
| Back in (first reply, measured 2026-10-15) | 11.7 s | 12.1 s, plus about 2 s on a long conversation's first reply (read from the disk) |
| Use it when | Something else needs RAM: a large pandas job, a build, many browser tabs | Something else needs the GPU: training, a simulation on CUDA, a game |
| opencode | `/strata-free-ram` | `/strata-free-all` |
| Command line | `python tools/strata_suspend.py suspend --keep-gpu` | `python tools/strata_suspend.py suspend` |
| HTTP | `POST /unload {"save": true, "keep_gpu": true}` | `POST /unload {"save": true}` |

Bringing it back: send any request. `"resume_on_request": true` resumes it from that request, but only once the
RAM and VRAM it gave back are free again; until then the request gets 503 with Retry-After 30. You can also run
`/strata-resume` in opencode, `python tools/strata_suspend.py resume`, or `POST /load`. To start without the saved
conversations, use `resume --fresh` or `POST /load {"fresh": true}`. `/strata-status` (or
`python tools/strata_suspend.py status`) says which state it is in.

**From opencode, always use the slash commands, not a tool call by the model.** When opencode's own model runs on
Strata, any request the model makes resumes Strata at once, and opencode sends one right after every tool call to
report its result. The commands run the script before the model sees anything, with `--detach --delay 8`. The
script returns immediately, the model repeats one line, and 8 s later a background process suspends Strata (retrying
while it is still busy, for up to 120 s). The log is in `data/strata-suspend-detached.log`. The commands are
`~/.config/opencode/command/strata-*.md`, so they appear in the `/` menu of every project.

### Automatically, when idle

```json
"idle_unload_s": 300
```

- After 5 minutes without requests, Strata unloads: RAM and VRAM are both freed.
- With the persistent disk tier (7), the engine writes its conversations first.
- The next request loads it again. With `"memory_check_on_load": true`, a load waits for free memory instead of
  pushing a running job aside: the request gets 503 with Retry-After meanwhile.

`"idle_release_s": N` (local) is the RAM-only variant: after N seconds idle, the RAM goes back and the GPU keeps the
model. Both can be set; for example, a release at 300 and a full unload at 1800. It is **not used here**, because
measured on this PC the two stages cost the same:

| State | GPU power at idle | VRAM | First reply after the next request |
|---|---|---|---|
| Loaded, idle | 19.7 W (P8) | 23.3 GB | none |
| RAM only | 19.8 W (P8) | 23.3 GB | 11.7 s |
| RAM and VRAM freed | 20.1 W (P8) | 0.07 GB | 12.1 s |

- Filled VRAM costs no power: the 3090 idles in P8 at about 20 W whatever it holds.
- Both resumes are dominated by reading the 47 GiB of experts back.
- After a full unload, a long conversation also comes back from the disk on its first reply: about 1.5-2.3 s at
  140K tokens. After a RAM-only release it is still in memory.
- These are single measurements (October 2026).

### Long jobs: keep Strata away until the job is done, however it ends

A plain suspend comes back on the first request. Any opencode window or agent can send one, so it is the wrong tool
for a job that runs for hours. For that, the job takes a **lease** on Strata: the wrapper asks the server for it,
renews it every 10 s while the job runs, and ends it when the job exits.

| You want | Command (add `--keep-gpu` to free RAM only) |
|---|---|
| Run the job here and wait for it | `python tools/strata_suspend.py run -- python my_job.py` |
| Start the job in its own window and return | `python tools/strata_suspend.py launch -- python my_job.py` |
| The same, from an opencode agent running on Strata | `python tools/strata_suspend.py launch --detach -- python my_job.py` |
| Take a lease for a job that is already running (it ends when that pid exits) | `python tools/strata_suspend.py hold --pid 12345` |
| End a job's lease (the job stops; the model comes back) | `python tools/strata_suspend.py stop --id LEASE_ID` (or `--all`) |

When Strata is busy (a request runs or waits), `run` retries for up to `--wait` seconds (default 120); behind another
job's RAM-only suspend it waits until that job ends.

What a lease does:

- **While a lease is live**, requests wait instead of getting 503, whichever opencode window or agent sends them,
  and even with `resume_on_request` on. A streamed request gets an SSE keep-alive comment every few seconds while it
  waits; one that is not streamed waits silently. A plain `POST /load` is refused with 409; only
  `POST /load {"force": true}` overrides it. That ends every live lease as "overridden", and each wrapper then stops
  its job (exit 75). `resume` is a plain `POST /load`, so it is refused too while a job holds the model.
- **If the wrapper dies** (killed, a crash, a power cut), the job goes with it (on Windows a kill-on-close job object,
  elsewhere a reaper process), its renewals stop, and the server expires the lease 45 s after the last renewal; the
  model then comes back.
- **The three rules:**
  1. Strata loads a model only when no lease is live. A lease is live when it has not ended and was renewed in the
     last 45 s. After a server start, a lease that was live gets 45 s from the start to renew before it can expire,
     so a restart does not stop running jobs.
  2. The model comes back when the last live lease ends, but only if it was loaded when the first lease of that
     period was made. A model that was unloaded before stays unloaded.
  3. At the next Strata start, the agent sessions told are: the session named in each expired lease (it is recorded
     when the lease is made), and, after an unclean shutdown only, the sessions that had a request in flight, or had
     just finished one (within 3 minutes), at the last checkpoint (written when a lease is made or a save runs). A stopped lease never wakes its session. A clean
     stop (window closed, Ctrl+C, `/restart`) wakes nobody for requests in flight.
- **The leases are files**, one per lease, in `leases` inside the hibernate folder (or `leases_dir` if set in the run
  config), so they survive a server restart. Only the server writes them.
- **Several jobs can hold at once.** Strata comes back when the last live lease ends.
- `status` (and `/leases`) lists the live leases. `/health` shows them as `held_by`: each with its id, `cmd`, `cwd`,
  `session`, `mode` and `age_s`.
- Over HTTP:
  - `POST /lease {"cmd", "cwd", "mode": "full" | "ram", "session", "log"}` makes a lease and suspends the model
    for it; the answer has its `id`;
  - `POST /lease/<id>/renew` (410 once the lease has ended), `POST /lease/<id>/end {"exit": code}`,
    `POST /lease/<id>/stop`, `POST /lease/stop {"all": true}`, `GET /leases`;
  - `POST /unload {"save": true, "hold_pids": ...}` is refused with 400, and `POST /hold` answers 410.

`launch --detach` is the version for agents: it returns at once, and 8 s later a background process starts the job,
which takes its lease and opens the job's window. The agent's reply finishes first, and its next request then waits
until the job is done.

With `--keep-gpu`, if Strata's expert arena uses large pages and cannot give its RAM back alone, the wrapper falls
back to a full suspend and prints "freeing RAM and VRAM instead".

## The changes

### 1. Faster engine start (engine; always on)

- Two changes:
  - The dense projections are uploaded by 8 threads, using ReadFile into pinned staging memory: 2.2 s, down to 0.4 s.
  - The expert arena's pages are touched by 16 threads before `cudaHostRegister` pins them: pinning 4.98 s, down to
    0.55 s.
- Result: the engine is ready in 11.6 s instead of 18.9 s, and a resume takes 11.5 s instead of 21 s. Decode and
  prompt speed are unchanged, and the weights on the GPU are checksum-identical.
- Who it helps: anyone who restarts, hibernates or reloads often.

### 2. CPU worker pinning with CPU Sets (engine; Windows; always on)

- On this 285K under Windows 11 26300, pinning a thread to particular cores with a hard affinity could take seconds,
  and it blocked every other thread creation in the process. The result was random 0.5-7 s stalls at start.
- CPU Sets give the same placement without the stall.
- Who it helps: Windows on hybrid CPUs (P-cores and E-cores).
- `STRATA_POOL_PIN=hard` restores the old behaviour.

### 3. Learned expert profile (setting)

```json
"expert_profile_save": "C:\\Strata-data\\expert-profile-learned.bin",
"expert_profile_save_every": 0
```

- The engine saves which experts it actually used, and the next start loads those onto the GPU first.
- The first reply after a start ran at 106 tok/s instead of 95 (+11%), with PCIe misses down from 9.8% to 6.5%.
  Later replies are the same as before.
- `expert_profile_save_every: 0` saves only when the engine stops, with no periodic writes.
- Who it helps: anyone who starts or resumes the model often; the gain is in the first few minutes.

### 4. Hibernate, resume on request, and the memory check (server; settings)

```json
"hibernate_dir": "C:\\Strata-data\\hibernate-iq3_s",
"resume_on_request": true,
"memory_check_on_load": true
```

- `/unload {"save": true}` saves the conversations and keeps the model unloaded: requests get 503 instead of loading
  it back over whatever is now using the GPU.
- With `resume_on_request`, a request resumes it instead, but only once the freed RAM and VRAM are free again. The
  margins are `resume_margin_mib` (default 4096) and `resume_wait_s`.
- `memory_check_on_load` applies the same check to a plain reload.
- Measured: free RAM went from 20 GB to 74 GB, and VRAM fell to 282 MiB.
- Who it helps: anyone who shares the PC between Strata and other heavy work.

### 5. Free RAM only, keep the GPU (server and engine)

- `POST /unload {"save": true, "keep_gpu": true}` (RELEASE / ACQUIRE in the engine) gives back about 47 GiB of RAM
  and keeps the model and its expert cache on the GPU.
- When it was built it resumed in about 7 s, against 15-17 s for the full hibernate (13 of 13 end-to-end tests passed). After the faster start (1), the two measure the same: 11.7 s against 12.1 s on 2026-10-15.
- Who it helps: anyone who needs RAM but not the GPU.

### 6. Per-conversation expert lists: `--conv-experts 1000` (engine; setting)

- When two conversations take turns (for example code and prose, or two agents), each one pulls the other's experts
  out of the GPU cache.
- With this, each parked conversation keeps its own routing counts and restores them when it comes back. Decode
  speed, weighted by tokens:

  | Two conversations | Decode |
  |---|---|
  | Each run as one block | 93.5 tok/s |
  | Taking turns | 83.1 tok/s |
  | Taking turns, with `--conv-experts 1000` | 86.1 tok/s |

- Who it helps: anyone running several agents or conversations at once.

### 7. Parked conversations on the disk, kept across restarts (PR #1937 plus local changes; engine; settings)

```json
"--conversation-disk-dir", "C:\\Strata-data\\conversations",
"--conversation-disk-mib", "65536",
"--conversation-disk-persist",
"--conversation-disk-max-age-days", "7"
```

- Strata keeps recent conversations parked in RAM, `--conversation-cache-mib 8192` here. That is about two
  conversations of 140K tokens (2.8-3 GB each).
- Without the disk tier, a conversation pushed out of RAM was lost, and continuing it meant reading the whole prompt
  again.
- Now it is written to the disk in the background and read back when a request continues it. Measured:

  | Situation | Time |
  |---|---|
  | Continuing an evicted 136K-token conversation, cold | 60.6 s |
  | The same, from the disk (2.2 GB at about 1.5 GB/s) | **2.3 s** |
  | Continuing a 138K-token conversation after the engine restarted, including the engine start | **17.2 s** |

- The parts of this:
  - `--conversation-disk-persist` keeps the files across restarts.
  - `--conversation-disk-mib` caps them, removing the oldest first.
  - `--conversation-disk-max-age-days` (local) also removes anything written more than N days ago, when the engine
    starts and while it runs.
- The files are bound to the model and settings, so a different model or setting never reads them.
- Who it helps: anyone running long agent sessions, several conversations at once, or stopping Strata between
  sessions.
- Measured and not changed: a 16 GiB RAM cache made the per-turn park reuse its memory on only 1 of 7 turns (about
  0.8 s per turn at 147K tokens either way), so the cache stays at 8 GiB and the disk takes the overflow.

### 8. Hibernate through the disk tier (server; local; automatic when 7 is on)

- With `--conversation-disk-persist`, a full hibernate no longer writes its own copy:
  - the engine quits and writes only what is not on the disk yet, about 9.7 s here;
  - the server writes a one-line latch.
- A resume is a plain start, and each conversation comes back when a request continues it. Nothing is read that
  nobody asks for.
- A hibernate folder written the old way still restores the old way.
- Asking to hibernate the instant a reply ends can answer 409 "busy". `strata_suspend.py` retries this.

### 9. Saving when the window is closed (server; Windows; setting)

```json
"save_on_close": true
```

- Closing the Strata window gives a program about 5 s before Windows ends it, which is too short to write gigabytes.
- With this setting, the engine runs without the console, so the close does not end it. The server lets the engine
  outlive it and closes the engine's input; the engine writes its conversations to the disk and exits on its own.
- Ctrl+C saves first too. A second Ctrl+C ends it at once, without saving.
- Verified on this PC by closing the window: the conversation was written, 11 were kept, and the engine ended by
  itself.
- It needs 7 (persist). Without it, the setting is switched off with a message.
- In Windows Terminal the window's X cannot be removed, so this is what makes that X safe.

### 10. Empty turns and early stops in agents (setting, plus an opencode plugin)

```json
"reasoning_close_retry": true
```

- On long agent sessions (above about 50K tokens of context), IQ3_S sometimes ended its turn inside its thinking.
  There was no text and no tool call, so opencode stopped. It happened right where the model meant to stop and
  report.
- With this setting, the server closes the thinking once and lets the model answer.
- Replaying 12 real stuck turns from a long opencode session, 4 tries each:
  - without the setting: 15 of 48 empty;
  - with it: 0 of 48, and 41 of 48 continued with a tool call.
- Who it helps: anyone running long agent sessions on a thinking model.
- A diagnostic line (`empty answer: ... parser=..., held=...`) logs any reply that still ends this way.

The opencode side is `~/.config/opencode/plugins/empty-stop-continuer.js`, with its logic in
`lib/empty-stop-nudge.mjs` and a test in `lib/empty-stop-nudge.test.mjs`. It sends one short nudge when a turn:

- is empty;
- announces a step and stops ("Running that:");
- stops with items still open on the session's todo list, during autonomous work.

It sends at most 2 nudges in a row. A "keep working through the list" instruction was also tried, and made no
difference in the same replay (37 of 48 continued), so it is not used.

### Installing the opencode pieces

`tools/opencode-local/` holds the copies kept in this repository. opencode reads them from
`%USERPROFILE%\.config\opencode\`. Copy them over after a change:

```powershell
Copy-Item -Recurse -Force tools\opencode-local\* $env:USERPROFILE\.config\opencode\
```

| File | What it is |
|---|---|
| `plugins/empty-stop-continuer.js` | The early-stop plugin (10) |
| `lib/empty-stop-nudge.mjs`, `lib/empty-stop-nudge.test.mjs` | The plugin's logic and its test: `node lib/empty-stop-nudge.test.mjs` |
| `command/strata-free-ram.md`, `strata-free-all.md`, `strata-resume.md`, `strata-status.md` | The `/strata-...` commands. They call this repository's `.venv` python by full path; edit it if the repository moves |

Restart opencode after copying, including the background `opencode serve` on port 4096, because plugins and commands
load at start.

### 11. Settings checked on this PC

| Setting | Result | Kept |
|---|---|---|
| `--ple-io direct` | Free RAM after load: 20.6 GB, against 1 GB with the automatic mmap choice | Yes |
| NTFS compression off for `C:\Strata-data` (C: is compressed drive-wide) | Load 87 s down to 21 s; a cold 48K prompt 73 s down to 20 s; reads 0.7 up to 9.8 GiB/s | Yes. A new top-level folder on C: compresses again; check with `compact /s:<dir>` |
| `STRATA_ROUTE_TAIL_SKIP` default (rank 7) | Decode 93-96 tok/s, against 78.9 with it off | The default |
| `--pool-workers 15` | Best of 4, 10, 12, 15, 16, 20 and 24 (24 puts a worker on the host core: 54 tok/s) | Yes |
| `--mmap-experts` | Windows can take the RAM back without writes, but decode is 38% slower | No; use 5 or 4 instead |
| `STRATA_DEFERRED_REGISTER=1` | Load 4.6 s faster, decode 1.3% slower | No |
| Reasoning effort none / low / medium / high | Equal accuracy over 33 checkable runs per level (logic puzzles, schedules, arithmetic, dates). "none" was the slowest on hard ones, because it reasons in the answer instead | low by default (opencode) |

Never calibrate or benchmark with the Strata window minimized. Windows' power throttling (EcoQoS, #691) slows it by
about 20%, although the server opts itself out.

### 12. Engine flags measured on this PC (2026-10-15)

RTX 3090 24 GB, Core Ultra 9 285K, 95 GB RAM, Windows 11, IQ3_S, `--kv int8`, 262K context. Each flag was tested on
a second port with 3 engine starts per value in ABBAAB order, so drift through the session cancels out. Each start
asked for 400 tokens after prompts of about 1.7K, 61K and 84K tokens of prose, plus an 8K code-editing prompt (copy
a function with comments) in the later runs. The numbers are medians of the 3 starts. Other processes used 4-10% of
the CPU throughout. Each A/B ran on top of the settings already chosen by the ones before it, so compare numbers
within a table, not across tables.

Decode speed, tok/s:

| Change | 1.7K | 7K edit | 60K | 84K | Prompt reading at 84K | Kept |
|---|---|---|---|---|---|---|
| `--prefill auto` to `auto:16384` | 99.3 to 99.4 | | 82.8 to 82.5 | 82.1 to 79.5 | 2206 to 2440 tok/s | Step to the next row |
| `--prefill auto:16384` to `auto:32768` | 99.5 to 99.9 | 85.1 to 84.7 | 82.1 to 81.8 | 81.7 to 83.2 | 2433 to 2671 tok/s | Yes (+10% prompt reading against `auto`) |
| `--spec 4` to `6` | 99.7 to 91.7 | | 83.7 to 81.9 | 83.2 to 77.6 | no change | No |
| `--spec 4` to `3` | 99.0 to 92.9 | | 81.7 to 85.2 | 78.9 to 81.6 | no change | Yes (long prompts are most of the time spent here) |
| `--lookup-chain 0` to `4` (with spec 3) | 92.7 to 90.0 | 83.1 to 83.9 | 86.0 to 86.2 | 79.4 to 80.1 | no change | No: within noise |
| `--kv int8` to `q4_0` | 99.9 to 87.6 | | 83.6 to 82.7 | 81.8 to 82.2 | 2199 to 2108 tok/s | No |
| `--kv-resident 32768` to `65536` | 93.7 to 89.3 | 82.5 to 80.0 | 85.6 to 87.9 | 79.1 to 83.5 | no change | Yes |

Prompt reading is about 59% of the time agents spend waiting here. Half of the agent turns run near 120K tokens of
context, and 63% are above 90K. Long turns also take longer, so they count for more of the waiting time than of
the turn count. That is why `--spec 3` and `--kv-resident 65536` were kept even though both are 3-5% slower on
short prompts. A larger `--kv-resident` keeps more of the conversation's KV on the GPU and leaves fewer expert
slots, which is why it helps long prompts and costs short ones. An earlier run (`--kv-resident` 16384 against
32768) used 20480 for the smaller value, the engine's minimum.

**Two agents at once.** `"parallel": 2` with `--batch-mtp`, against one request at a time (`--kv-resident 32768`,
`--spec 3`). Two ~100K-token conversations of different text were primed, and then each asked for 400 tokens. 4
engine starts per setting, in ABBA order:

| | One at a time | `"parallel": 2` |
|---|---|---|
| Both together, total | 70.3 tok/s | 55.8 tok/s |
| Both together, each request | 83.1 tok/s (the other waits) | 34.4 tok/s |
| One request alone | 83.2 tok/s | 69.8 tok/s |

On a 24 GB card the second slot is slower in every case, including when only one agent is working. Each slot takes
about 0.56 GiB from the GPU expert cache, so more experts come over PCIe. The +31-39% gain reported for batch slots
was measured on 48 GB cards. Leave `"parallel"` off here.

(The prose cases were meant to be 2K, 60K and 120K tokens; the tokenizer packs this text tighter than the script
assumed, so they were 1,661, 60,677 and 83,622 tokens. The code edit was 7,044.)

**v0.1.44.1 against v0.1.43, both with the local commits** (2026-10-15, same PC and config as above with
`--kv-resident 65536`, 3 + 3 engine starts in ABBAAB order, decode median tok/s, other CPU 5-13%):

| | 1.7K | 7K edit | 61K | 84K | Prompt reading at 84K | Engine to ready |
|---|---|---|---|---|---|---|
| v0.1.43 + local | 86.6 | 82.0 | 85.1 | 82.2 | 2682 tok/s | 11.6 s |
| v0.1.44.1 + local | 89.8 | 82.4 | 84.0 | 81.4 | 2679 tok/s | 12.2 s |

Short prompts are 3.7% faster (all three new runs above all three old ones); the rest is within the run-to-run
spread. The update brings upstream's fixes, not speed, on this setup.
