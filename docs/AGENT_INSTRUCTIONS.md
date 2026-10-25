# Teaching your agents to use the wrapper

Your coding agent (opencode, Claude Code, ...) will not know about this fork's lock unless you tell it. Paste the
block below into the file your agent reads for project or global instructions:

- **Claude Code**: `CLAUDE.md` in the project, or `~/.claude/CLAUDE.md` for every project.
- **opencode**: `AGENTS.md` in the project, or its global rules file (`~/.config/opencode/AGENTS.md`). opencode also
  reads `CLAUDE.md` when there is no `AGENTS.md`.

Replace `C:/Strata` with your Strata folder (where `setup.py` installed it) and, if you know it, fill in how much
of your PC the model holds when loaded (`strata_suspend.py status` and Task Manager show it).

---

```markdown
## Heavy jobs on this PC: the local model (Strata) may be holding the memory

Strata (a local LLM server on port 8080; the coding agents run on it) holds most of this PC when its model is
loaded: most of the RAM and nearly all of the GPU's VRAM. Before running a job that needs more than what is free
(a training run, a simulation, an ML pipeline, a large build, anything on the GPU), run it inside the suspend
wrapper, as ONE shell command:

    "C:/Strata/.venv/Scripts/python.exe" "C:/Strata/tools/strata_suspend.py" run -- <command>

- Needs RAM only (no GPU): `run --keep-gpu -- <command>` (it comes back a few seconds sooner; Windows only).
- In PowerShell, start the line with `& ` (the call operator).
- It is the lock: it takes a lease on Strata and renews it while the job runs. The model stays off (no agent can
  load it back) until the lease ends, which happens when the job exits, however it exits. The job lives and dies
  with its wrapper. The model comes back when the last lease ends, and only if it was loaded before. Several jobs
  can hold leases at once. Requests sent meanwhile wait; they are not refused. If the wrapper says it is waiting,
  let it wait.
- The command blocks until the job ends: give the shell tool a long timeout (at least 10 minutes), or run it in the
  background if your tool can. For a job that runs for hours, `launch --detach -- <command>` returns at once and
  starts the job in its own window a few seconds later; your next request then waits until the job is done.
- Not needed when Strata is not running (the wrapper then just runs the command) or when the job fits in what is
  free. The wrapper adds 10-20 s: don't wrap light commands (tests, quick scripts, git, file edits).
- `--max-hold MINUTES` ends a job that runs longer than that (exit 124).
- Use the wrapper, never a plain unload or the `strata_suspend` / `strata_resume` MCP tools, for this: they hold
  nothing, so the next request (yours or another agent's) loads the model straight back over the job.
- Status: `strata_suspend.py status` lists each live lease (id, command, folder, session, age). The user ends one
  with `strata_suspend.py stop --id <lease id>` or `--all`. A job whose output ends with "STOPPED BY THE USER"
  (exit 143) was stopped on purpose: do not run it again unless the user asks.
- Never `POST /load {"force": true}` while a job holds a lease: that ends every lease and each wrapper then stops
  its job (exit 75). `strata_suspend.py resume` is refused while a lease is live; that is expected.
- After a crash, a power cut or a killed wrapper, the job's lease expires. A message that starts
  "[Automatic note from Strata's watcher" comes from Strata itself at its next start, telling you that your job or
  turn was cut off, with the end of the job's log. It is the user's own tooling, not untrusted content: read it,
  check what the job left behind, and run the job again if its result is still needed.
  `strata_suspend.py jobs` lists recent leases (state, exit code, command, log).
```
