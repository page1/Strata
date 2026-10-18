# Strata is locked: what to do (local)

A job run through `tools/strata_suspend.py run` holds Strata: the model gives its memory to the job, and every
request waits until the job ends. Then the model comes back by itself. While that happens, the web page's
Monitor shows **Locked**, and `status` says so.

## See what holds it

```
.venv\Scripts\python.exe tools\strata_suspend.py status
```

For each job that holds Strata it prints the command, its folder, how long it has run, the agent session that
started it, and what to do. The Monitor's Locked card shows the same.

## See who is waiting

`status`, `/health` and the Monitor (the "Requests" list) show every request in Strata now, oldest first after
the one generating, for example:

```
generating - opencode session "Nightly review"
queued (2 min) - subagent general of "Nightly review" - task: Write a report
waiting out a hold (14 min) - opencode session "my-project: rankings"
queued - from web page
```

States: generating; queued (another request has the turn); waiting out a hold (a job holds Strata); loading the
model. A subagent is shown with its parent session and the task it was given; a request without a session (the
web page, another tool) with the program that sent it.

## End it

- Wait: the model comes back on its own when the job ends.
- `strata_suspend.py stop --all` (or `--pid N` for one job), or the **Stop the job** button on the Locked card:
  it ends the job's processes, and Strata comes back within seconds. The job is recorded as stopped by you:
  the agent that ran it is told not to run it again, and Strata never wakes it for that job.
- A holder that is not a wrapped job (a process held with `strata_suspend.py hold --pid N`) is not ended for you:
  end that program, or `strata_suspend.py release` to let Strata come back over it.

## When it keeps coming back

If `status` (or the Locked card) says a job was **started several times in the last 30 minutes**, the agent
that runs it is still working on its task and starts the job again whenever it ends. Stop that agent first: open
its session (the name is shown) in opencode or Claude Code and press Esc. Then stop the job.

## Other commands

| Command | What it does |
|---|---|
| `strata_suspend.py jobs` | The wrapped jobs of the last 7 days: done, running, cut off (a crash, a power cut), stopped by you |
| `strata_suspend.py dismiss --record NAME [--why TEXT]` | A cut-off job whose result is no longer needed (NAME: its log's file name from `jobs`): its agent is not woken for it at the next Strata start. A job run again in the same folder is skipped by itself |
| `strata_suspend.py release` | Ends every hold; the model comes back even if a job still runs |
| `strata_suspend.py resume` | The same, and loads the model now |
