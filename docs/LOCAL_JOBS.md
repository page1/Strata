# Strata is locked: what to do (local)

A job run through `tools/strata_suspend.py run` (or `launch`) holds Strata with a lease: the model gives its memory
to the job, and every request waits until the job's lease ends. Then the model comes back by itself. While that
happens, the web page's Monitor shows **Locked**, and `status` says so.

## See what holds it

```
.venv\Scripts\python.exe tools\strata_suspend.py status
```

For each live lease (each job that holds Strata) it prints the lease id, the command, its folder, how long it has
run, and the agent session that started it. The lease id is what `stop --id` takes. The Monitor's Locked card shows
the same jobs.

## See who is waiting

`status`, `/health` and the Monitor (the "Requests" list) show every request in Strata now, oldest first after
the one generating, for example:

```
generating - opencode session "Nightly review"
queued (2 min) - subagent general of "Nightly review" - task: Write a report
waiting out a hold (14 min) - opencode session "Data cleanup"
queued - from web page
```

States: generating; queued (another request has the turn); waiting out a hold (a job holds Strata); loading the
model. A subagent is shown with its parent session and the task it was given; a request without a session (the
web page, another tool) with the program that sent it.

## End it

- Wait: the model comes back on its own when the job's lease ends.
- `strata_suspend.py stop --all` (or `stop --id LEASE_ID` for one job), or the **Stop the job** button on the Locked
  card: the lease ends at once and Strata comes back within seconds. The job's wrapper sees that on its next
  renewal (within about 10 s), ends the job's processes and prints STOPPED BY THE USER. The job is recorded as
  stopped by you: the agent that ran it is told not to run it again, and Strata never wakes it for that job.
- A lease from `strata_suspend.py hold --pid N` is ended by `stop` too, but only the hold goes: process N itself is
  not ended. End that program yourself if it should stop.

## When it keeps coming back

If `status` (or the Locked card) says a job was **started several times in the last 30 minutes**, the agent
that runs it is still working on its task and starts the job again whenever it ends. Stop that agent first: open
its session (the name is shown) in opencode or Claude Code and press Esc. Then stop the job.

## Other commands

| Command | What it does |
|---|---|
| `strata_suspend.py jobs [--leases-dir DIR]` | The leases in the run config's lease folder (or `--leases-dir`), newest first: state (live/released/expired/stopped/overridden), exit, command, folder, log |
| `strata_suspend.py stop --id LEASE_ID` or `--all` | Ends a job's lease (every job's with `--all`): the model comes back within seconds, and the job stops at its wrapper's next renewal |
| `strata_suspend.py resume` | Loads the model now, if no lease holds it. While a job holds it, `resume` is refused (409): `stop` the job first |

`release` is gone: `stop` ends a job's lease, and `resume` loads the model back once no lease holds it.

## Who is woken after a crash

When Strata starts again (with `"wake_cut_off_jobs": true`), it runs `strata_suspend.py wake`, which tells exactly
the sessions that were working when it went down - and nothing guesses. Who was working comes from two places, both
in the lease folder: an **expired lease** (its wrapper stopped renewing, so the job never reported its result back -
the lease names the session that ran it), and the **lease checkpoint** (`checkpoint.json`, written every minute and
with every save, kept aside as `checkpoint.prev.json` at the next start), whose register lists the sessions that had
a request in Strata when it went down. A clean end (Ctrl+C, `/restart`, closing the window) marks the checkpoint
`"shutting_down"`, so its register is not treated as a cut-off list and only the expired leases are told. Each
session is told once: the waker marks it through the server (`/lease/<id>/woken`, `/lease/session-woken`), and the
mark is read back from `checkpoint.prev.json`.
