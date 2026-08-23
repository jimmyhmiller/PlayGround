# ox-runner

Drop a markdown file in a folder. Ox Alpha picks it up, does the work in an
isolated git worktree, commits, and pushes a branch. Runs on
`computer.jimmyhmiller.com` under systemd.

## The loop

```
~/ox-runner/inbox/*.md   ->  daemon claims it  ->  git worktree off the base branch
                         ->  codex exec (stealth/ox-alpha via OpenRouter)
                         ->  commit + push  ox/<slug>-<stamp>
                         ->  done/ or failed/  +  a JSON record in runs.jsonl
```

## Writing a task

```markdown
---
repo: coil          # required, a name from `ox repos`
base: master        # optional, defaults to the repo's registered base
title: Add --json to lint   # optional, drives the branch name
timeout: 3600       # optional seconds
effort: high        # optional: low | high | max
attempts: 3         # optional retry budget
branch: ox/my-name  # optional, pinned branch name
sandbox: danger-full-access   # optional per-task override
push: true          # optional, false keeps the branch local
---

Add a `--json` flag to `coil lint` that emits one JSON object per diagnostic
with file, line, column, severity and message. Keep the human output the
default. Add a test under `tests/lint/` and make `cargo test` pass.
```

Unknown frontmatter keys are a hard error, so a typo fails loudly instead of
being silently ignored.

Queue it:

```bash
scp task.md computer.jimmyhmiller.com:ox-runner/inbox/    # or
ox submit task.md                                          # validates first
ox new coil "Add --json to lint"                           # opens $EDITOR
```

Files are only claimed once they have stopped changing for 2 seconds, so a
partial `scp` is never picked up mid-write.

## Registering repos

```bash
ox repo add coil git@github.com:jimmyhmiller/coil.git         # base auto-detected
ox repo add beagle git@github.com:jimmyhmiller/beagle.git main
ox repos
```

Each repo is cloned once into `~/Documents/Code/projects/<name>` — the same
canonical projects root the laptop uses, so the box ends up with an ordinary
checkout you can `cd` into. Runs never work in that checkout directly: each run
fetches, then adds a detached `git worktree` under `~/ox-runner/work/<run-id>/`
and deletes it when the run ends. Parallel runs on the same repo never collide,
and your checkout is never left on a strange branch.

Bulk registration, for when you already know the list:

```bash
printf 'coil\tgit@github.com:jimmyhmiller/coil.git\tmain\n' | ox repo import
ox repo clone 6      # clone everything registered but missing, 6 at a time
```

## Monitoring

```bash
ox status          # daemon health, queue depth, recent runs
ox watch           # the same, refreshing
ox queue           # what is waiting and what is in flight
ox tail            # follow the live run's agent log
ox show <run>      # metadata + the agent's own summary + changed files
ox log <run>       # the full agent transcript: every command, output, message
ox diff <run>      # the patch it produced
ox stats           # success rates, durations, retry counts
ox failed          # failed tasks and exactly why
ox logs -f         # the daemon's journal
ox doctor          # check the key, the model, systemd (--repos also pings remotes)
```

Every finished run appends one JSON object to `~/ox-runner/runs.jsonl`:

```json
{"run_id":"20260823-201455-add-json-to-lint","status":"success","repo":"coil",
 "branch":"ox/add-json-to-lint-20260823-201455","pushed":true,"commits":2,
 "files_changed":4,"insertions":118,"deletions":6,"duration_seconds":412,
 "attempt":1,"codex_exit":"0","base_sha":"...","run_dir":"..."}
```

Per-run artifacts live in `~/ox-runner/runs/<run-id>/`: `task.md`, `prompt.txt`,
`codex.jsonl` (raw events), `codex.log` (rendered), `final-message.txt`,
`diff.patch`, `files.txt`, `git.log`, `meta.json`.

## Statuses

| status | meaning | retried |
|---|---|---|
| `success` | committed and pushed | no |
| `partial` | codex errored or timed out but work was pushed anyway | no |
| `no-changes` | agent finished but committed nothing | yes |
| `timeout` | exceeded the task timeout with nothing to push | yes |
| `codex-error` | codex exited nonzero with nothing to push | yes |
| `fetch-failed` / `clone-failed` / `push-failed` | git trouble | yes |
| `invalid-task` / `unknown-repo` / `bad-base` / `no-api-key` | your input or setup | no |

Retryable failures go back to `inbox/` until the attempt budget is spent, then
land in `failed/` with a `.last-run.json` sidecar. `ox retry <run-id>` requeues
anything by hand.

## Robustness properties

- **Singleton daemon.** `flock` on `.daemon.lock`; a second one refuses to start.
- **Crash recovery.** On startup anything stranded in `processing/` goes back to
  `inbox/`, stale worktrees are deleted and pruned. Attempt counts live in
  `state/` so recovery cannot loop forever.
- **No work is ever lost.** If the agent leaves uncommitted changes, the runner
  commits them. If the push fails, the commits are parked under `refs/ox/`.
- **Never force-pushes.** A branch name collision retries under a timestamped
  name rather than overwriting anything.
- **Never mirror-pushes.** Pushes use an explicit `HEAD:refs/heads/<branch>`
  refspec, never a bare `git push` that could touch other refs.
- **Never disturbs your checkout.** Work happens in throwaway worktrees; the
  project checkout is only ever fetched into. Unpushed recovery commits are
  parked under `refs/ox/<run-id>`, a namespace nothing else uses.
- **Bounded.** Per-task timeout with `timeout -k`, `OX_CONCURRENCY` cap,
  systemd `Restart=always` with a start-limit backoff.
- **Auditable.** The complete agent transcript, the prompt it was given, and the
  exact diff are kept for every run.

## Configuration

`~/ox-runner/config/settings` is sourced by everything; any of these override
the defaults:

```sh
OX_CONCURRENCY=2
OX_TIMEOUT=3600
OX_MAX_ATTEMPTS=3
OX_SANDBOX=danger-full-access
OX_EFFORT=high
OX_MODEL=stealth/ox-alpha
OX_BRANCH_PREFIX=ox
OX_PROJECTS=$HOME/Documents/Code/projects
```

`OX_PROJECTS` (default `~/Documents/Code/projects`) is where checkouts live.

`~/ox-runner/env` holds `OPENROUTER_API_KEY` at mode 600.

## Sandbox

Runs default to `danger-full-access`: the agent executes commands unsandboxed as
your user. That is deliberate (dependency installs and builds work), but it
means a prompt-injecting repo could reach anything your account can. Set
`OX_SANDBOX=workspace-write` in `config/settings`, or `sandbox:` per task, to
confine writes to the checkout.

## Model

`stealth/ox-alpha` on OpenRouter: free, 1M context, reasoning mandatory, tool
calling confirmed. When the free window closes, `ox doctor` reports the model as
unavailable. Point `OX_MODEL` at a replacement and restart.

## Updating

Edit the source in this repo, then re-run `install.sh` on the box. It replaces
the scripts and restarts the service; `env`, `config/`, run history and the
project checkouts are left alone.
