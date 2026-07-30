# Production Agent Runtime

_Started 2026-07-30. This is the execution plan for turning the observable-agent POC into a daily-use system._

## Product boundary

Scry owns the durable, inspectable agent state: agents, conversations, messages, jobs, tool calls,
artifacts, approvals, and traces are language objects in the per-type arenas. A provider owns model
inference. The viewer is a lens and control surface over the same live heap; it is not a second
agent implementation and does not reconstruct state from logs.

There are two provider shapes:

1. **Native API provider.** `AnthropicModel` participates in Scry's `Model ↔ Tool` loop. Scry sees
   and executes every tool call.
2. **Subscription CLI provider.** `CliModel` delegates a bounded turn to an installed `claude` or
   `codex` process using that CLI's existing login. The child owns its internal tool loop and Scry
   records the final turn. Scry never reads or copies either provider's credential cache.

These must remain distinct. Pretending opaque CLI-internal tool calls are native Scry tool calls
would make the viewer misleading.

## Current daily-use entry point

Authenticate once with each provider's own CLI, then run:

```bash
claude auth status
codex login status

SCRY_PROVIDER=claude ./scry run examples/assistant.scry
SCRY_PROVIDER=codex  ./scry run examples/assistant.scry
```

Both adapters default to read-only delegated work. To let the provider edit the current workspace:

```bash
SCRY_PROVIDER=codex SCRY_AGENT_ACCESS=workspace-write ./scry run examples/assistant.scry
```

`SCRY_MODEL` optionally selects a provider model; unset means the CLI's current default.
`SCRY_MAX_TURNS` bounds delegated Claude turns (default 8). Unknown providers fall back to the
offline `ScriptedModel` rather than silently selecting a paid provider.

### As-built provider commands

- Claude: `claude -p --output-format json --max-turns N --permission-mode plan|acceptEdits`
- Codex: `codex exec --ephemeral --sandbox read-only|workspace-write --skip-git-repo-check`

Prompts and model names are shell-quoted. Child stdin is `/dev/null`, so a provider cannot steal
input from Scry's REPL. `Process.capture` returns stdout only while leaving provider progress on
stderr; a private output marker carries the real child exit status. CLI rate-limit and auth errors
become model errors instead of empty assistant messages.

## Production gates

The current slice is usable, but the runtime is not yet production-ready. Work proceeds in this
order; each phase must be testable without live provider access.

### P1 — trustworthy process boundary

- Replace shell-string execution for provider adapters with `Process.spawn(executable, args,
  stdin, cwd, envPolicy) -> Child` and structured `ProcessResult { exitCode, stdout, stderr }`.
- Add wall-clock timeout, cancellation, process-group termination, output byte limits, and explicit
  UTF-8/binary behavior.
- Stream stdout/stderr incrementally while keeping safepoints cooperative.
- Add hermetic fake-provider contract tests for success, malformed output, rate limit, timeout,
  cancellation, oversized output, and a child that attempts to read stdin.

Exit gate: no provider path relies on `/bin/sh`; a hung or noisy child cannot hang or exhaust Scry.

### P2 — first-class agent execution model

- Add `Run`, `Turn`, `ToolInvocation`, `Artifact`, `Usage`, and typed `AgentError` entities.
- Replace string stop reasons/errors with enums and `Result` values.
- Add cancellation tokens, deadlines, retry policy with jitter, provider concurrency limits, and
  a bounded worker pool.
- Separate one-shot delegation from native tool-loop models in the type system.
- Give every run stable IDs and parent/child links so sub-agent trees are truthful and searchable.

Exit gate: every state transition is explicit, bounded, inspectable, and race-tested.

### P3 — capabilities and approvals

- Replace unrestricted `ShellTool` and path-taking file tools with declared capabilities rooted at
  canonical workspace paths. Reject traversal and symlink escapes.
- Model approval requests as live entities with allow-once, allow-for-run, deny, and expiry.
- Default all providers and tools to read-only. Workspace writes require an explicit run policy;
  network and out-of-workspace access are separate grants.
- Redact configured secret patterns from messages, process output, traces, and viewer payloads.
- Add audit records for every capability decision and mutation.

Exit gate: untrusted prompts cannot expand their own authority, and the viewer exposes every grant.

### P4 — persistence and recovery

- Add an append-only journal for run transitions plus versioned snapshots of durable agent state.
- Store large tool output/artifacts by content hash instead of retaining unbounded strings in the
  heap.
- Resume interrupted runs only from explicit provider/session checkpoints; never replay a write
  implicitly after a crash.
- Add schema migration, corruption detection, retention controls, and export/delete operations.

Exit gate: kill-and-restart tests preserve history and never duplicate side effects.

### P5 — viewer becomes the operations console

- Render run/sub-agent topology, queued/running/waiting/failed states, token/turn usage, deadlines,
  provider identity, permissions, and artifacts from the first-class entities.
- Stream partial model text and tool output with bounded backpressure.
- Add cancel, retry-from-checkpoint, approve/deny, diff review, and artifact download actions.
- Add search across messages, tool calls, files, errors, and traces; retain deep links across restart.
- Require viewer authentication and origin protection before binding beyond loopback.

Exit gate: all daily operations can be understood and controlled without reading terminal logs.

### P6 — release engineering

- Cross-platform packaging, config validation, health diagnostics, structured logs, metrics, and
  crash reports with opt-in redaction.
- Compatibility matrix for Scry runtime, viewer protocol, Claude CLI, and Codex CLI versions.
- Load/soak/fault-injection tests; performance budgets for heap census, viewer polling, and agent
  concurrency.
- Signed releases, dependency/SBOM scanning, backup/restore documentation, and an upgrade policy.

Exit gate: a clean machine can install, authenticate, run, inspect, upgrade, and recover the system.

## Immediate next implementation

P1 is next. The concrete vertical slice is structured process results plus timeout/cancellation,
then moving `CliModel` off shell strings. That unlocks honest streaming into `Turn` and
`ToolInvocation` entities without baking provider quirks into the language's core agent model.
