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

Authenticate once with each provider's own CLI, then run Scry normally:

```bash
claude auth status
codex login status

./scry run examples/assistant.scry
```

At `you>`, type `/models`. The in-terminal model picker contains the current subscription-backed
Claude aliases (Fable 5, Opus, Sonnet) and Codex GPT-5.6 family (Sol, Terra, Luna). Selection applies
to the current session immediately; `status` shows the active model. Startup remains safely
offline/read-only until a model is selected.

The developer viewer mirrors these controls for live inspection, but it is not required to operate
the agent. Provider/access/model policy is ordinary inspectable runtime state; no provider or
permission environment variable exists. API keys remain environment secrets and are never rendered.

### As-built provider commands

- Claude: `claude -p --output-format stream-json --include-partial-messages --model ALIAS`
- Codex: `codex app-server` over its JSONL protocol (`thread/start`, `turn/start`,
  `item/agentMessage/delta`, `turn/completed`)

Provider invocations use native `Process.spawnArgv`: executable and arguments go directly to
`execvp`, with no `/bin/sh` interpolation. The runtime exposes writable child stdin plus incremental
stdout/stderr polling, so both providers stream response deltas into the agent TUI. The browsable
result chunks contain `exitCode`, separately captured
`stdout`/`stderr`, `timedOut`, `truncated`, and `durationMs`. The native runtime polls both streams
cooperatively, enforces the TUI-configured deadline and combined output budget, and kills the whole
child process group on either limit. CLI rate-limit and auth errors become model errors instead of
empty assistant messages.

## Production gates

The current slice is usable, but the runtime is not yet production-ready. Work proceeds in this
order; each phase must be testable without live provider access.

### P1 — trustworthy process boundary (native core landed)

- **Done:** native argv execution for provider adapters; structured stdout/stderr, wall-clock
  timeout, process-group termination, combined output byte limit, duration, and cooperative polling.
- **Done:** first-class asynchronous `ChildProcess`, writable stdin, incremental output chunks, and
  provider delta streaming.
- **Next:** explicit user cancellation, cwd/environment policy, and explicit UTF-8/binary behavior.
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

P1's synchronous native boundary is built and used by both subscription adapters. The next vertical
slice is first-class `Run`/`Turn` state plus asynchronous cancellation and streaming into
`ToolInvocation` entities, without baking provider quirks into the language's core agent model.
