# Lifecycle & staleness stress battery: what the steelman bridge was hiding

The three agent evals compared *query semantics* and never exercised the
critique's architectural failure modes (CRITIQUE.md A1–A4, C11–C12):
lifecycle, document sync, memory, crash behavior. That was not an
accident of task choice — the steelman bridge **engineered those failure
modes away by construction** (sync-from-disk on every request, one warm
daemon per workspace, pull diagnostics, init-before-serve). This battery
measures the raw properties directly, mechanically, on the granite corpus
(15k LOC / 150 files), no agents involved. basedpyright 1.39.10 behind
`lspcli.py`; quarry prototype.

## T1 — Memory under agent fan-out

| Configuration | Resident memory |
|---|---|
| 1 pyright server, fully worked | **366 MB** |
| 6 servers (5-worktree fan-out + original) | **1,548 MB total** (~220–370 MB each) |
| quarry, any number of workspaces | **0 resident**; 7.3 MB disk cache per workspace |

Linear growth per concurrent agent workspace. This is the Serena-30GB
failure mode (CRITIQUE A4) in miniature: a 10-subagent fan-out on a
mid-size monorepo is multi-GB before any agent does anything.

## T2 — Cold-start silent-empty window: did NOT reproduce

First `workspace/symbol` query on a cold server blocked ~1.4 s and
returned complete results (105/105). Two reasons, both honest: the bridge
refuses to serve until `initialize` completes (bridge engineering again),
and basedpyright indexes 15k LOC fast. The silent-empty warm-up window
documented for jdtls/rust-analyzer-class servers on large repos is real
in the wild but not reproducible at this corpus scale with this server.
Recorded as a negative result, not evidence of absence.

## T3 — The central one: stale answers after un-notified disk edits

Scenario = the normal agent editing flow: the agent edits a file with its
own editor and never tells the language server (no didOpen/didChange on
that file; the bridge sends no didChangeWatchedFiles — file watching is
the LSP *client's* job, which is exactly the point).

Add a new call site to `notch_billet` in a never-opened file, then query
references from the definition side:

| | t+0s | t+1s | t+3s | t+6s | t+10s | t+15s | after explicit re-sync |
|---|---|---|---|---|---|---|---|
| LSP references count (truth: 19) | 17 | 17 | 17 | 17 | 17 | 17 | 19 |
| quarry refs (truth: 18, defs excluded) | **18, snapshot id changed** | | | | | | |

The LSP answer is stale **indefinitely** — not eventually-consistent,
frozen — and carries no error, no marker, nothing distinguishing it from
truth. Convergence requires someone re-syncing the edited file through
the session. Quarry's first query after the edit was correct because the
content-hash snapshot re-read the world, and the changed snapshot id is
the audit trail (CRITIQUE C12: "you cannot distinguish 'no' from 'don't
know'" — here, 'current' from 'frozen').

## T4 — Concurrent-editor hammer (40 s, edit every 0.5 s, query loop)

| | fresh | stale | error | stale rate |
|---|---|---|---|---|
| LSP | 130 | 115 | 0 | **46%** |
| quarry | 244 | 1 | 0 | **0.4%** |

Every stale LSP answer was served confidently. Quarry's single stale
response is consistent with its declared non-atomic snapshot (files are
hashed sequentially; an edit landing mid-scan can produce a torn
snapshot) — a real, disclosed limitation worth a spec note, three orders
of magnitude less frequent under this load.

## T5 — Crash behavior

`kill -9` the node analysis process mid-session: the first subsequent
call **hung for over two minutes** on the half-dead session; later calls
failed `BrokenPipeError`; the daemon never self-healed — recovery
required manually deleting the socket and restarting, losing all session
state. Quarry: there is nothing to kill; deleting its cache forces a
rebuild on the next call. "Restartability" is not a feature of the
stateless design — it is the absence of the problem.

## The conclusion the three evals could not deliver

The agent evals showed parity because the bridge quietly imported
quarry's design decisions: disk-as-truth per request, init-before-serve,
one supervised process. Each raw failure mode measured here is fixable
bridge-side — file watching, sync-all-dependents, process supervision,
readiness gating — and every such fix moves the bridge one step closer
to reimplementing the stateless protocol on top of the stateful one.
That is CRITIQUE.md's thesis in measurable form: the babysitting code is
the product. The difference is who writes it and whether its guarantees
are contractual: in LSP deployments each harness reinvents it (and the
46%/1.5 GB/hang numbers are what the defaults give you); in Quarry the
same guarantees are the protocol's conformance surface, checked by the
suite.
