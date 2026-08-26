# Agent Code Intelligence

LSP was designed for editors — a human at a cursor, a long-lived session, results
rendered as UI widgets. Coding agents are a different consumer with nearly opposite
requirements, and bolting LSP onto them has produced mostly lifecycle-babysitting code
and answers agents can't trust.

- **[CRITIQUE.md](./CRITIQUE.md)** — an enumeration of the ways LSP fails agents, grouped
  by the architectural assumption that causes each failure, with sources from real-world
  complaints (HN threads, Serena/opencode/Claude Code bridge bug trackers, tool-builder
  writeups).
- **[SPEC.md](./SPEC.md)** — *Quarry*, a v0.1 draft spec for an agent-native code
  intelligence protocol: stateless snapshot queries, name-based addressing, mandatory
  completeness/confidence metadata on every response, batch + budgeted results,
  pull-based diagnostics, speculative overlays, server-side transforms that return
  diffs, and per-language conformance levels (L0 textual → L3 semantic).
