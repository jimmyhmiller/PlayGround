# ADR 0001: Provider-neutral calls with opaque continuation state

Status: accepted, 2026-07-31

## Context

The harness must call models through substantially different protocols. OpenAI's
Responses API continues a response by ID, DeepSeek's OpenAI-compatible thinking mode
requires the prior assistant message (including `reasoning_content`) to be replayed,
DeepSeek's Anthropic dialect uses content blocks, and Codex exposes a stateful App
Server protocol.

Flattening these protocols into common chat messages would discard information needed
for correct continuations. Allowing provider wire types into the runtime would couple
orchestration to every external API.

## Decision

The runtime depends on `ModelProvider`, `ModelRequest`, `ModelResponse`, and structured
`ModelFailure` contracts. Each response carries provider-neutral output/tool calls plus
an opaque `raw-output`. `ModelContinuation` returns that opaque value, the previous
response ID, and provider-neutral tool results to the same adapter.

Adapters exclusively own authentication, transport, wire conversion, provider errors,
and usage extraction. Capability differences remain explicit request policy or adapter
behavior; the runtime never branches on provider identity.

Codex uses App Server instead of treating the CLI as a text subprocess because App
Server exposes structured thread/turn state, delta notifications, approvals, steering,
and interruption. The current adapter uses a one-shot process, but the protocol seam
allows a later long-lived session manager without changing the model/tool loop.

## Consequences

- Provider-specific reasoning and signatures can survive tool continuations.
- Common orchestration and tools are deterministic under fake-provider tests.
- Persisting `raw-output` will require provider-aware redaction and format versioning.
- Cross-provider continuation is intentionally unsupported.
- Provider capability discovery and routing remain future work rather than hidden
  string-based heuristics.

