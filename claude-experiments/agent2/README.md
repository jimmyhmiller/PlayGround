# COIL Agent Harness

This repository contains the first vertical slice of a headless, observable agent
runtime written in COIL. The durable architectural rules live in [agent.md](agent.md).

The current slice can:

- call OpenAI through the Responses API;
- call DeepSeek through its OpenAI-compatible and Anthropic-compatible APIs;
- use Codex as an external agent through the Codex App Server JSONL protocol;
- expose one provider-neutral model, continuation, tool, authorization, result, and
  event contract;
- validate tool arguments before authorization;
- execute independent tool calls concurrently with an explicit concurrency bound;
- continue the model with ordered tool results while preserving opaque provider state;
- run the entire model/tool loop on a background worker and join it later;
- emit structured JSON event records suitable for a terminal, server, or future UI.

This is not yet a durable or remote service. Event persistence, recovery, network APIs,
stream backpressure, and cancellation propagation are deliberately not claimed by this
slice.

## Build and verify

Requirements are COIL, a C11 compiler, libcurl, pthread support, and (for Codex) the
`codex` CLI installed and authenticated.

```sh
coil build -O1
coil test --list
coil test --jobs 4
coil verify
sh scripts/check_file_size.sh
```

The test suites are declared in `Coil.toml`, so project `coil test` inherits the
libcurl native dependency, C source, include directory, and linker configuration.
`coil verify` validates the manifest, formatting, lint, every entry/test target graph,
native inputs, and all tests. The standalone size-check script enforces the repository's
4,000-line guard. Neither command spends model credits.

## Credentials

Provider credentials are read only inside provider adapters:

- OpenAI: `OPENAI_API_KEY`, with `OPENAI_KEY` as a fallback;
- DeepSeek: `DEEPSEEK_API_KEY`, with `DEEPSEEK_KEY` as a fallback;
- Codex: the existing Codex CLI login/session.

Credentials are used to construct request headers and are never included in emitted
events. Do not pass a credential as a CLI argument.

## Run a smoke request

```sh
./harness run openai gpt-5.6 "Answer with one short sentence."
./harness tool deepseek-openai deepseek-v4-flash "Call echo with the text hello."
./harness tool deepseek-anthropic deepseek-v4-flash "Call echo with the text hello."
./harness run codex gpt-5.6-terra "Briefly describe this repository."
```

Use `run` for automatic tool selection and `tool` to force the built-in strict `echo`
tool. The final model answer is written to stdout. Versioned lifecycle events are
written as one JSON object per line to stderr, so a controller can consume the two
streams independently.

The model name is always explicit. Current sensible defaults are shown above, but the
runtime does not hard-code a provider's model catalogue.

## Module map

```text
src/core/       Provider-neutral JSON, events, models, tools, schema validation
src/runtime/    Bounded model/tool loop, background handle, parallel tool executor
src/providers/  OpenAI Responses, two DeepSeek dialects, Codex App Server
src/infra/      HTTP and the narrow C/POSIX bridge
native/         libcurl, time, process, and pipe implementation
schemas/codex/  Generated schemas for the locally installed Codex App Server protocol
tests/          Deterministic unit, contract, concurrency, and runtime tests
```

COIL places native objects and dependency files under `.coil/build/native/` and uses
collision-free test runners under `.coil/build/test/`; source directories remain free
of generated build products.

Provider adapters own URLs, authentication, request/response formats, and preservation
of provider-specific continuation state. Runtime code never switches on a provider
name. Tool proposal, schema validation, authorization, and execution are separate
steps, each represented by lifecycle events.

## Important current behavior

- OpenAI uses strict Responses API function definitions and `function_call_output`
  continuation items keyed by `call_id`.
- DeepSeek OpenAI-compatible strict tools use its beta endpoint. On a continuation,
  the full previous assistant message is replayed so `reasoning_content` is retained.
- DeepSeek Anthropic compatibility uses `tool_use` and `tool_result` content blocks.
- Codex is launched as `codex app-server --listen stdio://`. The adapter initializes a
  session, starts a thread and turn, converts notifications to harness events, and has
  pure builders for steering and interrupt requests. Its current one-shot CLI adapter
  rejects interactive server requests rather than silently approving them.
- Parallel tool results retain model call order even when completion order differs.
- Concurrent event writes may arrive out of order; the atomic `sequence` field is
  authoritative and consumers should order records by it.
- Background handles require the request, provider context, emitter, and allocator to
  remain alive until `agent-run-join` returns.

## Next architectural slice

The next slice should put the runtime behind a versioned remote command/query/event
API and persist the event stream. That is the right point to add durable run identity,
reconnection cursors, cancellation propagation, and restart recovery. Direct-provider
token streaming also needs an incremental SSE decoder and an explicit backpressure
policy; only Codex deltas are streamed in the current implementation.
