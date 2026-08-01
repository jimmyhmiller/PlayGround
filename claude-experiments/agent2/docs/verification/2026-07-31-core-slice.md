# Core slice verification — 2026-07-31

## Deterministic checks

`coil verify` and `sh scripts/check_file_size.sh` passed with 12 tests:

- 4 JSON parser and JSON Schema tests;
- 6 provider wire/response contract tests, including parallel calls;
- 1 concurrency test proving two tool handlers overlap and results retain call order;
- 1 background runtime test covering model → two tools → continuation → final output.

Formatting, COIL lint, entry-point compilation, C warnings-as-errors, and the 4,000-line
source guard also passed.

After the project-workflow upgrade, the repository was migrated to configured project
tests and structured native settings. Verification also passed through the new paths:

- `coil test --list` discovered all four suites;
- `coil test provider` selected only the provider suite and passed its six tests;
- `coil test --jobs 4` ran all suites concurrently and passed all 12 tests;
- `coil check --target tests` checked every test graph and native input;
- `coil verify` validated formatting, lint, all target graphs, native compilation,
  linking, and all tests;
- `sh scripts/check_file_size.sh` additionally passed the repository's 4,000-line
  source guard.

The manifest now declares `source-roots`, the test roots/suffixes, the native include
directory, and libcurl through `pkg-config`. Generated native and test artifacts live
under `.coil/build/`; no test-output workaround remains in the repository workflow.

## Live checks

The following checks used short prompts and the existing configured credentials. No
credential value was printed or persisted.

| Boundary | Model | Outcome |
| --- | --- | --- |
| DeepSeek OpenAI compatibility | `deepseek-v4-flash` | Strict echo tool call, execution, reasoning-preserving continuation, final `hello` |
| DeepSeek Anthropic compatibility | `deepseek-v4-flash` | `tool_use`, execution, `tool_result` continuation, final `hello` |
| DeepSeek parallel calls | `deepseek-v4-flash` | Two calls proposed in one response, run on separate workers, final `alpha beta` |
| Codex App Server | `gpt-5.6-terra` | Initialize/thread/turn handshake, delta notifications, completed turn, final `codex-ok` |
| OpenAI Responses | — | Not run: neither `OPENAI_API_KEY` nor `OPENAI_KEY` was present |

The OpenAI adapter is covered by request/response contract tests but still needs a live
credential smoke test before its external compatibility is considered verified.

## Explicit limitations

- Direct OpenAI and DeepSeek adapters currently use complete JSON responses rather
  than incremental SSE; Codex App Server deltas are streamed as events.
- Background execution is in-process and joinable, not durable across restart.
- Cancellation builders exist for Codex, but cancellation is not yet propagated
  through the common runtime and tool workers.
- Tool timeout metadata is represented but not yet enforced by the executor.
- Events are not persisted. Concurrent sinks can observe write order differing from
  sequence order; `sequence` is authoritative.
- The Codex adapter starts one App Server process per request and rejects interactive
  server requests. A long-lived session/approval manager is future work.
