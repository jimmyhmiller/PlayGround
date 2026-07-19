# hivemind

Drive a swarm of cheap LLM agents to fill stub markers in a codebase — each agent's work
**gated by a test** so the shared tree is always left valid. Generalized from a hand-rolled
DeepSeek harness used to port a Rust partial evaluator to JavaScript.

## The pattern

It's a **general parallelizable agent workload**: a pile of independent units, each built by
an agent and **kept only if a gate passes**. Nothing here is tied to any one kind of work —
the unit can be a function, a whole module, one handler/case, or a dispatch-table entry. How
you carve the work is up to you; smaller + more independent = more parallelism, easier debug.

1. **Skeleton with units.** You commit the structure (and any frozen shared shapes) and mark
   each independent unit `UNIMPLEMENTED("some.id")`. A unit is **one line** by default, or a
   **whole multi-line region** if you set `stubs.endMarker` (BEGIN…END) — good for a function
   body or an entire file.
2. **Manifest.** `tasks.json` describes each unit: `id` (matches the marker), `file`,
   `symbol`, `signature`, `spec`, `reference` (the oracle of truth), `deps`, `wave`.
3. **Conventions + context.** `AGENTS.md` and any frozen files (shared shapes, specs) are
   inlined into every prompt.
4. **The gate loop.** For each unit the model returns `{impl, test}` (or just `impl` if
   `output.testKey: null`). Under a **per-file lock**: snapshot → splice `impl` over the unit
   (line or region) → write the test → run the gate command (any command: test, build, lint,
   typecheck). Pass ⇒ keep; fail ⇒ **revert**. Passes loop until none resolve (a **fixpoint**),
   so a unit whose dependency is still unbuilt just fails its gate and is retried next pass.
   Failed attempts feed their gate output back to the model as the next prompt's feedback.

## Install

```sh
./install.sh                 # symlinks bin/hivemind -> ~/.local/bin/hivemind
```

`~/.local/bin` must be on your `PATH`.

## Use

```sh
export DEEPSEEK_KEY=sk-...           # or set provider.keyEnv to OPENAI_API_KEY, etc.
hivemind init my-port && cd my-port  # scaffold config, manifest, runner, AGENTS.md, demo
# ... build your skeleton + fill tasks.json ...
hivemind list                        # show discovered stubs
hivemind run --dry                   # print the first full prompt, make no API calls
hivemind run                         # fill stubs to a fixpoint
hivemind run --only=arith --limit=3  # narrow the batch
```

## Refactoring existing code

Stub-filling isn't the only use — `hivemind` also drives **parallel refactors** of code that
has no markers. Set `source: "manifest"` and `task: "refactor"` (or just run
`hivemind init --refactor`):

```sh
hivemind init --refactor my-refactor && cd my-refactor
```

- **Units come from the manifest**, not a scan. Each task names a `file` (whole-file
  rewrite) or a `file` + `begin`/`end` anchor regexes (a region — e.g. one function). Put
  the transformation in `instruction`.
- **The agent sees the current code** and returns the transformed version; it's spliced in,
  then the **gate runs**. The gate is your real behavior-preserving check (`npm test`,
  `cargo test`, `tsc --noEmit && eslint {file}`) — pass keeps it, fail reverts it.
- **Completion is tracked** in `.hivemind/done.json` (there's no marker to consume).
  Git-ignore it. Re-run a unit with `hivemind run --redo=<id substr>` (or `--redo` for all).
- Everything else is identical: per-file lock, concurrency, retry-with-feedback, fixpoint.

```json
{
  "source": "manifest", "task": "refactor",
  "manifest": "hivemind.tasks.json",
  "gate": { "command": "npm test", "final": "npm test" },
  "output": { "implKey": "code", "testKey": null }
}
```
```json
// hivemind.tasks.json
{ "tasks": [
  { "id": "async:api.js", "file": "src/api.js",
    "instruction": "Convert callbacks to async/await; keep the public API identical." },
  { "id": "rename:utils.parse", "file": "src/utils.js",
    "begin": "^function parse\\(", "end": "^}",
    "instruction": "Rename parse -> parseInput inside this function only." }
] }
```

> Tip: a *pattern-removing* codemod ("migrate every `oldApi(` call") doesn't even need
> manifest mode — point `stubs.marker` at the old pattern and use marker mode; the pattern's
> disappearance is the stateless done-signal.

## Config (`hivemind.config.json`)

| key | meaning |
|---|---|
| `source` | `"markers"` (scan for stubs, stateless) or `"manifest"` (units are tasks; for refactors) |
| `task` | `"implement"` (fill a stub) or `"refactor"` (transform existing code, preserve behavior) |
| `provider.baseUrl` / `model` | OpenAI-compatible `/chat/completions` endpoint + model |
| `provider.keyEnv` / `baseUrlEnv` / `modelEnv` | env vars consulted at runtime (override the literals) |
| `provider.jsonMode` | send `response_format: json_object` (default true) |
| `stubs.files` | explicit file list; if empty, `stubs.glob` is walked |
| `stubs.glob` | e.g. `src/**/*.js` (supports `**`, `*`, `?`) |
| `stubs.marker` | regex; capture group 1 is the marker id |
| `stubs.endMarker` | optional; when set, a unit is the multi-line region from the marker line through this end marker (inclusive). `{marker}` substituted. `null` = single-line |
| `manifest` | path to the task manifest (matched to markers by `id`/`symbol`) |
| `context` | files inlined verbatim into every prompt |
| `gate.command` | per-stub gate; `{marker}` / `{file}` substituted. Non-zero exit ⇒ revert |
| `gate.final` | run once after the fixpoint (optional) |
| `output.implKey` / `testKey` | JSON keys the model must return; set `testKey: null` for impl-only |
| `output.testPath` | where the test is written; `{marker}` substituted |
| `concurrency` / `attempts` | parallel agents; retries per stub |
| `prompt.system` / `prompt.user` | override the built-in templates (placeholders: `{marker}` `{file}` `{fileSrc}` `{contextBlock}` `{spec}` `{signature}` `{reference}` `{specComment}` `{feedback}` `{reqPath}` `{implKey}` `{testKey}`) |

## Why a gate, not just generation

Cheap models are unreliable per call but the *gate* makes the loop self-correcting: only
test-passing impls survive, failures revert cleanly and retry with the failure text, and the
fixpoint means you can run it unattended and inspect the small "stuck" set at the end.
