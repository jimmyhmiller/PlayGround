# `beagle` — Beagle live-REPL toolkit (CLI)

The **toolkit form** of the Beagle live-code agent. Same nREPL transport and
`beagle.reflect` introspection the DeepSeek agent uses, but exposed as plain
subcommands so *any* agent (Claude included) or human can drive a running
Beagle program over the shell — no editing the source files directly.

Where the agent owns its REPL server for the life of its process, the CLI is
one process per invocation, so it **auto-starts the server detached** (it
survives across invocations) and discovers an already-running server by probing
`127.0.0.1:7888`. Daemon state lives in `/tmp/beagle-repl-cli.json`; server
stdout/stderr in `/tmp/beagle-repl-cli.log`.

## Run it

```bash
# from this directory
npm run cli -- eval '1 + 2'
# or the bin launcher (after `npm link` / `npm i -g .`)
beagle eval '1 + 2'
# or directly
npx tsx beagle_repl_cli.ts eval '1 + 2'
```

Any command that needs the REPL auto-starts the **default standalone server**
(`~/Documents/Code/beagle/resources/examples/repl_server.bg`) if nothing is
listening. Use `up <file>` to instead embed the REPL inside a specific
program's `main()` so evals run in that program's context.

## Commands

### Lifecycle
| Command | Description |
|---|---|
| `up [file]` | Start/ensure the server. With a `.bg` file, embeds the REPL in that file's `main()` (replaces any running server). |
| `down` | Stop the detached server (idempotent). |
| `restart [file]` | `down` then `up`. |
| `status` | Process info + whether the REPL round-trips a `describe`. |

### Code
| Command | Description |
|---|---|
| `eval <code\|->` | Evaluate Beagle code. `-` reads from stdin. |
| `load <file>` | Read a `.bg` file and eval its contents into the live program. |
| `persist <text\|->` | Write def(s) to disk **and** the running program. Needs `--namespace`, a `namespace X` directive in the text, or a file-backed namespace. `-` reads stdin. |

### Introspection (via `beagle.reflect`)
| Command | Description |
|---|---|
| `namespaces` | List loaded namespaces. |
| `ns-info <ns>` | Functions (with args/docs), structs, enums of a namespace. |
| `source <name>` | Stored source of a def (null for builtins/FFI/REPL defs). |
| `ns-source <ns>` | Concatenated source of a whole namespace. |
| `location <name>` | `{file, byte/line start/end}` of a def. |
| `search <query>` | Apropos over names + docstrings. |
| `doc <name>` | Doc/args/kind. Note: `name` resolves as a *reference* in the introspect session — alias its namespace first (`eval 'use foo.bar as bar'`) for `ns/name` forms. |

### REPL control
| Command | Description |
|---|---|
| `describe` / `sessions` / `interrupt <session>` | Server capabilities / sessions / cancel a running eval. |
| `resume <code>` / `abort` | Recover from a resumable exception (supply a value / abandon). |
| `main-status` / `main-resume [code]` / `main-abort` | Recover the main thread (game loop / GUI) after a crash. |
| `ls [dir]` | List a directory to find `.bg` files. |

## Flags

- `--session <name>` — REPL session (default `agent`). Defs in one session are visible across sessions (one global namespace table), but each session has its own current-namespace.
- `--json` — emit machine-readable output. For `eval`/`load`/`describe`/`sessions` this is the **raw nREPL message array** (`[{out},{value},{status}]`); for introspection it's the structured reflect data. Use this when consuming output programmatically.
- `--file <path>` — explicit file for `up`/`restart`.

## Examples

```bash
beagle up ~/proj/game.bg                 # run game.bg with an embedded REPL
beagle eval 'render-frame()'             # call into the live program
printf 'fn tick() { ... }' | beagle persist - --namespace game   # hot-patch + save
beagle source game/tick                  # read a def back
beagle eval 'broken()' --json            # inspect raw out/err/value/status
beagle main-status                       # did the game loop crash?
beagle main-resume                       # resume it after redefining the culprit
beagle down
```

## Notes for an agent driving this

- **Stateful & live.** Each eval mutates the running program. Define in one call, use in the next.
- **Read source through the REPL, not the filesystem.** `source` / `ns-source` reflect what's *actually loaded*, including REPL redefinitions, which the on-disk file may not yet have.
- **`persist` is the keep-it path.** `eval` does no file I/O. Don't `eval` a def then `persist` the same text — it compiles twice; go straight to `persist`.
- **Drift.** If `persist` refuses because the file changed since load, re-fetch with `source`/`ns-source` before retrying — something else moved.
- **Exit codes.** Non-zero on usage errors and connection failures; `0` otherwise (a Beagle `[error]` in the output is still exit `0` — inspect the text or `--json`).
