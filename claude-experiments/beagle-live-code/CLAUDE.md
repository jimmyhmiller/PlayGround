# Beagle Live Code Agent

A custom DeepSeek-powered agent that codes in Beagle exclusively through a live REPL connection — no file editing access.

## What this is

Instead of the traditional "read file, edit file, run tests" loop, this agent interacts with a **running Beagle program** through its socket REPL. Every eval modifies the live program. The agent has no Read, Write, Edit, Glob, or Grep tools — the REPL is its interface, plus a `list_directory` tool for locating .bg files.

Built directly on DeepSeek's OpenAI-compatible ChatCompletions API (via the `openai` npm package) with a hand-rolled tool-calling agent loop.

## Running

```bash
export DEEPSEEK_API_KEY=sk-...

# Terminal 1: start the Beagle REPL server
beag run /path/to/repl_server.bg
# Server listens on 127.0.0.1:7888

# Terminal 2: start the agent
npm start
# or: npx tsx beagle_agent.ts --model flash
```

### Models

Two DeepSeek models are supported via aliases (default: `pro`):

| Alias | Model |
|-------|-------|
| `pro` | `deepseek-v4-pro` |
| `flash` | `deepseek-v4-flash` |

Select with `--model pro|flash` on the CLI, the `DEEPSEEK_MODEL` env var, or switch mid-session with `/model pro|flash` (conversation state is kept). Full model names are also accepted. `DEEPSEEK_BASE_URL` overrides the API endpoint (default `https://api.deepseek.com`).

## Architecture

```
User → beagle_agent.ts → DeepSeek API (OpenAI-compatible, tool calling)
              ↕ (local tool dispatch)
      beagle tool handlers in-process
              ↕ (TCP JSON)
      Beagle REPL server (:7888)
```

### REPL Protocol

Beagle's REPL uses an nREPL-style JSON protocol over TCP:

- **Request**: Single newline-terminated JSON object: `{"op":"eval","id":"1","session":"agent","code":"1 + 2"}\n`
- **Response**: Multiple newline-delimited JSON messages, terminated by a `{"status":["done"]}` message
- **Streaming fields**: `out` (stdout), `err` (stderr), `ex` (exception), `value` (return value)

Supported ops: `eval`, `describe`, `ls-sessions`, `interrupt`, `close`

Example eval response (for `println("hello")`):
```
{"id":"3","out":"hello"}
{"id":"3","value":"hello"}
{"id":"3","status":["done"]}
```

### Tools

| Tool | Description |
|------|-------------|
| `beagle_run` / `beagle_load` | Start a program with embedded REPL / load a .bg file into the REPL |
| `beagle_eval` | Evaluate Beagle code in a named session |
| `beagle_persist` | Persist fn/struct/enum defs to disk AND the running program |
| `beagle_source` / `beagle_namespace_source` / `beagle_location` | Read source from the running program |
| `beagle_list_namespaces` / `beagle_namespace_info` / `beagle_search` / `beagle_doc` | Introspection |
| `beagle_status` / `beagle_describe` / `beagle_sessions` / `beagle_interrupt` | REPL server status & control |
| `beagle_resume` / `beagle_abort` | Resumable exception recovery |
| `beagle_main_status` / `beagle_main_resume` / `beagle_main_abort` | Main-thread crash recovery |
| `list_directory` | List a directory on disk (the only file system tool) |

### Key Implementation Details

- **Agent loop** — `runAgentTurn()` calls `chat.completions.create` with the tool schemas, executes any `tool_calls` locally, appends `role: "tool"` results, and repeats until the model answers with plain text (capped at 100 rounds)
- **Tool schemas** — tools are defined with zod shapes (same `tool(name, desc, shape, handler)` signature as before) and converted to JSON Schema via `z.toJSONSchema` for the OpenAI tools parameter
- **Multi-turn state** — the full `messages` array (including the system prompt) persists across user turns in-process; no session-resume mechanism needed
- **Sanitized assistant echoes** — assistant messages are re-appended as `{role, content, tool_calls}` only, never echoing `reasoning_content` back to the API (DeepSeek rejects it)
- **Abort handling** — Escape aborts the in-flight API request; any unanswered `tool_calls` get placeholder `role: "tool"` responses so the message list stays valid for the next request
- **Response formatting** — `formatReplResponse()` combines the nREPL streaming messages into a clean `output\n=> value` format

## What works

- Evaluating arbitrary Beagle code through the agent
- Multi-turn conversations with session continuity (define in turn 1, use in turn 2)
- Defining/redefining functions in the live program
- stdout capture from evals
- Error reporting from failed evals

## Next Steps

### Introspection
Figure out what Beagle expressions enable the agent to explore the running program:
- List all loaded namespaces
- List vars/functions in a namespace with signatures
- Get the source of a specific definition
- Check types, protocol implementations

This is critical for the agent to understand what it's working with before making changes. These are likely just `beagle_eval` calls with the right Beagle reflection APIs — no new REPL ops needed.

### Write-back / Persistence
When the agent defines or redefines a function, persist it to disk:
- Determine which file a namespace maps to
- Replace the old definition or append new ones
- Could be a Beagle-side feature (eval a save call) or a new MCP tool
- Need to decide granularity: save per-definition, per-namespace, or explicit save command

### Richer System Prompt
Teach the agent more about Beagle:
- Standard library functions and modules
- Common patterns and idioms
- Data type syntax, pattern matching, protocols
- Feed in `beag export-docs` output for API reference

### Better Error Handling
- Detect and surface compilation errors vs runtime errors
- Handle REPL server disconnection gracefully (reconnect or restart)
- Timeout handling for long-running evals

### UX Improvements
- Stream assistant text as it arrives (currently waits for full result)
- Show tool calls as they happen (so you can see what code the agent is evaluating)
- Color output for REPL results vs agent commentary
