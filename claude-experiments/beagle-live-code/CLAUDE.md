# Beagle Live Code Agent

A custom Claude agent that codes in Beagle exclusively through a live REPL connection — no file system access.

## What this is

Instead of the traditional "read file, edit file, run tests" loop, this agent interacts with a **running Beagle program** through its socket REPL. Every eval modifies the live program. The agent has no Read, Write, Edit, Glob, or Grep tools — the REPL is its only interface.

Built with the Claude Agent SDK (TypeScript) using custom MCP tools.

## Running

```bash
# Terminal 1: start the Beagle REPL server
beag run /path/to/repl_server.bg
# Server listens on 127.0.0.1:7888

# Terminal 2: start the agent
npm start
# or: npx tsx beagle_agent.ts
```

## Architecture

```
User → beagle_agent.ts → Claude Agent SDK → Claude Code CLI
                                ↕ (MCP)
                        beagle-repl MCP server
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

### MCP Tools

| Tool | Description |
|------|-------------|
| `beagle_eval` | Evaluate Beagle code in a named session |
| `beagle_describe` | Query REPL server capabilities |
| `beagle_sessions` | List active sessions |
| `beagle_interrupt` | Cancel a running evaluation |

### Key Implementation Details

- **`delete process.env.CLAUDECODE`** at top of file — required to allow the Agent SDK (which spawns Claude Code CLI) to run from within a Claude Code session
- **`allowedTools` whitelist** — only the 4 MCP tools are available; all built-in file tools are blocked
- **Session resumption** — captures `session_id` from the init message, passes `resume: sessionId` on subsequent turns for multi-turn conversations
- **`mcpServers` and `permissionMode` must be passed on every turn** — they are NOT persisted by the session resume mechanism
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
