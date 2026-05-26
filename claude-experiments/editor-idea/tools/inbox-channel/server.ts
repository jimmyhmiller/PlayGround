#!/usr/bin/env bun
//
// terminal-bevy inbox channel — pushes flagged inbox messages into the
// running Claude Code session as <channel source="terminal-bevy"> tags.
//
// Wiring:
//   1. terminal-bevy GUI: in the Inbox pane, click "Send to Claude" on
//      a message. That appends the message to:
//        ~/.terminal-bevy/claude-outbox/<project_id>.jsonl
//   2. This server (spawned by Claude Code via MCP stdio) tails that
//      file. Every new line is emitted as one notifications/claude/channel
//      event to the session.
//   3. Claude reads the <channel> tag and acts on it.
//
// Picks the project_id from the env var TERMINAL_BEVY_PROJECT_ID. If
// unset, falls back to tailing ALL projects in the directory and
// stamps `project_id` onto each event's meta.
//
// Setup:
//   Register this in your project's .mcp.json:
//     {
//       "mcpServers": {
//         "terminal-bevy-inbox": {
//           "command": "bun",
//           "args": ["/abs/path/to/tools/inbox-channel/server.ts"],
//           "env": { "TERMINAL_BEVY_PROJECT_ID": "12345" }
//         }
//       }
//     }
//   Then launch claude with:
//     claude --dangerously-load-development-channels server:terminal-bevy-inbox

import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import * as fs from 'node:fs'
import * as path from 'node:path'
import * as os from 'node:os'

// Project id this server is bound to, if any. When set, we only tail
// that one project's outbox. When unset, we tail every *.jsonl in the
// directory and tag each event with its project_id.
const PROJECT_ID = process.env.TERMINAL_BEVY_PROJECT_ID || null

const OUTBOX_DIR = path.join(os.homedir(), '.terminal-bevy', 'claude-outbox')

// Poll interval. File watching with chokidar would be more reactive
// but adds a dep — terminal-bevy appends infrequently, so 500ms is
// plenty.
const POLL_MS = 500

type InboxMessage = {
  id: number
  ts: number
  sender?: string
  subject?: string | null
  body: string
  read?: boolean
}

const mcp = new Server(
  { name: 'terminal-bevy-inbox', version: '0.1.0' },
  {
    capabilities: { experimental: { 'claude/channel': {} } },
    instructions:
      "Events from terminal-bevy's inbox arrive as " +
      '<channel source="terminal-bevy-inbox" sender="..." project_id="..." subject="..."> ' +
      'tags. They are user-flagged messages the user explicitly chose to forward ' +
      'to you from their per-project inbox in the terminal-bevy app. Read and act ' +
      'on them like any user prompt. One-way: no reply tool exposed.',
  },
)

await mcp.connect(new StdioServerTransport())

// Per-file byte offset we've already forwarded. Persists in-memory; on
// server restart we re-emit everything from the start of the file (the
// user can clear the outbox file if they don't want that).
const offsets: Map<string, number> = new Map()

function safeReadAfter(file: string, offset: number): { text: string; size: number } {
  try {
    const st = fs.statSync(file)
    if (st.size <= offset) return { text: '', size: st.size }
    const fd = fs.openSync(file, 'r')
    try {
      const len = st.size - offset
      const buf = Buffer.alloc(len)
      fs.readSync(fd, buf, 0, len, offset)
      return { text: buf.toString('utf8'), size: st.size }
    } finally {
      fs.closeSync(fd)
    }
  } catch {
    return { text: '', size: offset }
  }
}

function projectFiles(): { id: string; path: string }[] {
  if (PROJECT_ID) {
    return [{ id: PROJECT_ID, path: path.join(OUTBOX_DIR, `${PROJECT_ID}.jsonl`) }]
  }
  try {
    const entries = fs.readdirSync(OUTBOX_DIR)
    return entries
      .filter(f => f.endsWith('.jsonl'))
      .map(f => ({ id: f.replace(/\.jsonl$/, ''), path: path.join(OUTBOX_DIR, f) }))
  } catch {
    return []
  }
}

async function forward(file: { id: string; path: string }) {
  const offset = offsets.get(file.path) ?? 0
  const { text, size } = safeReadAfter(file.path, offset)
  if (!text) {
    if (offsets.get(file.path) !== size) offsets.set(file.path, size)
    return
  }
  offsets.set(file.path, size)
  const lines = text.split('\n').filter(l => l.trim().length > 0)
  for (const line of lines) {
    let msg: InboxMessage
    try {
      msg = JSON.parse(line)
    } catch (e) {
      process.stderr.write(`[inbox-channel] bad line in ${file.path}: ${e}\n`)
      continue
    }
    // Meta keys must be identifiers (letters/digits/underscore).
    const meta: Record<string, string> = {
      project_id: file.id,
      message_id: String(msg.id ?? 0),
      ts: String(msg.ts ?? 0),
    }
    if (msg.sender) meta.sender = msg.sender
    if (msg.subject) meta.subject = msg.subject
    await mcp.notification({
      method: 'notifications/claude/channel',
      params: {
        content: msg.body ?? '',
        meta,
      },
    })
  }
}

// Ensure the directory exists so the first tail doesn't error.
try {
  fs.mkdirSync(OUTBOX_DIR, { recursive: true })
} catch {}

// Start at end-of-file: we only forward messages flagged AFTER this
// session began, never replay history. (If the user wants replay they
// can `cat` the file into claude via a slash command.)
for (const f of projectFiles()) {
  try {
    offsets.set(f.path, fs.statSync(f.path).size)
  } catch {
    offsets.set(f.path, 0)
  }
}

setInterval(async () => {
  for (const f of projectFiles()) {
    try {
      await forward(f)
    } catch (e) {
      process.stderr.write(`[inbox-channel] forward ${f.path}: ${e}\n`)
    }
  }
}, POLL_MS)
