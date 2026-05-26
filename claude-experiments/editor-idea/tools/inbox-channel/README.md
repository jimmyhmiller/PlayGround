# terminal-bevy inbox channel

Push messages from terminal-bevy's per-project inbox into a running
Claude Code session as `<channel source="terminal-bevy-inbox">` tags.

## How it works

1. terminal-bevy's GUI maintains a per-project inbox at
   `~/.terminal-bevy/inbox/<project_id>.jsonl`. Messages arrive via
   `tbinbox` (CLI), inter-project sends, or any external POST.
2. When the user clicks "Send to Claude" on a message in the Inbox
   pane, the message is appended to
   `~/.terminal-bevy/claude-outbox/<project_id>.jsonl`.
3. This MCP channel server (run by Claude Code as a stdio subprocess)
   tails that outbox file. Each new line becomes one
   `notifications/claude/channel` event, surfaced to Claude as:

   ```
   <channel source="terminal-bevy-inbox"
            project_id="..."
            message_id="..."
            sender="..."
            subject="...">
   body of the message
   </channel>
   ```

The server only forwards lines appended **after** the session started.
Old messages in the outbox are skipped (no replay flood on session
launch).

## Setup

Install the MCP SDK once:

```bash
cd tools/inbox-channel
bun add @modelcontextprotocol/sdk
```

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "terminal-bevy-inbox": {
      "command": "bun",
      "args": ["/absolute/path/to/editor-idea/tools/inbox-channel/server.ts"],
      "env": {
        "TERMINAL_BEVY_PROJECT_ID": "12345"
      }
    }
  }
}
```

Replace `12345` with the numeric project id from terminal-bevy. If you
omit `TERMINAL_BEVY_PROJECT_ID`, the server tails every project's
outbox and tags each event with its `project_id` meta — handy if one
Claude session works across multiple projects.

Launch Claude Code with the development flag (required during the
channels research preview):

```bash
claude --dangerously-load-development-channels server:terminal-bevy-inbox
```

## Sending messages

From a terminal:

```bash
tbinbox --body "deploy finished"
tbinbox --project alpha --sender ci --subject "build" --body "$(curl ...)"
echo "stdin body works too" | tbinbox --project alpha
```

From any tool that can open a Unix socket: connect to
`~/.terminal-bevy/socket` and send a single JSON object:

```json
{"action": "send_inbox", "project": "alpha", "sender": "deploy-bot", "body": "..."}
```

terminal-bevy reads to EOF, then appends.
