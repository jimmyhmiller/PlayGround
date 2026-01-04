# Dashboard MCP Server Implementation

This document tracks the implementation status of the Dashboard MCP server tools.

## Overview

The MCP server (`src/main/mcp/server.ts`) exposes dashboard functionality to Claude agents, allowing them to:
- Discover and query dashboard state
- Create and manage windows
- Register custom widgets at runtime
- Execute code and manage processes

---

## Phase 1: Core Dashboard Control ✅

Basic tools for discovering and controlling the dashboard.

| Tool | Status | Description |
|------|--------|-------------|
| `dashboard_list_commands` | ✅ Done | List all available state commands with schemas |
| `dashboard_list_widget_types` | ✅ Done | List all widget types (built-in + custom) |
| `dashboard_get_state` | ✅ Done | Get dashboard state at a path |
| `dashboard_list_events` | ✅ Done | Query recent events from event system |
| `state_command` | ✅ Done | Execute any state command |
| `event_emit` | ✅ Done | Emit events into the event system |
| `window_create` | ✅ Done | Create window with widget (supports `pinned`) |
| `window_update` | ✅ Done | Update window properties |
| `window_close` | ✅ Done | Close a window |

**Files:**
- `src/main/mcp/server.ts` - MCP tool definitions and handlers
- `src/main/events/bridges/externalBridge.ts` - WebSocket bridge to dashboard

---

## Phase 2: Dynamic Widget System ✅

Register and manage custom widgets at runtime with React/JS code.

| Tool | Status | Description |
|------|--------|-------------|
| `widget_register` | ✅ Done | Register new widget type with React code |
| `widget_unregister` | ✅ Done | Remove a custom widget type |
| `widget_update_code` | ✅ Done | Hot-reload widget code |
| `widget_list_custom` | ✅ Done | List all custom widgets |

**Features:**
- Widgets persist across app restarts
- Hot-reload support via events
- Same context as EvalWidget (React, hooks, state access)

**Files:**
- `src/types/state.ts` - `CustomWidgetDefinition`, `CustomWidgetsState` types
- `src/main/state/StateStore.ts` - `customWidgets` command handlers
- `src/renderer/widgets/DynamicWidget.tsx` - Runtime compilation
- `src/renderer/hooks/useCustomWidgets.ts` - Load/sync custom widgets

**Example:**
```javascript
// Register a counter widget
widget_register({
  name: "my-counter",
  description: "Click counter widget",
  code: `
    return (props) => {
      const [count, setCount] = useState(0);
      return React.createElement('div', { style: baseWidgetStyle },
        React.createElement('button',
          { onClick: () => setCount(c => c + 1) },
          'Count: ' + count
        )
      );
    };
  `
});

// Create a window with it
window_create({ component: "my-counter", title: "Counter" });
```

---

## Phase 3: Events & Execution ✅

Async event handling and code execution.

| Tool | Status | Description |
|------|--------|-------------|
| `event_subscribe_temporary` | ✅ Done | Subscribe and wait for events (returns when matched) |
| `code_execute` | ✅ Done | Execute JS code in renderer context |
| `shell_spawn` | ✅ Done | Spawn a shell process |
| `shell_kill` | ✅ Done | Kill a spawned process |

**Files:**
- `src/main/mcp/server.ts` - MCP tool definitions and handlers
- `src/main/events/bridges/externalBridge.ts` - WebSocket handlers for subscribe-once, code:execute, shell:spawn, shell:kill
- `src/renderer/widgets/DynamicWidget.tsx` - `CodeExecutionHandler` component for code:execute
- `src/renderer/components/Desktop.tsx` - Mounts `CodeExecutionHandler`

**`event_subscribe_temporary` flow:**
1. Agent calls tool with pattern and optional timeout
2. MCP server sends `subscribe-once` to WebSocket bridge
3. Bridge subscribes to event store with pattern
4. When event matches, bridge returns it and auto-unsubscribes
5. If timeout occurs first, returns `{ timedOut: true }`

**Example:**
```javascript
// Wait for a pipeline to complete
event_subscribe_temporary({
  pattern: "pipeline.completed.**",
  timeout: 30000
})
// Returns the matching event or { timedOut: true }
```

**`code_execute` flow:**
1. Agent calls tool with code string
2. MCP server sends `code:execute` to WebSocket bridge
3. Bridge emits `code.execute` event with execution ID
4. Renderer's `CodeExecutionHandler` picks up event
5. Code executes in sandbox with access to emit, command, getState, etc.
6. Result emitted as `code.result.<executionId>` event
7. Bridge returns result to MCP server

**Available in code scope:**
- `React` - React library
- `emit(type, payload)` - Emit events
- `command(type, payload)` - Dispatch state commands
- `getState(path?)` - Get current state
- `getThemeVar(name)` - Get CSS variable value
- `console`, `JSON`, `Math`, `Date`, `Promise`, `fetch`

**Example:**
```javascript
// Simple calculation
code_execute({ code: "return 2 + 2" })
// Returns: 4

// Emit an event and return
code_execute({ code: "emit('data.loaded', { count: 42 }); return 'done'" })
// Returns: "done"

// Get state
code_execute({ code: "return getState('windows.list').length" })
// Returns: number of windows
```

**`shell_spawn` / `shell_kill` flow:**
1. Agent calls `shell_spawn` with command, args, cwd, env
2. Process is spawned with `child_process.spawn`
3. Returns `{ pid: "proc-123", message: "..." }`
4. stdout/stderr/exit streamed as events: `shell.stdout.<pid>`, `shell.stderr.<pid>`, `shell.exit.<pid>`
5. Use `shell_kill` with pid to terminate

**Example:**
```javascript
// Spawn a process
shell_spawn({
  command: "npm",
  args: ["run", "dev"],
  cwd: "/path/to/project"
})
// Returns: { pid: "proc-1", message: "Process spawned. Listen for events..." }

// Kill it later
shell_kill({ pid: "proc-1" })
// Returns: { success: true }
```

---

## Phase 4: Polish & Convenience ✅

Quality-of-life improvements and layout management.

| Tool | Status | Description |
|------|--------|-------------|
| `window_arrange` | ✅ Done | Arrange windows (grid, tile, cascade, stack) |
| `dashboard_create` | ✅ Done | Create a new dashboard (convenience wrapper) |
| `dashboard_switch` | ✅ Done | Switch to a dashboard (convenience wrapper) |
| `file_read` | ✅ Done | Read file contents |
| `file_write` | ✅ Done | Write file contents |

**Files:**
- `src/main/mcp/server.ts` - All tool implementations

**`window_arrange` modes:**
- `grid` - Arrange in grid (auto rows/cols based on count)
- `tile-horizontal` - Tile horizontally across screen
- `tile-vertical` - Tile vertically down screen
- `cascade` - Cascade from top-left with offset
- `stack` - Stack all windows at center

**Example:**
```javascript
// Arrange all windows in a grid
window_arrange({ mode: "grid" })

// Tile specific windows horizontally
window_arrange({
  mode: "tile-horizontal",
  windowIds: ["win-1", "win-2", "win-3"],
  padding: 10,
  gap: 5
})

// Cascade windows
window_arrange({ mode: "cascade" })
```

**`dashboard_create` / `dashboard_switch`:**
```javascript
// Create and switch to new dashboard
dashboard_create({ name: "My Dashboard" })

// Create without switching
dashboard_create({ name: "Background Dashboard", switchTo: false })

// Switch by ID
dashboard_switch({ id: "dash-abc123" })

// Switch by name (searches active project first)
dashboard_switch({ name: "My Dashboard" })
```

**`file_read` / `file_write`:**
```javascript
// Read a file
file_read({ path: "/path/to/file.txt" })
// Returns: { path, content, size, modified }

// Write a file (creates directories if needed)
file_write({
  path: "/path/to/file.txt",
  content: "Hello, world!"
})

// Append to file
file_write({
  path: "/path/to/log.txt",
  content: "New log entry\n",
  append: true
})
```

---

## Architecture

```
┌─────────────────┐     stdio      ┌─────────────────┐
│  Claude Agent   │◄──────────────►│   MCP Server    │
│  (claude-code)  │                │  (server.ts)    │
└─────────────────┘                └────────┬────────┘
                                            │ WebSocket
                                            │ ws://127.0.0.1:9876
                                            ▼
                                   ┌─────────────────┐
                                   │ External Bridge │
                                   │ (externalBridge)│
                                   └────────┬────────┘
                                            │
                        ┌───────────────────┼───────────────────┐
                        ▼                   ▼                   ▼
               ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
               │ Event Store │     │ State Store │     │  Renderer   │
               │  (events)   │     │   (state)   │     │  (widgets)  │
               └─────────────┘     └─────────────┘     └─────────────┘
```

---

## Testing

To test the MCP server manually:

1. Start the dashboard: `npm run dev`
2. Connect via WebSocket: `wscat -c ws://127.0.0.1:9876`
3. Send commands:
   ```json
   {"type": "state:list-widgets", "requestId": "1"}
   {"type": "state:command", "commandType": "windows.create", "payload": {"component": "chart", "title": "Test"}, "requestId": "2"}
   ```

---

## Changelog

- **2026-01-02**: Phase 4 complete - Polish & Convenience (window_arrange, dashboard_create, dashboard_switch, file_read, file_write)
- **2026-01-02**: Phase 3 complete - Events & Execution (event_subscribe_temporary, code_execute, shell_spawn, shell_kill)
- **2024-01-02**: Phase 2 complete - Dynamic Widget System
- **2024-01-01**: Phase 1 complete - Core Dashboard Control
