#!/usr/bin/env node
/**
 * Dashboard MCP Server
 *
 * Standalone MCP server that connects to the dashboard via WebSocket
 * and exposes dashboard functionality as MCP tools.
 *
 * Usage: node server.js [--port 9876]
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from '@modelcontextprotocol/sdk/types.js';
import WebSocket from 'ws';
import * as fs from 'fs';
import * as path from 'path';

// WebSocket connection to dashboard
let ws: WebSocket | null = null;
let wsConnected = false;
let pendingRequests: Map<string, {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
}> = new Map();
let requestIdCounter = 0;

// Dashboard WebSocket port (can be overridden via args)
const DASHBOARD_PORT = parseInt(process.argv.find(arg => arg.startsWith('--port='))?.split('=')[1] ?? '9876');

/**
 * Connect to dashboard WebSocket
 */
async function connectToDashboard(): Promise<void> {
  return new Promise((resolve, reject) => {
    if (wsConnected && ws?.readyState === WebSocket.OPEN) {
      resolve();
      return;
    }

    ws = new WebSocket(`ws://127.0.0.1:${DASHBOARD_PORT}`);

    ws.on('open', () => {
      wsConnected = true;
      console.error('[mcp-server] Connected to dashboard');
      resolve();
    });

    ws.on('message', (data: Buffer) => {
      try {
        const msg = JSON.parse(data.toString());

        // Handle responses to our requests
        if (msg.requestId && pendingRequests.has(msg.requestId)) {
          const pending = pendingRequests.get(msg.requestId)!;
          pendingRequests.delete(msg.requestId);

          if (msg.type === 'error') {
            pending.reject(new Error(msg.error));
          } else {
            pending.resolve(msg);
          }
        }
      } catch (err) {
        console.error('[mcp-server] Failed to parse message:', err);
      }
    });

    ws.on('close', () => {
      wsConnected = false;
      console.error('[mcp-server] Disconnected from dashboard');
    });

    ws.on('error', (err) => {
      console.error('[mcp-server] WebSocket error:', err);
      if (!wsConnected) {
        reject(err);
      }
    });
  });
}

/**
 * Send a request to the dashboard and wait for response
 */
async function sendRequest(type: string, payload: Record<string, unknown> = {}): Promise<unknown> {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    await connectToDashboard();
  }

  const requestId = `req-${++requestIdCounter}`;

  return new Promise((resolve, reject) => {
    pendingRequests.set(requestId, { resolve, reject });

    ws!.send(JSON.stringify({
      type,
      requestId,
      ...payload,
    }));

    // Timeout after 30 seconds
    setTimeout(() => {
      if (pendingRequests.has(requestId)) {
        pendingRequests.delete(requestId);
        reject(new Error('Request timeout'));
      }
    }, 30000);
  });
}

/**
 * Define available MCP tools
 */
const TOOLS: Tool[] = [
  // Discovery tools
  {
    name: 'dashboard_list_commands',
    description: 'List all available state commands with their payload schemas. Use this to discover what commands can be executed.',
    inputSchema: {
      type: 'object' as const,
      properties: {},
    },
  },
  {
    name: 'dashboard_list_widget_types',
    description: 'List all available widget types with their props schemas, descriptions, and default values. Use this to discover what widgets can be created.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        category: {
          type: 'string',
          description: 'Filter by category: state, code, display, data, layout, input, interaction. Omit for all.',
        },
      },
    },
  },
  {
    name: 'state_command',
    description: 'Execute a state command. Use dashboard_list_commands to see available commands and their schemas.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        commandType: {
          type: 'string',
          description: 'The command type (e.g., "windows.create", "dashboards.switch")',
        },
        payload: {
          type: 'object',
          description: 'The command payload (schema depends on command type)',
        },
      },
      required: ['commandType'],
    },
  },
  {
    name: 'dashboard_get_state',
    description: 'Get the current dashboard state. Can optionally specify a path to get a specific part of the state.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        path: {
          type: 'string',
          description: 'Optional dot-separated path to a specific state value (e.g., "windows.list", "theme.current")',
        },
        format: {
          type: 'string',
          enum: ['json', 'summary'],
          description: 'Output format. "json" returns raw state, "summary" returns human-readable description. Default: json',
        },
      },
    },
  },
  {
    name: 'dashboard_list_events',
    description: 'List recent events from the dashboard event system. Useful for understanding data flow.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        pattern: {
          type: 'string',
          description: 'Event pattern to filter (e.g., "eval.**", "state.changed.*"). Default: "**" (all events)',
        },
        limit: {
          type: 'number',
          description: 'Maximum number of events to return. Default: 50',
        },
      },
    },
  },
  {
    name: 'event_emit',
    description: 'Emit an event into the dashboard event system. Widgets subscribed to matching patterns will receive it.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        eventType: {
          type: 'string',
          description: 'Event type (hierarchical, e.g., "data.loaded.users")',
        },
        payload: {
          type: 'object',
          description: 'Event payload data',
        },
      },
      required: ['eventType'],
    },
  },
  // Convenience tools
  {
    name: 'window_create',
    description: 'Create a new window with a widget. Returns the window ID.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        title: {
          type: 'string',
          description: 'Window title',
        },
        component: {
          type: 'string',
          description: 'Widget type (e.g., "eval-editor", "event-display", "chart", "code-block")',
        },
        props: {
          type: 'object',
          description: 'Widget props (varies by widget type)',
        },
        x: { type: 'number', description: 'X position' },
        y: { type: 'number', description: 'Y position' },
        width: { type: 'number', description: 'Width in pixels' },
        height: { type: 'number', description: 'Height in pixels' },
        pinned: { type: 'boolean', description: 'Whether the window is pinned' },
      },
      required: ['component'],
    },
  },
  {
    name: 'window_update',
    description: 'Update an existing window\'s properties.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        id: { type: 'string', description: 'Window ID' },
        title: { type: 'string', description: 'New title' },
        x: { type: 'number', description: 'New X position' },
        y: { type: 'number', description: 'New Y position' },
        width: { type: 'number', description: 'New width' },
        height: { type: 'number', description: 'New height' },
        props: { type: 'object', description: 'Widget props to merge' },
        pinned: { type: 'boolean', description: 'Pinned state' },
      },
      required: ['id'],
    },
  },
  {
    name: 'window_close',
    description: 'Close a window.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        id: { type: 'string', description: 'Window ID to close' },
      },
      required: ['id'],
    },
  },
  // Dynamic Widget System
  {
    name: 'widget_register',
    description: `Register a new custom widget type with React/JS code. The code will be compiled at runtime.

Available in widget code scope:
- React, useState, useEffect, useMemo, useCallback, useRef, memo
- useBackendState, useBackendStateSelector, useDispatch (state hooks)
- useEventSubscription, useEmit (event hooks)
- usePersistentState (persistent widget state)
- baseWidgetStyle (common styling)
- command(type, payload) - dispatch state commands
- getThemeVar(name) - get CSS variable value

Example code:
\`\`\`
return (props) => {
  const [count, setCount] = useState(0);
  return React.createElement('div', { style: baseWidgetStyle },
    React.createElement('button', { onClick: () => setCount(c => c + 1) },
      \`Clicked \${count} times\`
    )
  );
};
\`\`\``,
    inputSchema: {
      type: 'object' as const,
      properties: {
        name: {
          type: 'string',
          description: 'Unique widget type name (lowercase with hyphens, e.g., "my-widget")',
        },
        description: {
          type: 'string',
          description: 'Human-readable description of what the widget does',
        },
        code: {
          type: 'string',
          description: 'React component code as a string. Should return a function component.',
        },
        category: {
          type: 'string',
          description: 'Category for grouping (e.g., "display", "data", "input"). Default: "custom"',
        },
        defaultProps: {
          type: 'object',
          description: 'Default props for the widget',
        },
        propsSchema: {
          type: 'object',
          description: 'Documentation for props (key: description pairs)',
        },
      },
      required: ['name', 'description', 'code'],
    },
  },
  {
    name: 'widget_unregister',
    description: 'Remove a custom widget type. Existing instances will show an error.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        name: {
          type: 'string',
          description: 'Widget type name to remove',
        },
      },
      required: ['name'],
    },
  },
  {
    name: 'widget_update_code',
    description: 'Update a custom widget\'s code for hot-reloading. Existing instances will be recompiled.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        name: {
          type: 'string',
          description: 'Widget type name to update',
        },
        code: {
          type: 'string',
          description: 'New React component code',
        },
        description: {
          type: 'string',
          description: 'New description (optional)',
        },
        defaultProps: {
          type: 'object',
          description: 'New default props (optional)',
        },
      },
      required: ['name'],
    },
  },
  {
    name: 'widget_list_custom',
    description: 'List all custom (dynamically registered) widget types.',
    inputSchema: {
      type: 'object' as const,
      properties: {},
    },
  },
  // Phase 3: Events & Execution
  {
    name: 'event_subscribe_temporary',
    description: `Subscribe to events matching a pattern and wait for the first match. Useful for async workflows like waiting for pipeline results or user interactions.

Returns the first event that matches the pattern, or times out.

Example: Wait for a pipeline to complete
  event_subscribe_temporary({ pattern: "pipeline.completed.**", timeout: 30000 })`,
    inputSchema: {
      type: 'object' as const,
      properties: {
        pattern: {
          type: 'string',
          description: 'Event pattern to match (e.g., "pipeline.completed.**", "data.loaded.*")',
        },
        timeout: {
          type: 'number',
          description: 'Maximum time to wait in milliseconds. Default: 30000 (30 seconds)',
        },
      },
      required: ['pattern'],
    },
  },
  {
    name: 'code_execute',
    description: `Execute JavaScript code in the renderer context. Returns the result of the execution.

Available in code scope:
- React, useState, useEffect, useMemo, useCallback, useRef, memo
- useBackendState, useBackendStateSelector, useDispatch (state hooks)
- useEventSubscription, useEmit (event hooks)
- command(type, payload) - dispatch state commands
- getThemeVar(name) - get CSS variable value
- emit(type, payload) - emit an event

Example:
  code_execute({ code: "return 2 + 2" })
  code_execute({ code: "emit('data.loaded', { count: 42 }); return 'done'" })`,
    inputSchema: {
      type: 'object' as const,
      properties: {
        code: {
          type: 'string',
          description: 'JavaScript code to execute. Use "return" to return a value.',
        },
      },
      required: ['code'],
    },
  },
  {
    name: 'shell_spawn',
    description: `Spawn a shell process. Returns a process ID that can be used to track output and kill the process.

The process runs in the background. Use shell_kill to terminate it.
Output is streamed as events: shell.stdout.<pid>, shell.stderr.<pid>, shell.exit.<pid>`,
    inputSchema: {
      type: 'object' as const,
      properties: {
        command: {
          type: 'string',
          description: 'Command to run',
        },
        args: {
          type: 'array',
          items: { type: 'string' },
          description: 'Command arguments',
        },
        cwd: {
          type: 'string',
          description: 'Working directory (optional)',
        },
        env: {
          type: 'object',
          description: 'Environment variables to set (optional)',
        },
      },
      required: ['command'],
    },
  },
  {
    name: 'shell_kill',
    description: 'Kill a spawned shell process.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        pid: {
          type: 'string',
          description: 'Process ID returned by shell_spawn',
        },
        signal: {
          type: 'string',
          enum: ['SIGTERM', 'SIGKILL', 'SIGINT'],
          description: 'Signal to send. Default: SIGTERM',
        },
      },
      required: ['pid'],
    },
  },
  // Phase 4: Polish & Convenience
  {
    name: 'window_arrange',
    description: `Arrange windows in various layouts. Useful for organizing the desktop.

Arrangement modes:
- grid: Arrange in a grid (auto rows/cols based on count)
- tile-horizontal: Tile horizontally across the screen
- tile-vertical: Tile vertically down the screen
- cascade: Cascade from top-left with offset
- stack: Stack all windows at the same position`,
    inputSchema: {
      type: 'object' as const,
      properties: {
        mode: {
          type: 'string',
          enum: ['grid', 'tile-horizontal', 'tile-vertical', 'cascade', 'stack'],
          description: 'Arrangement mode',
        },
        padding: {
          type: 'number',
          description: 'Padding from screen edges in pixels. Default: 20',
        },
        gap: {
          type: 'number',
          description: 'Gap between windows in pixels. Default: 10',
        },
        windowIds: {
          type: 'array',
          items: { type: 'string' },
          description: 'Specific window IDs to arrange. If omitted, arranges all non-pinned windows.',
        },
      },
      required: ['mode'],
    },
  },
  {
    name: 'dashboard_create',
    description: 'Create a new dashboard in the current project. Convenience wrapper for state_command.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        name: {
          type: 'string',
          description: 'Name for the new dashboard',
        },
        projectId: {
          type: 'string',
          description: 'Project ID to create dashboard in. If omitted, uses active project.',
        },
        switchTo: {
          type: 'boolean',
          description: 'Switch to the new dashboard after creation. Default: true',
        },
      },
      required: ['name'],
    },
  },
  {
    name: 'dashboard_switch',
    description: 'Switch to a different dashboard. Convenience wrapper for state_command.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        id: {
          type: 'string',
          description: 'Dashboard ID to switch to',
        },
        name: {
          type: 'string',
          description: 'Dashboard name to switch to (searches in active project)',
        },
      },
    },
  },
  {
    name: 'file_read',
    description: 'Read the contents of a file.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        path: {
          type: 'string',
          description: 'Path to the file to read',
        },
        encoding: {
          type: 'string',
          description: 'File encoding. Default: utf-8',
        },
      },
      required: ['path'],
    },
  },
  {
    name: 'file_write',
    description: 'Write content to a file. Creates parent directories if needed.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        path: {
          type: 'string',
          description: 'Path to the file to write',
        },
        content: {
          type: 'string',
          description: 'Content to write to the file',
        },
        encoding: {
          type: 'string',
          description: 'File encoding. Default: utf-8',
        },
        append: {
          type: 'boolean',
          description: 'Append to file instead of overwriting. Default: false',
        },
      },
      required: ['path', 'content'],
    },
  },
];

/**
 * Format state as human-readable summary
 */
function formatStateSummary(state: unknown, path?: string): string {
  if (path === 'windows' || path === 'windows.list') {
    const windows = (state as { list?: unknown[] })?.list ?? state;
    if (!Array.isArray(windows)) return JSON.stringify(state, null, 2);

    const lines = ['Windows:', ''];
    for (const win of windows as Array<{ id: string; title: string; component: string; x: number; y: number; width: number; height: number; pinned?: boolean }>) {
      lines.push(`  [${win.id}] "${win.title}" (${win.component})`);
      lines.push(`    Position: ${win.x}, ${win.y}  Size: ${win.width}x${win.height}${win.pinned ? '  [PINNED]' : ''}`);
    }
    return lines.join('\n');
  }

  if (path === 'projects' || path === 'projects.list') {
    const projects = (state as { list?: unknown[] })?.list ?? state;
    if (!Array.isArray(projects)) return JSON.stringify(state, null, 2);

    const lines = ['Projects:', ''];
    for (const proj of projects as Array<{ id: string; name: string; dashboardIds: string[]; activeDashboardId?: string }>) {
      lines.push(`  [${proj.id}] "${proj.name}"`);
      lines.push(`    Dashboards: ${proj.dashboardIds.length}  Active: ${proj.activeDashboardId ?? 'none'}`);
    }
    return lines.join('\n');
  }

  // Default: pretty-print JSON
  return JSON.stringify(state, null, 2);
}

/**
 * Handle tool calls
 */
async function handleToolCall(name: string, args: Record<string, unknown>): Promise<string> {
  switch (name) {
    case 'dashboard_list_commands': {
      const result = await sendRequest('state:list-commands') as { commands: unknown };
      return JSON.stringify(result.commands, null, 2);
    }

    case 'dashboard_list_widget_types': {
      const result = await sendRequest('state:list-widgets') as { widgets: Record<string, { category: string }> };
      const { category } = args as { category?: string };

      // Filter by category if specified
      let widgets = result.widgets;
      if (category) {
        widgets = Object.fromEntries(
          Object.entries(widgets).filter(([_, info]) => info.category === category)
        );
      }

      return JSON.stringify(widgets, null, 2);
    }

    case 'state_command': {
      const { commandType, payload } = args;
      const result = await sendRequest('state:command', {
        commandType,
        payload: payload ?? {},
      }) as { result: unknown };
      return JSON.stringify(result.result, null, 2);
    }

    case 'dashboard_get_state': {
      const { path, format } = args as { path?: string; format?: string };
      const result = await sendRequest('state:get', { path }) as { state: unknown };

      if (format === 'summary') {
        return formatStateSummary(result.state, path);
      }
      return JSON.stringify(result.state, null, 2);
    }

    case 'dashboard_list_events': {
      const { pattern, limit } = args as { pattern?: string; limit?: number };
      const result = await sendRequest('query', {
        filter: {
          type: pattern ?? '**',
          limit: limit ?? 50,
        },
      }) as { events: unknown[] };
      return JSON.stringify(result.events, null, 2);
    }

    case 'event_emit': {
      const { eventType, payload } = args as { eventType: string; payload?: unknown };
      const result = await sendRequest('emit', {
        eventType,
        payload: payload ?? {},
      }) as { event: unknown };
      return JSON.stringify(result.event, null, 2);
    }

    case 'window_create': {
      const result = await sendRequest('state:command', {
        commandType: 'windows.create',
        payload: args,
      }) as { result: { id?: string } };
      return JSON.stringify(result.result, null, 2);
    }

    case 'window_update': {
      const result = await sendRequest('state:command', {
        commandType: 'windows.update',
        payload: args,
      }) as { result: unknown };
      return JSON.stringify(result.result, null, 2);
    }

    case 'window_close': {
      const result = await sendRequest('state:command', {
        commandType: 'windows.close',
        payload: { id: args.id },
      }) as { result: unknown };
      return JSON.stringify(result.result, null, 2);
    }

    case 'widget_register': {
      const { name, description, code, category, defaultProps, propsSchema } = args as {
        name: string;
        description: string;
        code: string;
        category?: string;
        defaultProps?: Record<string, unknown>;
        propsSchema?: Record<string, string>;
      };
      const result = await sendRequest('state:command', {
        commandType: 'customWidgets.register',
        payload: { name, description, code, category, defaultProps, propsSchema },
      }) as { result: { success?: boolean; error?: string; id?: string } };
      if (result.result.success) {
        return JSON.stringify({
          success: true,
          message: `Widget "${name}" registered successfully. Use it with: window_create component="${name}"`,
        }, null, 2);
      }
      return JSON.stringify(result.result, null, 2);
    }

    case 'widget_unregister': {
      const { name } = args as { name: string };
      const result = await sendRequest('state:command', {
        commandType: 'customWidgets.unregister',
        payload: { name },
      }) as { result: unknown };
      return JSON.stringify(result.result, null, 2);
    }

    case 'widget_update_code': {
      const { name, code, description, defaultProps } = args as {
        name: string;
        code?: string;
        description?: string;
        defaultProps?: Record<string, unknown>;
      };
      const result = await sendRequest('state:command', {
        commandType: 'customWidgets.update',
        payload: { name, code, description, defaultProps },
      }) as { result: unknown };
      return JSON.stringify(result.result, null, 2);
    }

    case 'widget_list_custom': {
      const result = await sendRequest('state:command', {
        commandType: 'customWidgets.list',
        payload: {},
      }) as { result: { widgets?: Array<{ name: string; description: string; category: string }> } };
      if (result.result.widgets) {
        const widgets = result.result.widgets.map(w => ({
          name: w.name,
          description: w.description,
          category: w.category,
        }));
        return JSON.stringify(widgets, null, 2);
      }
      return JSON.stringify(result.result, null, 2);
    }

    // Phase 3: Events & Execution
    case 'event_subscribe_temporary': {
      const { pattern, timeout } = args as { pattern: string; timeout?: number };
      const result = await sendRequest('subscribe-once', {
        pattern,
        timeout: timeout ?? 30000,
      }) as { event?: unknown; timedOut?: boolean };
      if (result.timedOut) {
        return JSON.stringify({ timedOut: true, message: `No event matching "${pattern}" received within timeout` }, null, 2);
      }
      return JSON.stringify(result.event, null, 2);
    }

    case 'code_execute': {
      const { code } = args as { code: string };
      const result = await sendRequest('code:execute', { code }) as { result?: unknown; error?: string };
      if (result.error) {
        return JSON.stringify({ error: result.error }, null, 2);
      }
      return JSON.stringify(result.result, null, 2);
    }

    case 'shell_spawn': {
      const { command, args: cmdArgs, cwd, env } = args as {
        command: string;
        args?: string[];
        cwd?: string;
        env?: Record<string, string>;
      };
      const result = await sendRequest('shell:spawn', {
        command,
        args: cmdArgs,
        cwd,
        env,
      }) as { pid: string };
      return JSON.stringify({
        pid: result.pid,
        message: `Process spawned. Listen for events: shell.stdout.${result.pid}, shell.stderr.${result.pid}, shell.exit.${result.pid}`,
      }, null, 2);
    }

    case 'shell_kill': {
      const { pid, signal } = args as { pid: string; signal?: string };
      const result = await sendRequest('shell:kill', {
        pid,
        signal: signal ?? 'SIGTERM',
      }) as { success: boolean; error?: string };
      return JSON.stringify(result, null, 2);
    }

    // Phase 4: Polish & Convenience
    case 'window_arrange': {
      const { mode, padding = 20, gap = 10, windowIds } = args as {
        mode: 'grid' | 'tile-horizontal' | 'tile-vertical' | 'cascade' | 'stack';
        padding?: number;
        gap?: number;
        windowIds?: string[];
      };

      // Get current windows
      const stateResult = await sendRequest('state:get', { path: 'windows.list' }) as {
        state: Array<{ id: string; x: number; y: number; width: number; height: number; pinned?: boolean }>;
      };
      let windows = stateResult.state;

      // Filter to specified windows or non-pinned
      if (windowIds && windowIds.length > 0) {
        windows = windows.filter(w => windowIds.includes(w.id));
      } else {
        windows = windows.filter(w => !w.pinned);
      }

      if (windows.length === 0) {
        return JSON.stringify({ message: 'No windows to arrange' }, null, 2);
      }

      // Get screen dimensions (assume reasonable defaults)
      const screenWidth = 1920;
      const screenHeight = 1080;
      const availWidth = screenWidth - padding * 2;
      const availHeight = screenHeight - padding * 2;

      const updates: Array<{ id: string; x: number; y: number; width: number; height: number }> = [];

      switch (mode) {
        case 'grid': {
          const count = windows.length;
          const cols = Math.ceil(Math.sqrt(count));
          const rows = Math.ceil(count / cols);
          const cellWidth = (availWidth - gap * (cols - 1)) / cols;
          const cellHeight = (availHeight - gap * (rows - 1)) / rows;

          windows.forEach((win, i) => {
            const col = i % cols;
            const row = Math.floor(i / cols);
            updates.push({
              id: win.id,
              x: padding + col * (cellWidth + gap),
              y: padding + row * (cellHeight + gap),
              width: Math.floor(cellWidth),
              height: Math.floor(cellHeight),
            });
          });
          break;
        }

        case 'tile-horizontal': {
          const tileWidth = (availWidth - gap * (windows.length - 1)) / windows.length;
          windows.forEach((win, i) => {
            updates.push({
              id: win.id,
              x: padding + i * (tileWidth + gap),
              y: padding,
              width: Math.floor(tileWidth),
              height: availHeight,
            });
          });
          break;
        }

        case 'tile-vertical': {
          const tileHeight = (availHeight - gap * (windows.length - 1)) / windows.length;
          windows.forEach((win, i) => {
            updates.push({
              id: win.id,
              x: padding,
              y: padding + i * (tileHeight + gap),
              width: availWidth,
              height: Math.floor(tileHeight),
            });
          });
          break;
        }

        case 'cascade': {
          const cascadeOffset = 30;
          const baseWidth = Math.min(600, availWidth - cascadeOffset * (windows.length - 1));
          const baseHeight = Math.min(400, availHeight - cascadeOffset * (windows.length - 1));

          windows.forEach((win, i) => {
            updates.push({
              id: win.id,
              x: padding + i * cascadeOffset,
              y: padding + i * cascadeOffset,
              width: baseWidth,
              height: baseHeight,
            });
          });
          break;
        }

        case 'stack': {
          const stackWidth = Math.min(800, availWidth);
          const stackHeight = Math.min(600, availHeight);
          const stackX = padding + (availWidth - stackWidth) / 2;
          const stackY = padding + (availHeight - stackHeight) / 2;

          windows.forEach((win) => {
            updates.push({
              id: win.id,
              x: stackX,
              y: stackY,
              width: stackWidth,
              height: stackHeight,
            });
          });
          break;
        }
      }

      // Apply updates
      for (const update of updates) {
        await sendRequest('state:command', {
          commandType: 'windows.update',
          payload: update,
        });
      }

      return JSON.stringify({
        message: `Arranged ${updates.length} windows in ${mode} layout`,
        windows: updates.map(u => u.id),
      }, null, 2);
    }

    case 'dashboard_create': {
      const { name, projectId, switchTo = true } = args as {
        name: string;
        projectId?: string;
        switchTo?: boolean;
      };

      // Get active project if not specified
      let targetProjectId = projectId;
      if (!targetProjectId) {
        const projectsResult = await sendRequest('state:get', { path: 'projects' }) as {
          state: { activeId?: string };
        };
        targetProjectId = projectsResult.state.activeId;
      }

      if (!targetProjectId) {
        return JSON.stringify({ error: 'No active project and no projectId specified' }, null, 2);
      }

      // Create the dashboard
      const result = await sendRequest('state:command', {
        commandType: 'dashboards.create',
        payload: { projectId: targetProjectId, name },
      }) as { result: { id?: string } };

      // Optionally switch to it
      if (switchTo && result.result.id) {
        await sendRequest('state:command', {
          commandType: 'dashboards.switch',
          payload: { id: result.result.id },
        });
      }

      return JSON.stringify({
        ...result.result,
        message: `Dashboard "${name}" created${switchTo ? ' and activated' : ''}`,
      }, null, 2);
    }

    case 'dashboard_switch': {
      const { id, name: dashboardName } = args as { id?: string; name?: string };

      if (!id && !dashboardName) {
        return JSON.stringify({ error: 'Either id or name must be specified' }, null, 2);
      }

      let targetId = id;

      // If name specified, find the dashboard
      if (!targetId && dashboardName) {
        const dashResult = await sendRequest('state:get', { path: 'dashboards' }) as {
          state: { list: Array<{ id: string; name: string; projectId: string }> };
        };
        const projectsResult = await sendRequest('state:get', { path: 'projects' }) as {
          state: { activeId?: string };
        };

        // Search in active project first
        const activeProjectId = projectsResult.state.activeId;
        let dashboard = dashResult.state.list.find(
          d => d.name.toLowerCase() === dashboardName.toLowerCase() && d.projectId === activeProjectId
        );

        // If not found, search all
        if (!dashboard) {
          dashboard = dashResult.state.list.find(
            d => d.name.toLowerCase() === dashboardName.toLowerCase()
          );
        }

        if (!dashboard) {
          return JSON.stringify({ error: `Dashboard "${dashboardName}" not found` }, null, 2);
        }
        targetId = dashboard.id;
      }

      const result = await sendRequest('state:command', {
        commandType: 'dashboards.switch',
        payload: { id: targetId },
      }) as { result: unknown };

      return JSON.stringify(result.result, null, 2);
    }

    case 'file_read': {
      const { path: filePath, encoding = 'utf-8' } = args as { path: string; encoding?: BufferEncoding };

      try {
        const resolvedPath = path.resolve(filePath);
        const content = fs.readFileSync(resolvedPath, { encoding });
        const stats = fs.statSync(resolvedPath);

        return JSON.stringify({
          path: resolvedPath,
          content,
          size: stats.size,
          modified: stats.mtime.toISOString(),
        }, null, 2);
      } catch (err) {
        return JSON.stringify({
          error: `Failed to read file: ${err instanceof Error ? err.message : String(err)}`,
        }, null, 2);
      }
    }

    case 'file_write': {
      const { path: filePath, content, encoding = 'utf-8', append = false } = args as {
        path: string;
        content: string;
        encoding?: BufferEncoding;
        append?: boolean;
      };

      try {
        const resolvedPath = path.resolve(filePath);

        // Create parent directories if needed
        const dir = path.dirname(resolvedPath);
        if (!fs.existsSync(dir)) {
          fs.mkdirSync(dir, { recursive: true });
        }

        if (append) {
          fs.appendFileSync(resolvedPath, content, { encoding });
        } else {
          fs.writeFileSync(resolvedPath, content, { encoding });
        }

        const stats = fs.statSync(resolvedPath);

        return JSON.stringify({
          success: true,
          path: resolvedPath,
          size: stats.size,
          append,
        }, null, 2);
      } catch (err) {
        return JSON.stringify({
          error: `Failed to write file: ${err instanceof Error ? err.message : String(err)}`,
        }, null, 2);
      }
    }

    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

/**
 * Main entry point
 */
async function main(): Promise<void> {
  // Create MCP server
  const server = new Server(
    {
      name: 'dashboard-mcp',
      version: '1.0.0',
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // Handle list tools
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    return { tools: TOOLS };
  });

  // Handle tool calls
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    try {
      const result = await handleToolCall(name, (args ?? {}) as Record<string, unknown>);
      return {
        content: [
          {
            type: 'text' as const,
            text: result,
          },
        ],
      };
    } catch (err) {
      return {
        content: [
          {
            type: 'text' as const,
            text: `Error: ${err instanceof Error ? err.message : String(err)}`,
          },
        ],
        isError: true,
      };
    }
  });

  // Try to connect to dashboard
  try {
    await connectToDashboard();
  } catch (err) {
    console.error('[mcp-server] Warning: Could not connect to dashboard. Will retry on first tool call.');
  }

  // Start server
  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error('[mcp-server] MCP server started');
}

main().catch((err) => {
  console.error('[mcp-server] Fatal error:', err);
  process.exit(1);
});
