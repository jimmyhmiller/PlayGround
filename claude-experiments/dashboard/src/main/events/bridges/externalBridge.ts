/**
 * External Bridge via WebSocket
 *
 * Allows external processes to connect and receive events.
 * Protocol is JSON-based with message types for emit, query, subscribe,
 * state operations, and command execution.
 */

import { WebSocketServer, WebSocket } from 'ws';
import { spawn, ChildProcess } from 'child_process';
import { matchesPattern } from '../utils';

// Track spawned processes by ID
const spawnedProcesses = new Map<string, ChildProcess>();
let processIdCounter = 0;
import type { EventStore } from '../EventStore';
import type { EventFilter, DashboardEvent } from '../../../types/events';
import type { StateStore } from '../../state/StateStore';

interface ExternalBridgeOptions {
  port?: number;
  stateStore?: StateStore;
}

interface ExternalBridgeHandle {
  close(): void;
  getClientCount(): number;
}

interface WebSocketMessage {
  type: 'emit' | 'query' | 'subscribe' | 'subscribe-once' | 'state:get' | 'state:command' | 'state:list-commands' | 'state:list-widgets' | 'code:execute' | 'shell:spawn' | 'shell:kill';
  eventType?: string;
  payload?: unknown;
  filter?: EventFilter;
  pattern?: string;
  requestId?: string;
  path?: string;
  commandType?: string;
  // For code:execute
  code?: string;
  // For shell:spawn
  command?: string;
  args?: string[];
  cwd?: string;
  env?: Record<string, string>;
  // For shell:kill
  pid?: string;
  signal?: string;
  // For subscribe-once
  timeout?: number;
}

interface ExtendedWebSocket extends WebSocket {
  subscriptionPattern?: string;
}

// Available widget types for discovery
const WIDGET_TYPES_INFO: Record<string, {
  description: string;
  category: string;
  defaultProps: Record<string, unknown>;
  propsSchema: Record<string, string>;
}> = {
  'state-value': {
    description: 'Display a value from the state tree',
    category: 'state',
    defaultProps: { path: 'projects', selector: 'count', suffix: 'projects' },
    propsSchema: { path: 'string - state path', selector: 'string - value selector', suffix: 'string (optional)' },
  },
  'state-list': {
    description: 'Display a list from the state tree',
    category: 'state',
    defaultProps: { path: 'projects', selector: 'list', labelKey: 'name', idKey: 'id' },
    propsSchema: { path: 'string', selector: 'string', labelKey: 'string', idKey: 'string' },
  },
  'dashboard-list': {
    description: 'Display list of dashboards in current project',
    category: 'state',
    defaultProps: {},
    propsSchema: {},
  },
  'eval': {
    description: 'Interactive JavaScript evaluation widget',
    category: 'code',
    defaultProps: {},
    propsSchema: { initialCode: 'string (optional)' },
  },
  'eval-editor': {
    description: 'Code editor with evaluation capability',
    category: 'code',
    defaultProps: { title: 'Code', iterations: 1 },
    propsSchema: {
      title: 'string',
      channel: 'string - event channel for results',
      language: 'string - javascript, python, etc.',
      initialCode: 'string (optional)',
      iterations: 'number - times to run',
    },
  },
  'chart': {
    description: 'Bar or line chart from event data',
    category: 'display',
    defaultProps: { subscribePattern: 'data.**', chartType: 'bar' },
    propsSchema: {
      subscribePattern: 'string - event pattern to subscribe',
      chartType: 'bar | line',
      dataKey: 'string - payload key for value',
      labelKey: 'string - payload key for label',
      maxPoints: 'number (optional)',
      height: 'number (optional)',
      title: 'string (optional)',
    },
  },
  'table': {
    description: 'Table display from event data',
    category: 'display',
    defaultProps: { subscribePattern: 'data.**' },
    propsSchema: {
      subscribePattern: 'string - event pattern',
      columns: 'array of column configs (optional)',
    },
  },
  'stats': {
    description: 'Statistics display (min, max, avg, count)',
    category: 'display',
    defaultProps: { subscribePattern: 'data.**', dataKey: 'value', showStats: true },
    propsSchema: {
      subscribePattern: 'string',
      dataKey: 'string - payload key for value',
      showStats: 'boolean',
    },
  },
  'transform': {
    description: 'Transform events and re-emit',
    category: 'data',
    defaultProps: { subscribePattern: 'data.**', emitAs: 'transformed', transform: '(p) => p' },
    propsSchema: {
      subscribePattern: 'string',
      emitAs: 'string - output event type',
      transform: 'string - JS function as string',
    },
  },
  'layout': {
    description: 'Container for nested widgets',
    category: 'layout',
    defaultProps: { direction: 'horizontal', gap: 8, children: [] },
    propsSchema: {
      direction: 'horizontal | vertical',
      gap: 'number',
      children: 'array of widget configs',
    },
  },
  'event-display': {
    description: 'Display events matching a pattern',
    category: 'display',
    defaultProps: { subscribePattern: '**', maxEvents: 10 },
    propsSchema: {
      subscribePattern: 'string - event pattern',
      maxEvents: 'number - max events to show',
      fields: 'array of field names (optional)',
      title: 'string (optional)',
    },
  },
  'selector': {
    description: 'List selector from event data',
    category: 'input',
    defaultProps: { direction: 'vertical', labelKey: 'label', idKey: 'id' },
    propsSchema: {
      subscribePattern: 'string - pattern for items',
      labelKey: 'string - key for display label',
      idKey: 'string - key for item id',
      channel: 'string - output event channel',
      direction: 'vertical | horizontal',
    },
  },
  'code-block': {
    description: 'Syntax-highlighted code display',
    category: 'display',
    defaultProps: { lineNumbers: true },
    propsSchema: {
      subscribePattern: 'string - event pattern',
      codeKey: 'string - payload key for code',
      language: 'string - syntax highlighting language',
      lineNumbers: 'boolean',
    },
  },
  'webview': {
    description: 'Embedded web page',
    category: 'display',
    defaultProps: { height: 300 },
    propsSchema: {
      url: 'string - static URL',
      subscribePattern: 'string - pattern for dynamic URL',
      pathKey: 'string - payload key for URL path',
      height: 'number',
    },
  },
  'web-frame': {
    description: 'Alias for webview',
    category: 'display',
    defaultProps: { height: 300 },
    propsSchema: { url: 'string', height: 'number' },
  },
  'file-loader': {
    description: 'Load and transform files',
    category: 'data',
    defaultProps: { files: [], reloadInterval: 0 },
    propsSchema: {
      files: 'array of file paths or glob patterns',
      channel: 'string - output event channel',
      transform: 'string - JS transform function',
      reloadInterval: 'number - reload interval in ms (0 = no reload)',
    },
  },
  'process-runner': {
    description: 'Run a process/command',
    category: 'data',
    defaultProps: { showOutput: true, maxOutputLines: 100 },
    propsSchema: {
      command: 'string - command to run',
      args: 'array of strings',
      cwd: 'string - working directory',
      showOutput: 'boolean',
      maxOutputLines: 'number',
    },
  },
  'file-drop': {
    description: 'Drag and drop file input',
    category: 'input',
    defaultProps: { title: 'Drop file here', showInfo: true },
    propsSchema: {
      title: 'string',
      channel: 'string - output event channel',
      showInfo: 'boolean',
    },
  },
  'pipeline': {
    description: 'Data processing pipeline',
    category: 'data',
    defaultProps: { autoStart: true, showStatus: false },
    propsSchema: {
      stages: 'array of pipeline stage configs',
      autoStart: 'boolean',
      showStatus: 'boolean',
    },
  },
  'pipeline-status': {
    description: 'Display pipeline status',
    category: 'display',
    defaultProps: { showAll: true },
    propsSchema: { showAll: 'boolean' },
  },
  'processor-list': {
    description: 'List available pipeline processors',
    category: 'display',
    defaultProps: {},
    propsSchema: {},
  },
  'inline-pipeline': {
    description: 'Compact inline pipeline',
    category: 'data',
    defaultProps: { autoStart: true },
    propsSchema: { stages: 'array', autoStart: 'boolean' },
  },
  'chat': {
    description: 'Claude chat interface (ACP integration)',
    category: 'interaction',
    defaultProps: { title: 'Claude Chat' },
    propsSchema: { title: 'string' },
  },
};

// Available state command types for discovery
const STATE_COMMAND_TYPES = {
  'windows.create': {
    description: 'Create a new window',
    payloadSchema: {
      title: 'string',
      component: 'string',
      props: 'object (optional)',
      x: 'number (optional)',
      y: 'number (optional)',
      width: 'number (optional)',
      height: 'number (optional)',
      pinned: 'boolean (optional)',
    },
  },
  'windows.close': {
    description: 'Close a window',
    payloadSchema: { id: 'string' },
  },
  'windows.focus': {
    description: 'Focus a window',
    payloadSchema: { id: 'string' },
  },
  'windows.update': {
    description: 'Update window properties',
    payloadSchema: {
      id: 'string',
      title: 'string (optional)',
      x: 'number (optional)',
      y: 'number (optional)',
      width: 'number (optional)',
      height: 'number (optional)',
      props: 'object (optional)',
      pinned: 'boolean (optional)',
    },
  },
  'theme.set': {
    description: 'Set the current theme',
    payloadSchema: { theme: 'string' },
  },
  'theme.setVariable': {
    description: 'Set a theme CSS variable',
    payloadSchema: { variable: 'string', value: 'string' },
  },
  'theme.resetVariable': {
    description: 'Reset a theme CSS variable to default',
    payloadSchema: { variable: 'string' },
  },
  'theme.resetOverrides': {
    description: 'Reset all theme overrides',
    payloadSchema: {},
  },
  'settings.update': {
    description: 'Update a setting',
    payloadSchema: { key: 'string', value: 'unknown' },
  },
  'settings.reset': {
    description: 'Reset all settings to defaults',
    payloadSchema: {},
  },
  'components.add': {
    description: 'Add a component instance',
    payloadSchema: { id: 'string (optional)', type: 'string', props: 'object (optional)' },
  },
  'components.remove': {
    description: 'Remove a component instance',
    payloadSchema: { id: 'string' },
  },
  'components.updateProps': {
    description: 'Update component props',
    payloadSchema: { id: 'string', props: 'object' },
  },
  'projects.create': {
    description: 'Create a new project',
    payloadSchema: { name: 'string' },
  },
  'projects.delete': {
    description: 'Delete a project',
    payloadSchema: { id: 'string' },
  },
  'projects.rename': {
    description: 'Rename a project',
    payloadSchema: { id: 'string', name: 'string' },
  },
  'projects.switch': {
    description: 'Switch to a different project',
    payloadSchema: { id: 'string' },
  },
  'projects.setTheme': {
    description: 'Set project default theme',
    payloadSchema: { id: 'string', theme: 'ThemeState object' },
  },
  'dashboards.create': {
    description: 'Create a new dashboard',
    payloadSchema: { projectId: 'string', name: 'string' },
  },
  'dashboards.delete': {
    description: 'Delete a dashboard',
    payloadSchema: { id: 'string' },
  },
  'dashboards.rename': {
    description: 'Rename a dashboard',
    payloadSchema: { id: 'string', name: 'string' },
  },
  'dashboards.switch': {
    description: 'Switch to a different dashboard',
    payloadSchema: { id: 'string' },
  },
  'dashboards.setThemeOverride': {
    description: 'Set dashboard theme override',
    payloadSchema: { id: 'string', themeOverride: 'ThemeState object or null' },
  },
  'dashboards.saveLayout': {
    description: 'Save current windows to active dashboard',
    payloadSchema: {},
  },
  'globalUI.addSlot': {
    description: 'Add a UI slot',
    payloadSchema: { id: 'string', position: 'object', zIndex: 'number (optional)' },
  },
  'globalUI.removeSlot': {
    description: 'Remove a UI slot',
    payloadSchema: { id: 'string' },
  },
  'globalUI.addWidget': {
    description: 'Add a widget to a slot',
    payloadSchema: { id: 'string (optional)', type: 'string', slot: 'string', props: 'object (optional)', priority: 'number (optional)' },
  },
  'globalUI.removeWidget': {
    description: 'Remove a widget',
    payloadSchema: { id: 'string' },
  },
  'globalUI.updateWidget': {
    description: 'Update widget properties',
    payloadSchema: { id: 'string', props: 'object (optional)', slot: 'string (optional)', priority: 'number (optional)', visible: 'boolean (optional)' },
  },
  'globalUI.setWidgetVisible': {
    description: 'Set widget visibility',
    payloadSchema: { id: 'string', visible: 'boolean' },
  },
  'widgetState.set': {
    description: 'Set widget state',
    payloadSchema: { widgetId: 'string', state: 'unknown' },
  },
  'widgetState.get': {
    description: 'Get widget state',
    payloadSchema: { widgetId: 'string' },
  },
  'widgetState.clear': {
    description: 'Clear widget state',
    payloadSchema: { widgetId: 'string' },
  },
  'customWidgets.register': {
    description: 'Register a new custom widget type with React/JS code',
    payloadSchema: {
      name: 'string - unique widget type name (lowercase with hyphens)',
      description: 'string - widget description',
      category: 'string (optional) - category for grouping',
      code: 'string - React component code',
      defaultProps: 'object (optional) - default props',
      propsSchema: 'object (optional) - props documentation',
    },
  },
  'customWidgets.unregister': {
    description: 'Remove a custom widget type',
    payloadSchema: { name: 'string - widget type name to remove' },
  },
  'customWidgets.update': {
    description: 'Update a custom widget\'s code (hot-reload)',
    payloadSchema: {
      name: 'string - widget type name',
      code: 'string (optional) - new React component code',
      description: 'string (optional) - new description',
      defaultProps: 'object (optional) - new default props',
      propsSchema: 'object (optional) - new props documentation',
    },
  },
  'customWidgets.list': {
    description: 'List all registered custom widgets',
    payloadSchema: {},
  },
  'customWidgets.get': {
    description: 'Get a specific custom widget definition',
    payloadSchema: { name: 'string - widget type name' },
  },
};

/**
 * Setup WebSocket server for external event consumers
 */
export function setupExternalBridge(
  eventStore: EventStore,
  options: ExternalBridgeOptions = {}
): ExternalBridgeHandle {
  const port = options.port ?? 9876;
  const stateStore = options.stateStore;

  const wss = new WebSocketServer({ host: '127.0.0.1', port });
  const clients = new Set<ExtendedWebSocket>();

  console.log(`[events] External bridge listening on ws://127.0.0.1:${port}`);

  wss.on('connection', (ws: ExtendedWebSocket) => {
    clients.add(ws);
    ws.subscriptionPattern = '**'; // Default: receive all events

    console.log('[events] External client connected');

    ws.on('message', (data: Buffer) => {
      try {
        const msg = JSON.parse(data.toString()) as WebSocketMessage;
        handleMessage(ws, msg, eventStore, stateStore);
      } catch (_err) {
        ws.send(
          JSON.stringify({
            type: 'error',
            error: 'Invalid JSON message',
          })
        );
      }
    });

    ws.on('close', () => {
      clients.delete(ws);
      console.log('[events] External client disconnected');
    });

    ws.on('error', (err: Error) => {
      console.error('[events] WebSocket error:', err);
      clients.delete(ws);
    });

    // Send welcome message
    ws.send(
      JSON.stringify({
        type: 'connected',
        sessionId: eventStore.sessionId,
      })
    );
  });

  // Push events to external clients
  const unsubscribe = eventStore.subscribe('**', (event: DashboardEvent) => {
    const message = JSON.stringify({ type: 'event', event });
    for (const ws of clients) {
      if (ws.readyState === WebSocket.OPEN) {
        const pattern = ws.subscriptionPattern ?? '**';
        if (matchesPattern(event.type, pattern)) {
          ws.send(message);
        }
      }
    }
  });

  return {
    close(): void {
      unsubscribe();
      for (const ws of clients) {
        ws.close();
      }
      wss.close();
    },

    getClientCount(): number {
      return clients.size;
    },
  };
}

/**
 * Handle incoming WebSocket message
 */
function handleMessage(
  ws: ExtendedWebSocket,
  msg: WebSocketMessage,
  eventStore: EventStore,
  stateStore?: StateStore
): void {
  switch (msg.type) {
    case 'emit': {
      // Emit event from external source
      if (!msg.eventType) {
        ws.send(JSON.stringify({ type: 'error', error: 'Missing eventType' }));
        return;
      }
      const event = eventStore.emit(msg.eventType, msg.payload ?? {}, {
        source: 'external',
      });
      ws.send(
        JSON.stringify({
          type: 'emit-result',
          requestId: msg.requestId,
          event,
        })
      );
      break;
    }

    case 'query': {
      // Query events
      const events = eventStore.getEvents(msg.filter ?? {});
      ws.send(
        JSON.stringify({
          type: 'query-result',
          requestId: msg.requestId,
          events,
        })
      );
      break;
    }

    case 'subscribe': {
      // Update subscription pattern
      ws.subscriptionPattern = msg.pattern ?? '**';
      ws.send(
        JSON.stringify({
          type: 'subscribed',
          pattern: ws.subscriptionPattern,
        })
      );
      break;
    }

    case 'state:get': {
      // Get state at path
      if (!stateStore) {
        ws.send(JSON.stringify({
          type: 'error',
          requestId: msg.requestId,
          error: 'StateStore not available',
        }));
        return;
      }
      try {
        const state = stateStore.getState(msg.path);
        ws.send(JSON.stringify({
          type: 'state:get-result',
          requestId: msg.requestId,
          state,
        }));
      } catch (err) {
        ws.send(JSON.stringify({
          type: 'error',
          requestId: msg.requestId,
          error: String(err),
        }));
      }
      break;
    }

    case 'state:command': {
      // Execute a state command
      if (!stateStore) {
        ws.send(JSON.stringify({
          type: 'error',
          requestId: msg.requestId,
          error: 'StateStore not available',
        }));
        return;
      }
      if (!msg.commandType) {
        ws.send(JSON.stringify({
          type: 'error',
          requestId: msg.requestId,
          error: 'Missing commandType',
        }));
        return;
      }
      try {
        const result = stateStore.handleCommand(msg.commandType, msg.payload ?? {});
        ws.send(JSON.stringify({
          type: 'state:command-result',
          requestId: msg.requestId,
          result,
        }));
      } catch (err) {
        ws.send(JSON.stringify({
          type: 'error',
          requestId: msg.requestId,
          error: String(err),
        }));
      }
      break;
    }

    case 'state:list-commands': {
      // List available state commands
      ws.send(JSON.stringify({
        type: 'state:list-commands-result',
        requestId: msg.requestId,
        commands: STATE_COMMAND_TYPES,
      }));
      break;
    }

    case 'state:list-widgets': {
      // List available widget types (built-in + custom)
      let allWidgets = { ...WIDGET_TYPES_INFO };

      // Add custom widgets from state if available
      if (stateStore) {
        try {
          const result = stateStore.handleCommand('customWidgets.list', {}) as {
            widgets?: Array<{
              name: string;
              description: string;
              category: string;
              defaultProps: Record<string, unknown>;
              propsSchema: Record<string, string>;
            }>;
          };
          if (result.widgets) {
            for (const widget of result.widgets) {
              allWidgets[widget.name] = {
                description: widget.description,
                category: widget.category,
                defaultProps: widget.defaultProps,
                propsSchema: widget.propsSchema,
              };
            }
          }
        } catch (_err) {
          // Ignore errors - just return built-in widgets
        }
      }

      ws.send(JSON.stringify({
        type: 'state:list-widgets-result',
        requestId: msg.requestId,
        widgets: allWidgets,
      }));
      break;
    }

    case 'subscribe-once': {
      // Subscribe and wait for first matching event
      const pattern = msg.pattern ?? '**';
      const timeout = msg.timeout ?? 30000;

      let resolved = false;
      let timeoutId: NodeJS.Timeout;

      const unsubscribe = eventStore.subscribe(pattern, (event: DashboardEvent) => {
        if (resolved) return;
        resolved = true;
        clearTimeout(timeoutId);
        unsubscribe();
        ws.send(JSON.stringify({
          type: 'subscribe-once-result',
          requestId: msg.requestId,
          event,
        }));
      });

      timeoutId = setTimeout(() => {
        if (resolved) return;
        resolved = true;
        unsubscribe();
        ws.send(JSON.stringify({
          type: 'subscribe-once-result',
          requestId: msg.requestId,
          timedOut: true,
        }));
      }, timeout);
      break;
    }

    case 'code:execute': {
      // Execute JS code - this needs to be forwarded to the renderer
      // For now, emit an event that the renderer can handle
      if (!msg.code) {
        ws.send(JSON.stringify({
          type: 'error',
          requestId: msg.requestId,
          error: 'Missing code parameter',
        }));
        return;
      }

      const executionId = `exec-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

      // Set up listener for result
      let resolved = false;
      const unsubscribe = eventStore.subscribe(`code.result.${executionId}`, (event: DashboardEvent) => {
        if (resolved) return;
        resolved = true;
        unsubscribe();
        const payload = event.payload as { result?: unknown; error?: string };
        ws.send(JSON.stringify({
          type: 'code:execute-result',
          requestId: msg.requestId,
          result: payload.result,
          error: payload.error,
        }));
      });

      // Timeout after 30 seconds
      setTimeout(() => {
        if (resolved) return;
        resolved = true;
        unsubscribe();
        ws.send(JSON.stringify({
          type: 'code:execute-result',
          requestId: msg.requestId,
          error: 'Execution timeout',
        }));
      }, 30000);

      // Emit request event for renderer to pick up
      eventStore.emit('code.execute', {
        executionId,
        code: msg.code,
      }, { source: 'external' });
      break;
    }

    case 'shell:spawn': {
      // Spawn a shell process
      if (!msg.command) {
        ws.send(JSON.stringify({
          type: 'error',
          requestId: msg.requestId,
          error: 'Missing command parameter',
        }));
        return;
      }

      const pid = `proc-${++processIdCounter}`;

      try {
        const proc = spawn(msg.command, msg.args ?? [], {
          cwd: msg.cwd,
          env: msg.env ? { ...process.env, ...msg.env } : process.env,
          shell: true,
        });

        spawnedProcesses.set(pid, proc);

        // Stream stdout as events
        proc.stdout?.on('data', (data: Buffer) => {
          eventStore.emit(`shell.stdout.${pid}`, {
            pid,
            data: data.toString(),
          }, { source: 'shell' });
        });

        // Stream stderr as events
        proc.stderr?.on('data', (data: Buffer) => {
          eventStore.emit(`shell.stderr.${pid}`, {
            pid,
            data: data.toString(),
          }, { source: 'shell' });
        });

        // Handle exit
        proc.on('close', (code: number | null, signal: string | null) => {
          spawnedProcesses.delete(pid);
          eventStore.emit(`shell.exit.${pid}`, {
            pid,
            code,
            signal,
          }, { source: 'shell' });
        });

        proc.on('error', (err: Error) => {
          spawnedProcesses.delete(pid);
          eventStore.emit(`shell.error.${pid}`, {
            pid,
            error: err.message,
          }, { source: 'shell' });
        });

        ws.send(JSON.stringify({
          type: 'shell:spawn-result',
          requestId: msg.requestId,
          pid,
        }));
      } catch (err) {
        ws.send(JSON.stringify({
          type: 'error',
          requestId: msg.requestId,
          error: `Failed to spawn process: ${err instanceof Error ? err.message : String(err)}`,
        }));
      }
      break;
    }

    case 'shell:kill': {
      // Kill a spawned process
      if (!msg.pid) {
        ws.send(JSON.stringify({
          type: 'error',
          requestId: msg.requestId,
          error: 'Missing pid parameter',
        }));
        return;
      }

      const proc = spawnedProcesses.get(msg.pid);
      if (!proc) {
        ws.send(JSON.stringify({
          type: 'shell:kill-result',
          requestId: msg.requestId,
          success: false,
          error: `Process ${msg.pid} not found`,
        }));
        return;
      }

      try {
        const signal = msg.signal ?? 'SIGTERM';
        proc.kill(signal as NodeJS.Signals);
        ws.send(JSON.stringify({
          type: 'shell:kill-result',
          requestId: msg.requestId,
          success: true,
        }));
      } catch (err) {
        ws.send(JSON.stringify({
          type: 'shell:kill-result',
          requestId: msg.requestId,
          success: false,
          error: err instanceof Error ? err.message : String(err),
        }));
      }
      break;
    }

    default:
      ws.send(
        JSON.stringify({
          type: 'error',
          error: `Unknown message type: ${msg.type}`,
        })
      );
  }
}
