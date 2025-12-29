import { contextBridge, ipcRenderer, IpcRendererEvent } from 'electron';
import type { DashboardEvent, EventFilter } from '../types/events';
import type { CommandResult } from '../types/state';
import type { PipelineConfig, PipelineStats, ProcessorDescriptor } from '../types/pipeline';

// Pattern matching for client-side filtering
function matchesPattern(type: string, pattern: string): boolean {
  if (pattern === '*' || pattern === '**') return true;
  if (pattern.endsWith('.**')) {
    const prefix = pattern.slice(0, -3);
    return type === prefix || type.startsWith(prefix + '.');
  }
  if (pattern.endsWith('.*')) {
    const prefix = pattern.slice(0, -2);
    if (!type.startsWith(prefix + '.')) return false;
    const remainder = type.slice(prefix.length + 1);
    return !remainder.includes('.');
  }
  return type === pattern;
}

// Legacy Electron API
contextBridge.exposeInMainWorld('electronAPI', {
  getMessage: (): Promise<string> => ipcRenderer.invoke('get-message'),
  increment: (): Promise<number> => ipcRenderer.invoke('increment'),
  getCounter: (): Promise<number> => ipcRenderer.invoke('get-counter'),
});

// Event API for renderer
contextBridge.exposeInMainWorld('eventAPI', {
  // Emit an event
  emit: (type: string, payload?: unknown): Promise<DashboardEvent> =>
    ipcRenderer.invoke('events:emit', type, payload),

  // Query historical events
  query: (filter?: EventFilter): Promise<DashboardEvent[]> =>
    ipcRenderer.invoke('events:query', filter),

  // Get event count
  count: (): Promise<number> => ipcRenderer.invoke('events:count'),

  // Subscribe to events matching a pattern
  subscribe: (pattern: string, callback: (event: DashboardEvent) => void): (() => void) => {
    const handler = (_ipcEvent: IpcRendererEvent, event: DashboardEvent): void => {
      if (matchesPattern(event.type, pattern)) {
        callback(event);
      }
    };
    ipcRenderer.on('events:push', handler);

    // Return unsubscribe function
    return (): void => {
      ipcRenderer.removeListener('events:push', handler);
    };
  },
});

// File API for renderer
contextBridge.exposeInMainWorld('fileAPI', {
  load: (filePath: string): Promise<{ success: boolean; content?: string; error?: string }> =>
    ipcRenderer.invoke('file:load', filePath),
  watch: (watchPath: string): Promise<{ success: boolean; path: string }> =>
    ipcRenderer.invoke('file:watch', watchPath),
  unwatch: (watchPath: string): Promise<{ success: boolean }> =>
    ipcRenderer.invoke('file:unwatch', watchPath),
  getWatched: (): Promise<string[]> => ipcRenderer.invoke('file:getWatched'),
});

// Git API for renderer
contextBridge.exposeInMainWorld('gitAPI', {
  refresh: (): Promise<unknown> => ipcRenderer.invoke('git:refresh'),
  status: (): Promise<{ files: Array<{ status: string; path: string }>; branch: string | null }> =>
    ipcRenderer.invoke('git:status'),
  diff: (filePath?: string): Promise<string> => ipcRenderer.invoke('git:diff', filePath),
  diffStaged: (filePath?: string): Promise<string> => ipcRenderer.invoke('git:diffStaged', filePath),
  startPolling: (intervalMs?: number): Promise<{ success: boolean }> =>
    ipcRenderer.invoke('git:startPolling', intervalMs),
  stopPolling: (): Promise<{ success: boolean }> => ipcRenderer.invoke('git:stopPolling'),
  stage: (filePath: string): Promise<{ success: boolean }> => ipcRenderer.invoke('git:stage', filePath),
  unstage: (filePath: string): Promise<{ success: boolean }> => ipcRenderer.invoke('git:unstage', filePath),
});

// Evaluation API for renderer - code execution service
contextBridge.exposeInMainWorld('evalAPI', {
  // Execute a single code snippet
  execute: (
    code: string,
    language: string = 'javascript',
    context?: Record<string, unknown>
  ): Promise<{
    id: string;
    success: boolean;
    value?: unknown;
    displayValue: string;
    type: string;
    executionTimeMs: number;
    error?: string;
  }> => ipcRenderer.invoke('eval:execute', code, language, context),

  // Execute multiple code snippets
  batch: (
    requests: Array<{
      id: string;
      code: string;
      language: string;
      context?: Record<string, unknown>;
      timeout?: number;
    }>
  ): Promise<
    Array<{
      id: string;
      success: boolean;
      value?: unknown;
      displayValue: string;
      type: string;
      executionTimeMs: number;
      error?: string;
    }>
  > => ipcRenderer.invoke('eval:batch', requests),

  // Register a subprocess executor for a language
  registerExecutor: (config: {
    language: string;
    command: string;
    args?: string[];
    cwd?: string;
  }): Promise<{ success: boolean; language: string }> =>
    ipcRenderer.invoke('eval:registerExecutor', config),

  // Unregister an executor
  unregisterExecutor: (language: string): Promise<{ success: boolean }> =>
    ipcRenderer.invoke('eval:unregisterExecutor', language),

  // Get list of registered executors
  getExecutors: (): Promise<
    Array<{
      language: string;
      command: string;
      args?: string[];
      cwd?: string;
    }>
  > => ipcRenderer.invoke('eval:getExecutors'),
});

// Shell API for renderer - spawn and manage processes
contextBridge.exposeInMainWorld('shellAPI', {
  // Spawn a new process
  spawn: (
    id: string,
    command: string,
    args: string[] = [],
    options: { cwd?: string; env?: Record<string, string> } = {}
  ): Promise<{ success: boolean; id: string; pid?: number }> =>
    ipcRenderer.invoke('shell:spawn', id, command, args, options),

  // Kill a running process
  kill: (id: string): Promise<{ success: boolean; id?: string; error?: string }> =>
    ipcRenderer.invoke('shell:kill', id),

  // Check if a process is running
  isRunning: (id: string): Promise<{ running: boolean }> =>
    ipcRenderer.invoke('shell:isRunning', id),
});

// State API for renderer - backend-driven state management
contextBridge.exposeInMainWorld('stateAPI', {
  // Get state at path (or full state if no path)
  get: (path?: string): Promise<unknown> => ipcRenderer.invoke('state:get', path),

  // Execute a state command
  command: (type: string, payload?: unknown): Promise<CommandResult> =>
    ipcRenderer.invoke('state:command', type, payload),

  // Subscribe to state changes (uses event system under the hood)
  subscribe: (path: string, callback: (event: DashboardEvent) => void): (() => void) => {
    const pattern = path ? `state.changed.${path}` : 'state.changed';
    const handler = (_ipcEvent: IpcRendererEvent, event: DashboardEvent): void => {
      if (matchesPattern(event.type, pattern + '.**') || matchesPattern(event.type, pattern)) {
        callback(event);
      }
    };
    ipcRenderer.on('events:push', handler);

    // Return unsubscribe function
    return (): void => {
      ipcRenderer.removeListener('events:push', handler);
    };
  },
});

// Pipeline API for renderer - Unix-pipes style data processing
contextBridge.exposeInMainWorld('pipelineAPI', {
  // Start a pipeline
  start: (config: PipelineConfig): Promise<{ success: boolean; error?: string }> =>
    ipcRenderer.invoke('pipeline:start', config),

  // Stop a pipeline
  stop: (id: string): Promise<{ success: boolean; error?: string }> =>
    ipcRenderer.invoke('pipeline:stop', id),

  // Get pipeline stats
  stats: (id: string): Promise<PipelineStats | undefined> =>
    ipcRenderer.invoke('pipeline:stats', id),

  // Check if pipeline is running
  isRunning: (id: string): Promise<{ running: boolean }> =>
    ipcRenderer.invoke('pipeline:isRunning', id),

  // List running pipeline IDs
  list: (): Promise<string[]> => ipcRenderer.invoke('pipeline:list'),

  // List running pipelines with details
  listDetailed: (): Promise<
    Array<{ id: string; config: PipelineConfig; stats: PipelineStats }>
  > => ipcRenderer.invoke('pipeline:listDetailed'),

  // Stop all pipelines
  stopAll: (): Promise<{ success: boolean }> => ipcRenderer.invoke('pipeline:stopAll'),

  // List available processor names
  processors: (): Promise<string[]> => ipcRenderer.invoke('pipeline:processors'),

  // Describe all processors (for LLM discovery)
  describeProcessors: (): Promise<ProcessorDescriptor[]> =>
    ipcRenderer.invoke('pipeline:describeProcessors'),
});
