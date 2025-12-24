import { contextBridge, ipcRenderer, IpcRendererEvent } from 'electron';
import type { DashboardEvent, EventFilter } from '../types/events';
import type { CommandResult } from '../types/state';

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
