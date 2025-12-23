const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  getMessage: () => ipcRenderer.invoke('get-message'),
  increment: () => ipcRenderer.invoke('increment'),
  getCounter: () => ipcRenderer.invoke('get-counter'),
});

// Pattern matching for client-side filtering
function matchesPattern(type, pattern) {
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

// Event API for renderer
contextBridge.exposeInMainWorld('eventAPI', {
  // Emit an event
  emit: (type, payload) => ipcRenderer.invoke('events:emit', type, payload),

  // Query historical events
  query: (filter) => ipcRenderer.invoke('events:query', filter),

  // Get event count
  count: () => ipcRenderer.invoke('events:count'),

  // Subscribe to events matching a pattern
  subscribe: (pattern, callback) => {
    const handler = (ipcEvent, event) => {
      if (matchesPattern(event.type, pattern)) {
        callback(event);
      }
    };
    ipcRenderer.on('events:push', handler);

    // Return unsubscribe function
    return () => {
      ipcRenderer.removeListener('events:push', handler);
    };
  },
});

// File API for renderer
contextBridge.exposeInMainWorld('fileAPI', {
  load: (filePath) => ipcRenderer.invoke('file:load', filePath),
  watch: (watchPath) => ipcRenderer.invoke('file:watch', watchPath),
  unwatch: (watchPath) => ipcRenderer.invoke('file:unwatch', watchPath),
  getWatched: () => ipcRenderer.invoke('file:getWatched'),
});

// Git API for renderer
contextBridge.exposeInMainWorld('gitAPI', {
  refresh: () => ipcRenderer.invoke('git:refresh'),
  status: () => ipcRenderer.invoke('git:status'),
  diff: (filePath) => ipcRenderer.invoke('git:diff', filePath),
  diffStaged: (filePath) => ipcRenderer.invoke('git:diffStaged', filePath),
  startPolling: (intervalMs) => ipcRenderer.invoke('git:startPolling', intervalMs),
  stopPolling: () => ipcRenderer.invoke('git:stopPolling'),
  stage: (filePath) => ipcRenderer.invoke('git:stage', filePath),
  unstage: (filePath) => ipcRenderer.invoke('git:unstage', filePath),
});

// State API for renderer - backend-driven state management
contextBridge.exposeInMainWorld('stateAPI', {
  // Get state at path (or full state if no path)
  get: (path) => ipcRenderer.invoke('state:get', path),

  // Execute a state command
  command: (type, payload) => ipcRenderer.invoke('state:command', type, payload),

  // Subscribe to state changes (uses event system under the hood)
  subscribe: (path, callback) => {
    const pattern = path ? `state.changed.${path}` : 'state.changed';
    const handler = (ipcEvent, event) => {
      if (matchesPattern(event.type, pattern + '.**') || matchesPattern(event.type, pattern)) {
        callback(event);
      }
    };
    ipcRenderer.on('events:push', handler);

    // Return unsubscribe function
    return () => {
      ipcRenderer.removeListener('events:push', handler);
    };
  },
});
