"use strict";

// src/main/preload.ts
var import_electron = require("electron");
function matchesPattern(type, pattern) {
  if (pattern === "*" || pattern === "**") return true;
  if (pattern.endsWith(".**")) {
    const prefix = pattern.slice(0, -3);
    return type === prefix || type.startsWith(prefix + ".");
  }
  if (pattern.endsWith(".*")) {
    const prefix = pattern.slice(0, -2);
    if (!type.startsWith(prefix + ".")) return false;
    const remainder = type.slice(prefix.length + 1);
    return !remainder.includes(".");
  }
  return type === pattern;
}
import_electron.contextBridge.exposeInMainWorld("electronAPI", {
  getMessage: () => import_electron.ipcRenderer.invoke("get-message"),
  increment: () => import_electron.ipcRenderer.invoke("increment"),
  getCounter: () => import_electron.ipcRenderer.invoke("get-counter")
});
import_electron.contextBridge.exposeInMainWorld("eventAPI", {
  // Emit an event
  emit: (type, payload) => import_electron.ipcRenderer.invoke("events:emit", type, payload),
  // Query historical events
  query: (filter) => import_electron.ipcRenderer.invoke("events:query", filter),
  // Get event count
  count: () => import_electron.ipcRenderer.invoke("events:count"),
  // Subscribe to events matching a pattern
  subscribe: (pattern, callback) => {
    const handler = (_ipcEvent, event) => {
      if (matchesPattern(event.type, pattern)) {
        callback(event);
      }
    };
    import_electron.ipcRenderer.on("events:push", handler);
    return () => {
      import_electron.ipcRenderer.removeListener("events:push", handler);
    };
  }
});
import_electron.contextBridge.exposeInMainWorld("fileAPI", {
  load: (filePath) => import_electron.ipcRenderer.invoke("file:load", filePath),
  watch: (watchPath) => import_electron.ipcRenderer.invoke("file:watch", watchPath),
  unwatch: (watchPath) => import_electron.ipcRenderer.invoke("file:unwatch", watchPath),
  getWatched: () => import_electron.ipcRenderer.invoke("file:getWatched")
});
import_electron.contextBridge.exposeInMainWorld("gitAPI", {
  refresh: () => import_electron.ipcRenderer.invoke("git:refresh"),
  status: () => import_electron.ipcRenderer.invoke("git:status"),
  diff: (filePath) => import_electron.ipcRenderer.invoke("git:diff", filePath),
  diffStaged: (filePath) => import_electron.ipcRenderer.invoke("git:diffStaged", filePath),
  startPolling: (intervalMs) => import_electron.ipcRenderer.invoke("git:startPolling", intervalMs),
  stopPolling: () => import_electron.ipcRenderer.invoke("git:stopPolling"),
  stage: (filePath) => import_electron.ipcRenderer.invoke("git:stage", filePath),
  unstage: (filePath) => import_electron.ipcRenderer.invoke("git:unstage", filePath)
});
import_electron.contextBridge.exposeInMainWorld("stateAPI", {
  // Get state at path (or full state if no path)
  get: (path) => import_electron.ipcRenderer.invoke("state:get", path),
  // Execute a state command
  command: (type, payload) => import_electron.ipcRenderer.invoke("state:command", type, payload),
  // Subscribe to state changes (uses event system under the hood)
  subscribe: (path, callback) => {
    const pattern = path ? `state.changed.${path}` : "state.changed";
    const handler = (_ipcEvent, event) => {
      if (matchesPattern(event.type, pattern + ".**") || matchesPattern(event.type, pattern)) {
        callback(event);
      }
    };
    import_electron.ipcRenderer.on("events:push", handler);
    return () => {
      import_electron.ipcRenderer.removeListener("events:push", handler);
    };
  }
});
