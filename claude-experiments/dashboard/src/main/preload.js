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
import_electron.contextBridge.exposeInMainWorld("evalAPI", {
  // Execute a single code snippet
  execute: (code, language = "javascript", context) => import_electron.ipcRenderer.invoke("eval:execute", code, language, context),
  // Execute multiple code snippets
  batch: (requests) => import_electron.ipcRenderer.invoke("eval:batch", requests),
  // Register a subprocess executor for a language
  registerExecutor: (config) => import_electron.ipcRenderer.invoke("eval:registerExecutor", config),
  // Unregister an executor
  unregisterExecutor: (language) => import_electron.ipcRenderer.invoke("eval:unregisterExecutor", language),
  // Get list of registered executors
  getExecutors: () => import_electron.ipcRenderer.invoke("eval:getExecutors")
});
import_electron.contextBridge.exposeInMainWorld("shellAPI", {
  // Spawn a new process
  spawn: (id, command, args = [], options = {}) => import_electron.ipcRenderer.invoke("shell:spawn", id, command, args, options),
  // Kill a running process
  kill: (id) => import_electron.ipcRenderer.invoke("shell:kill", id),
  // Check if a process is running
  isRunning: (id) => import_electron.ipcRenderer.invoke("shell:isRunning", id)
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
import_electron.contextBridge.exposeInMainWorld("pipelineAPI", {
  // Start a pipeline
  start: (config) => import_electron.ipcRenderer.invoke("pipeline:start", config),
  // Stop a pipeline
  stop: (id) => import_electron.ipcRenderer.invoke("pipeline:stop", id),
  // Get pipeline stats
  stats: (id) => import_electron.ipcRenderer.invoke("pipeline:stats", id),
  // Check if pipeline is running
  isRunning: (id) => import_electron.ipcRenderer.invoke("pipeline:isRunning", id),
  // List running pipeline IDs
  list: () => import_electron.ipcRenderer.invoke("pipeline:list"),
  // List running pipelines with details
  listDetailed: () => import_electron.ipcRenderer.invoke("pipeline:listDetailed"),
  // Stop all pipelines
  stopAll: () => import_electron.ipcRenderer.invoke("pipeline:stopAll"),
  // List available processor names
  processors: () => import_electron.ipcRenderer.invoke("pipeline:processors"),
  // Describe all processors (for LLM discovery)
  describeProcessors: () => import_electron.ipcRenderer.invoke("pipeline:describeProcessors")
});
