import { once, defonce } from 'hot-reload/api';

// Mock ipcMain - state preserved across reloads
const handlers = defonce(new Map());
const ipcMain = {
  handle(name, fn) {
    handlers.set(name, fn);
    console.log(`[ipc] Registered handler: ${name}`);
  }
};

// State preserved across reloads
let counter = defonce(0);

// Functions - always hot-reloadable
function getCounter() {
  return counter;
}

function increment() {
  counter++;
  return counter;
}

function getMessage() {
  return `Counter is ${getCounter()} - Hello from hot reload!`;
}

// Register handlers only once
once(ipcMain.handle('get-counter', () => getCounter()));
once(ipcMain.handle('increment', () => increment()));
once(ipcMain.handle('get-message', () => getMessage()));

// This runs every reload (for demo)
console.log('[app] Module loaded/reloaded');

// Start interval only once
once(setInterval(() => {
  const handler = handlers.get('get-message');
  if (handler) {
    console.log(`[app] ${handler()}`);
  }
}, 2000));
