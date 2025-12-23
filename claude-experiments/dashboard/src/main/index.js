const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { once } = require('hot-reload/api');
const events = require('./events');
const { initServices, setupServiceIPC } = require('./services');
const { initStateStore, setupStateIPC } = require('./state');

// Track app state - persists across hot reloads (use let for hot-reload to preserve)
let counter = 0;
let mainWindow = null;

// This function can be hot-reloaded - change it and see updates!
function getMessage() {
  return `Hello from the hot-reloadable backend! Counter: ${counter}`;
}

function incrementCounter() {
  const oldValue = counter;
  counter += 1;
  events.emit('data.counter.changed', {
    oldValue,
    newValue: counter,
    delta: 1,
  });
  return counter;
}

function getCounter() {
  return counter;
}

// IPC handlers - wrapped in once() so they only register on first load
once(ipcMain.handle('get-message', getMessage));

once(ipcMain.handle('increment', incrementCounter));

once(ipcMain.handle('get-counter', getCounter));

// Initialize event system
once(events.setupIPC());
once(events.setupExternalBridge({ port: 9876 }));

// Initialize state store
once(initStateStore(events));
once(setupStateIPC());

// Initialize services (file watcher, git)
once(initServices(events, { repoPath: process.cwd() }));
once(setupServiceIPC());

// Emit app startup event
events.emit('system.app.started', {
  version: app.getVersion(),
  platform: process.platform,
});

// Create window
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // In development, load from Vite dev server
  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadURL('http://localhost:5173');
  } else {
    mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
  }
}

once(app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
}));

once(app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
}));

console.log('Main process loaded/reloaded!');
