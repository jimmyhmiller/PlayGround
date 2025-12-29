import { app, BrowserWindow, ipcMain } from 'electron';
import * as path from 'path';
import { once } from 'hot-reload/api';
import events = require('./events');
import { initServices, setupServiceIPC } from './services';
import { initStateStore, setupStateIPC } from './state';

// Track app state - persists across hot reloads (use let for hot-reload to preserve)
let counter = 0;
let mainWindow: BrowserWindow | null = null;

// This function can be hot-reloaded - change it and see updates!
function getMessage(): string {
  return `Hello from the hot-reloadable backend! Counter: ${counter}`;
}

function incrementCounter(): number {
  const oldValue = counter;
  counter += 1;
  events.emit('data.counter.changed', {
    oldValue,
    newValue: counter,
    delta: 1,
  });
  return counter;
}

function getCounter(): number {
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
function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      webviewTag: true,
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
