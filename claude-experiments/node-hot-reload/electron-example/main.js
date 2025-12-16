import { once, defonce } from 'hot-reload/api';
import { app, BrowserWindow, ipcMain } from 'electron';
import * as path from 'path';

// State preserved across hot reloads
const state = defonce({
  mainWindow: null,
  clickCount: 0,
});

// Create window function - can be hot-reloaded
function createWindow() {
  if (state.mainWindow) return state.mainWindow;

  state.mainWindow = new BrowserWindow({
    width: 600,
    height: 400,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
    },
  });

  state.mainWindow.loadFile('index.html');

  state.mainWindow.on('closed', () => {
    state.mainWindow = null;
  });

  return state.mainWindow;
}

// These functions can be hot-reloaded!
function getGreeting() {
  return `✨ CHANGED AGAIN! Total clicks: ${state.clickCount} ✨`;
}

function handleClick() {
  state.clickCount += 10;
  return getGreeting();
}

function getStatus() {
  return `Status: Running | Clicks: ${state.clickCount}`;
}

// Register IPC handlers only once
once(ipcMain.handle('get-greeting', () => getGreeting()));
once(ipcMain.handle('handle-click', () => handleClick()));
once(ipcMain.handle('get-status', () => getStatus()));

// App lifecycle - only once
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

// This logs on every hot-reload so you can see it working
console.log('[main] Module loaded/reloaded');
