import { app, BrowserWindow, ipcMain } from 'electron';
import * as path from 'path';
import * as fs from 'fs';
import { once } from 'hot-reload/api';
import events = require('./events');
import { initServices, setupServiceIPC } from './services';
import { initStateStore, setupStateIPC } from './state';

// Track app state - persists across hot reloads (use let for hot-reload to preserve)
let counter = 0;
let mainWindow: BrowserWindow | null = null;

// Window bounds persistence
interface WindowBounds {
  x?: number;
  y?: number;
  width: number;
  height: number;
  isMaximized?: boolean;
}

const BOUNDS_FILE = path.join(app.getPath('userData'), 'window-bounds.json');

function loadWindowBounds(): WindowBounds {
  try {
    if (fs.existsSync(BOUNDS_FILE)) {
      const data = fs.readFileSync(BOUNDS_FILE, 'utf-8');
      return JSON.parse(data);
    }
  } catch (err) {
    console.error('[window] Failed to load bounds:', err);
  }
  return { width: 1200, height: 800 };
}

function saveWindowBounds(bounds: WindowBounds): void {
  try {
    fs.writeFileSync(BOUNDS_FILE, JSON.stringify(bounds, null, 2));
  } catch (err) {
    console.error('[window] Failed to save bounds:', err);
  }
}

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

// Initialize state store (before external bridge so it can use state)
once(() => {
  const store = initStateStore(events);
  events.setStateStore(store);
});
once(setupStateIPC());

// Setup external bridge with state store access (after state store is set)
once(events.setupExternalBridge({ port: 9876 }));

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
  const bounds = loadWindowBounds();

  mainWindow = new BrowserWindow({
    x: bounds.x,
    y: bounds.y,
    width: bounds.width,
    height: bounds.height,
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 12, y: 12 },
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      webviewTag: true,
    },
  });

  // Restore maximized state
  if (bounds.isMaximized) {
    mainWindow.maximize();
  }

  // Save bounds on resize/move (debounced)
  let saveTimeout: NodeJS.Timeout | null = null;
  const debouncedSave = () => {
    if (saveTimeout) clearTimeout(saveTimeout);
    saveTimeout = setTimeout(() => {
      if (mainWindow && !mainWindow.isDestroyed()) {
        const isMaximized = mainWindow.isMaximized();
        // Only save actual bounds if not maximized
        if (!isMaximized) {
          const currentBounds = mainWindow.getBounds();
          saveWindowBounds({ ...currentBounds, isMaximized: false });
        } else {
          // Just update the maximized flag, keep previous bounds
          const existing = loadWindowBounds();
          saveWindowBounds({ ...existing, isMaximized: true });
        }
      }
    }, 500);
  };

  mainWindow.on('resize', debouncedSave);
  mainWindow.on('move', debouncedSave);
  mainWindow.on('close', () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      const isMaximized = mainWindow.isMaximized();
      if (!isMaximized) {
        const currentBounds = mainWindow.getBounds();
        saveWindowBounds({ ...currentBounds, isMaximized: false });
      }
    }
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
