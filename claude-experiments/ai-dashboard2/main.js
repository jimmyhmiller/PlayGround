const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const util = require('util');
const { query } = require('@anthropic-ai/claude-agent-sdk');

const isDev = process.env.NODE_ENV === 'development';

let mainWindow = null;
const watchedPaths = new Map(); // path -> { watcher, dashboard, lastWriteTime }
const configFilePath = path.join(app.getPath('userData'), 'dashboard-paths.json');

// Load saved config paths
function loadSavedPaths() {
  try {
    if (fs.existsSync(configFilePath)) {
      const data = JSON.parse(fs.readFileSync(configFilePath, 'utf-8'));
      return data.paths || [];
    }
  } catch (e) {
    console.error('Error loading saved paths:', e);
  }
  return [];
}

// Save config paths
function savePaths() {
  try {
    const paths = Array.from(watchedPaths.keys());
    fs.writeFileSync(configFilePath, JSON.stringify({ paths }, null, 2));
  } catch (e) {
    console.error('Error saving paths:', e);
  }
}

// Parse a dashboard JSON file
function parseDashboardFile(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const dashboard = JSON.parse(content);
    // Add source path for identification
    dashboard._sourcePath = filePath;
    return dashboard;
  } catch (e) {
    console.error(`Error parsing ${filePath}:`, e);
    return null;
  }
}

// Watch a file for changes
function watchFile(filePath) {
  if (watchedPaths.has(filePath)) return;

  const dashboard = parseDashboardFile(filePath);
  if (!dashboard) return;

  // Use fs.watchFile instead of fs.watch to handle atomic writes (temp file + rename)
  // This is common with editors like VS Code, Claude Code, etc.
  const watcher = fs.watchFile(filePath, { interval: 100 }, (curr, prev) => {
    // Check if the file was actually modified (not just accessed)
    if (curr.mtime > prev.mtime) {
      const entry = watchedPaths.get(filePath);
      if (!entry) return;

      // Check if this change was made by us (within 1 second of our last write)
      const timeSinceWrite = curr.mtime.getTime() - (entry.lastWriteTime || 0);
      if (timeSinceWrite < 1000 && entry.lastWriteTime) {
        console.log(`[FileWatcher] Ignoring self-triggered change: ${filePath} (${timeSinceWrite}ms since write)`);
        return;
      }

      console.log(`[FileWatcher] External file modification detected: ${filePath}`);
      const updated = parseDashboardFile(filePath);
      if (updated) {
        console.log(`[FileWatcher] Successfully reloaded ${filePath}`);
        entry.dashboard = updated;
        broadcastDashboards();
      } else {
        console.error(`[FileWatcher] Failed to parse ${filePath}`);
      }
    }
  });

  watchedPaths.set(filePath, { watcher, dashboard, lastWriteTime: null });
}

// Stop watching a file
function unwatchFile(filePath) {
  const entry = watchedPaths.get(filePath);
  if (entry) {
    fs.unwatchFile(filePath); // Use unwatchFile for fs.watchFile
    watchedPaths.delete(filePath);
  }
}

// Get all loaded dashboards
function getAllDashboards() {
  return Array.from(watchedPaths.values())
    .map(entry => entry.dashboard)
    .filter(Boolean);
}

// Broadcast dashboards to renderer
function broadcastDashboards() {
  const dashboards = getAllDashboards();
  console.log(`[Broadcast] Sending ${dashboards.length} dashboards to renderer`);
  if (mainWindow) {
    mainWindow.webContents.send('dashboard-updated', dashboards);
  } else {
    console.warn('[Broadcast] No main window available');
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 950,
    height: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    backgroundColor: '#050505',
    titleBarStyle: 'hiddenInset',
  });

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
  } else {
    mainWindow.loadFile(path.join(__dirname, 'dist', 'index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// IPC Handlers
ipcMain.handle('load-dashboards', () => {
  return getAllDashboards();
});

ipcMain.handle('add-config-path', (event, filePath) => {
  if (!fs.existsSync(filePath)) {
    return { success: false, error: 'File does not exist' };
  }
  watchFile(filePath);
  savePaths();
  broadcastDashboards();
  return { success: true };
});

ipcMain.handle('remove-config-path', (event, filePath) => {
  unwatchFile(filePath);
  savePaths();
  broadcastDashboards();
  return { success: true };
});

ipcMain.handle('get-watched-paths', () => {
  return Array.from(watchedPaths.keys());
});

ipcMain.handle('update-widget-dimensions', async (event, { dashboardId, widgetId, dimensions }) => {
  try {
    // Find the dashboard in memory
    let targetPath = null;
    let entry = null;
    for (const [path, e] of watchedPaths.entries()) {
      if (e.dashboard.id === dashboardId) {
        targetPath = path;
        entry = e;
        break;
      }
    }

    if (!targetPath || !entry) {
      return { success: false, error: 'Dashboard not found' };
    }

    // Update the in-memory dashboard
    const widget = entry.dashboard.widgets.find(w => w.id === widgetId);
    if (!widget) {
      return { success: false, error: 'Widget not found' };
    }

    // Only update the changed properties
    if (dimensions.width !== undefined) widget.width = dimensions.width;
    if (dimensions.height !== undefined) widget.height = dimensions.height;
    if (dimensions.x !== undefined) widget.x = dimensions.x;
    if (dimensions.y !== undefined) widget.y = dimensions.y;

    // Record the time of this write so we can ignore the file watcher event
    entry.lastWriteTime = Date.now();

    // Write the updated in-memory dashboard to file
    fs.writeFileSync(targetPath, JSON.stringify(entry.dashboard, null, 2), 'utf-8');

    // DON'T broadcast - the frontend already has this state from the drag action
    // Broadcasting would cause unnecessary re-renders and state resets

    return { success: true };
  } catch (error) {
    console.error('Error updating widget dimensions:', error);
    return { success: false, error: error.message };
  }
});

// Claude Agent SDK Chat Handlers
const activeChats = new Map(); // chatId -> { query, abortController }

ipcMain.handle('claude-chat-send', async (event, { chatId, message, options = {} }) => {
  try {
    const responseId = `${chatId}-${Date.now()}`;

    // Start streaming in the background - don't await it
    (async () => {
      try {
        console.log(`[Claude] Starting chat for ${chatId} with message: ${message.substring(0, 50)}...`);

        // Create query with streaming enabled
        const result = query({
          prompt: message,
          options: {
            model: options.model || 'claude-sonnet-4-5-20250929',
            cwd: options.cwd || process.cwd(),
            ...options
          }
        });

        // Store the query so we can interrupt it if needed
        activeChats.set(chatId, { query: result });

        // Stream messages back to renderer
        for await (const msg of result) {
          console.log(`[Claude] Received message type: ${msg.type}`);
          console.log(util.inspect(msg, { depth: null, colors: true }));

          // Send each message chunk to the renderer
          mainWindow?.webContents.send('claude-chat-message', {
            chatId,
            responseId,
            message: msg
          });
        }

        console.log(`[Claude] Streaming complete for ${chatId}`);

        // Signal completion
        mainWindow?.webContents.send('claude-chat-complete', {
          chatId,
          responseId
        });

        // Clean up when done
        activeChats.delete(chatId);
      } catch (streamError) {
        console.error('[Claude] Streaming error:', streamError);
        mainWindow?.webContents.send('claude-chat-error', {
          chatId,
          responseId,
          error: streamError.message
        });
        activeChats.delete(chatId);
      }
    })();

    // Return immediately to not block the IPC call
    return { success: true, responseId };
  } catch (error) {
    console.error('[Claude] Handler error:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('claude-chat-interrupt', async (event, { chatId }) => {
  const chat = activeChats.get(chatId);
  if (chat?.query) {
    try {
      await chat.query.interrupt();
      activeChats.delete(chatId);
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }
  return { success: false, error: 'No active chat found' };
});

app.whenReady().then(() => {
  const savedPaths = loadSavedPaths();
  savedPaths.forEach(watchFile);
  createWindow();
});

app.on('window-all-closed', () => {
  // On macOS, apps typically stay open even when all windows are closed
  if (process.platform !== 'darwin') {
    // Clean up watchers on non-macOS platforms
    watchedPaths.forEach((entry) => entry.watcher.close());
    app.quit();
  }
});

// On macOS, recreate window when clicking dock icon while app is running
app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});
