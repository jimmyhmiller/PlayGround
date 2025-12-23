/**
 * IPC Bridge for Electron
 *
 * Connects the EventStore to Electron's IPC system.
 * Allows renderer processes to emit events, subscribe, and query.
 * Broadcasts events to all windows.
 */

const { ipcMain, BrowserWindow } = require('electron');

/**
 * Setup IPC handlers for the event system
 *
 * @param {EventStore} eventStore
 */
function setupEventIPC(eventStore) {
  // Emit event from renderer
  ipcMain.handle('events:emit', (ipcEvent, type, payload) => {
    const windowId = BrowserWindow.fromWebContents(ipcEvent.sender)?.id;
    return eventStore.emit(type, payload, {
      source: 'renderer',
      windowId,
    });
  });

  // Query events from renderer
  ipcMain.handle('events:query', (ipcEvent, filter) => {
    return eventStore.getEvents(filter);
  });

  // Get event count
  ipcMain.handle('events:count', () => {
    return eventStore.count();
  });

  // Subscribe to all events and push to all windows
  eventStore.subscribe('**', (event) => {
    const windows = BrowserWindow.getAllWindows();
    for (const win of windows) {
      if (!win.isDestroyed() && win.webContents) {
        win.webContents.send('events:push', event);
      }
    }
  });
}

module.exports = { setupEventIPC };
