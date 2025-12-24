/**
 * IPC Bridge for Electron
 *
 * Connects the EventStore to Electron's IPC system.
 * Allows renderer processes to emit events, subscribe, and query.
 * Broadcasts events to all windows.
 */

import { ipcMain, BrowserWindow, IpcMainInvokeEvent } from 'electron';
import type { EventStore } from '../EventStore';
import type { EventFilter } from '../../../types/events';

/**
 * Setup IPC handlers for the event system
 */
export function setupEventIPC(eventStore: EventStore): void {
  // Emit event from renderer
  ipcMain.handle(
    'events:emit',
    (ipcEvent: IpcMainInvokeEvent, type: string, payload: unknown) => {
      const windowId = BrowserWindow.fromWebContents(ipcEvent.sender)?.id;
      return eventStore.emit(type, payload, {
        source: 'renderer',
        windowId: windowId?.toString(),
      });
    }
  );

  // Query events from renderer
  ipcMain.handle(
    'events:query',
    (_ipcEvent: IpcMainInvokeEvent, filter: EventFilter) => {
      return eventStore.getEvents(filter);
    }
  );

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
