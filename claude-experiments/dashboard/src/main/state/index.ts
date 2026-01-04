/**
 * State Module
 *
 * Exports StateStore singleton and IPC setup function.
 */

import { ipcMain, IpcMainInvokeEvent } from 'electron';
import { defonce } from 'hot-reload/api';
import { StateStore } from './StateStore';

// Type for the events module
interface EventEmitter {
  emit(type: string, payload: unknown): void;
}

// Persist state store across hot reloads
const stateStoreRef = defonce<{ current: StateStore | null }>({ current: null });

/**
 * Initialize the state store
 */
export function initStateStore(events: EventEmitter): StateStore {
  if (!stateStoreRef.current) {
    stateStoreRef.current = new StateStore(events);
  }
  return stateStoreRef.current;
}

/**
 * Get the state store instance
 */
export function getStateStore(): StateStore {
  if (!stateStoreRef.current) {
    throw new Error('StateStore not initialized. Call initStateStore first.');
  }
  return stateStoreRef.current;
}

/**
 * Set up IPC handlers for state operations
 */
export function setupStateIPC(): void {
  // Get state at path
  ipcMain.handle('state:get', (_event: IpcMainInvokeEvent, path?: string) => {
    if (!stateStoreRef.current) {
      throw new Error('StateStore not initialized');
    }
    return stateStoreRef.current.getState(path);
  });

  // Execute a command
  ipcMain.handle('state:command', (_event: IpcMainInvokeEvent, type: string, payload: unknown) => {
    if (!stateStoreRef.current) {
      throw new Error('StateStore not initialized');
    }
    return stateStoreRef.current.handleCommand(type, payload);
  });

  // Subscribe to state changes - handled via events:push in the event system
  // The IPC bridge already broadcasts all events to renderers

  console.log('State IPC handlers registered');
}
