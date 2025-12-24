/**
 * State Module
 *
 * Exports StateStore singleton and IPC setup function.
 */

import { ipcMain, IpcMainInvokeEvent } from 'electron';
import { StateStore } from './StateStore';

// Type for the events module
interface EventEmitter {
  emit(type: string, payload: unknown): void;
}

let stateStore: StateStore | null = null;

/**
 * Initialize the state store
 */
export function initStateStore(events: EventEmitter): StateStore {
  if (!stateStore) {
    stateStore = new StateStore(events);
  }
  return stateStore;
}

/**
 * Get the state store instance
 */
export function getStateStore(): StateStore {
  if (!stateStore) {
    throw new Error('StateStore not initialized. Call initStateStore first.');
  }
  return stateStore;
}

/**
 * Set up IPC handlers for state operations
 */
export function setupStateIPC(): void {
  // Get state at path
  ipcMain.handle('state:get', (_event: IpcMainInvokeEvent, path?: string) => {
    return stateStore!.getState(path);
  });

  // Execute a command
  ipcMain.handle('state:command', (_event: IpcMainInvokeEvent, type: string, payload: unknown) => {
    return stateStore!.handleCommand(type, payload);
  });

  // Subscribe to state changes - handled via events:push in the event system
  // The IPC bridge already broadcasts all events to renderers

  console.log('State IPC handlers registered');
}
