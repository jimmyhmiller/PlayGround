/**
 * State Module
 *
 * Exports StateStore singleton and IPC setup function.
 */

const { ipcMain, BrowserWindow } = require('electron');
const { StateStore } = require('./StateStore');

let stateStore = null;

/**
 * Initialize the state store
 * @param {object} events - The event system
 * @returns {StateStore} The state store instance
 */
function initStateStore(events) {
  if (!stateStore) {
    stateStore = new StateStore(events);
  }
  return stateStore;
}

/**
 * Get the state store instance
 * @returns {StateStore} The state store instance
 */
function getStateStore() {
  if (!stateStore) {
    throw new Error('StateStore not initialized. Call initStateStore first.');
  }
  return stateStore;
}

/**
 * Set up IPC handlers for state operations
 */
function setupStateIPC() {
  // Get state at path
  ipcMain.handle('state:get', (event, path) => {
    return stateStore.getState(path);
  });

  // Execute a command
  ipcMain.handle('state:command', (event, type, payload) => {
    return stateStore.handleCommand(type, payload);
  });

  // Subscribe to state changes - handled via events:push in the event system
  // The IPC bridge already broadcasts all events to renderers

  console.log('State IPC handlers registered');
}

module.exports = {
  initStateStore,
  getStateStore,
  setupStateIPC,
};
