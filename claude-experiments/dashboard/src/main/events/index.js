/**
 * Event Sourcing API
 *
 * Main entry point for the event system.
 * Exports the public API for emitting, subscribing, and querying events.
 */

const { EventStore } = require('./EventStore');
const { createCommandHandler } = require('./commands');
const { setupEventIPC } = require('./bridges/ipcBridge');
const { setupExternalBridge } = require('./bridges/externalBridge');
const { matchesPattern } = require('./utils');

// Create singleton event store
const eventStore = new EventStore({
  maxInMemory: 10000,
});

// Create command handler
const commandHandler = createCommandHandler(eventStore);

// Track bridge handles for cleanup
let externalBridgeHandle = null;

module.exports = {
  /**
   * Emit an event
   * @param {string} type - Event type (e.g., "user.button.clicked")
   * @param {Object} payload - Event payload
   * @param {Object} meta - Additional metadata
   * @returns {Object} The created event
   */
  emit: (type, payload, meta) => eventStore.emit(type, payload, meta),

  /**
   * Subscribe to events matching a pattern
   * @param {string} pattern - Pattern to match (e.g., "user.**", "*")
   * @param {Function} callback - Called with each matching event
   * @returns {Function} Unsubscribe function
   */
  subscribe: (pattern, callback) => eventStore.subscribe(pattern, callback),

  /**
   * Query historical events
   * @param {Object} filter - Filter criteria (type, since, until, limit, correlationId)
   * @returns {Array} Matching events
   */
  getEvents: (filter) => eventStore.getEvents(filter),

  /**
   * Execute a command (returns helper to emit success/failure events)
   * @param {string} commandType - Command type
   * @param {Object} payload - Command payload
   * @param {Object} meta - Additional metadata
   * @returns {Object} Handler with correlationId, success(), failure()
   */
  command: (commandType, payload, meta) =>
    commandHandler.execute(commandType, payload, meta),

  /**
   * Setup IPC handlers for renderer communication
   * Call this once during app initialization (use once() wrapper)
   */
  setupIPC: () => setupEventIPC(eventStore),

  /**
   * Setup WebSocket bridge for external processes
   * @param {Object} options - { port: number }
   * @returns {Object} Bridge handle with close() method
   */
  setupExternalBridge: (options) => {
    externalBridgeHandle = setupExternalBridge(eventStore, options);
    return externalBridgeHandle;
  },

  /**
   * Set persistence backend
   * @param {Object} persistence - Backend with append(event) method
   */
  setPersistence: (persistence) => eventStore.setPersistence(persistence),

  /**
   * Get event count
   * @returns {number}
   */
  count: () => eventStore.count(),

  /**
   * Clear all events
   */
  clear: () => eventStore.clear(),

  /**
   * Get session ID
   * @returns {string}
   */
  getSessionId: () => eventStore.sessionId,

  /**
   * Access underlying store (for advanced use cases)
   */
  store: eventStore,

  /**
   * Utility: pattern matching function
   */
  matchesPattern,
};
