/**
 * Event Sourcing API
 *
 * Main entry point for the event system.
 * Exports the public API for emitting, subscribing, and querying events.
 */

import { EventStore } from './EventStore';
import { createCommandHandler, CommandHandlerFactory } from './commands';
import { setupEventIPC } from './bridges/ipcBridge';
import { setupExternalBridge } from './bridges/externalBridge';
import { matchesPattern } from './utils';
import type {
  DashboardEvent,
  EventMeta,
  EventFilter,
  EventHandler,
  EventPersistence,
  Unsubscribe,
} from '../../types/events';

// Create singleton event store
const eventStore = new EventStore({
  maxInMemory: 10000,
});

// Create command handler
const commandHandler: CommandHandlerFactory = createCommandHandler(eventStore);

// Track bridge handles for cleanup
interface ExternalBridgeHandle {
  close(): void;
  getClientCount(): number;
}
let externalBridgeHandle: ExternalBridgeHandle | null = null;

interface ExternalBridgeOptions {
  port?: number;
}

const events = {
  /**
   * Emit an event
   */
  emit: <T = unknown>(
    type: string,
    payload?: T,
    meta?: Partial<EventMeta>
  ): DashboardEvent<T> => eventStore.emit(type, payload, meta),

  /**
   * Subscribe to events matching a pattern
   */
  subscribe: (pattern: string, callback: EventHandler): Unsubscribe =>
    eventStore.subscribe(pattern, callback),

  /**
   * Query historical events
   */
  getEvents: (filter?: EventFilter): DashboardEvent[] => eventStore.getEvents(filter),

  /**
   * Execute a command (returns helper to emit success/failure events)
   */
  command: (commandType: string, payload?: unknown, meta?: Partial<EventMeta>) =>
    commandHandler.execute(commandType, payload, meta),

  /**
   * Setup IPC handlers for renderer communication
   * Call this once during app initialization (use once() wrapper)
   */
  setupIPC: (): void => setupEventIPC(eventStore),

  /**
   * Setup WebSocket bridge for external processes
   */
  setupExternalBridge: (options?: ExternalBridgeOptions): ExternalBridgeHandle => {
    externalBridgeHandle = setupExternalBridge(eventStore, options);
    return externalBridgeHandle;
  },

  /**
   * Set persistence backend
   */
  setPersistence: (persistence: EventPersistence | null): void =>
    eventStore.setPersistence(persistence),

  /**
   * Get event count
   */
  count: (): number => eventStore.count(),

  /**
   * Clear all events
   */
  clear: (): void => eventStore.clear(),

  /**
   * Get session ID
   */
  getSessionId: (): string => eventStore.sessionId,

  /**
   * Access underlying store (for advanced use cases)
   */
  store: eventStore,

  /**
   * Utility: pattern matching function
   */
  matchesPattern,
};

export = events;
