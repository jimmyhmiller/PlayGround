/**
 * Event System Types
 *
 * Type definitions for the event sourcing system
 */

/**
 * Event metadata
 */
export interface EventMeta {
  sessionId: string;
  source: 'main' | 'renderer' | 'external';
  version: number;
  windowId?: string;
  correlationId?: string;
}

/**
 * Base event structure
 */
export interface DashboardEvent<T = unknown> {
  id: string;
  type: string;
  payload: T;
  timestamp: number;
  meta: EventMeta;
}

/**
 * Event filter for queries
 */
export interface EventFilter {
  type?: string;
  since?: number;
  until?: number;
  limit?: number;
  correlationId?: string;
}

/**
 * Event handler callback type
 */
export type EventHandler<T = unknown> = (event: DashboardEvent<T>) => void;

/**
 * Unsubscribe function type
 */
export type Unsubscribe = () => void;

/**
 * Persistence backend interface
 */
export interface EventPersistence {
  append(event: DashboardEvent): void;
  getEvents(filter?: EventFilter): DashboardEvent[];
  clear(): void;
}

/**
 * EventStore options
 */
export interface EventStoreOptions {
  maxInMemory?: number;
  persistence?: EventPersistence | null;
}

/**
 * EventStore interface
 */
export interface IEventStore {
  emit<T = unknown>(type: string, payload?: T, meta?: Partial<EventMeta>): DashboardEvent<T>;
  subscribe(pattern: string, callback: EventHandler): Unsubscribe;
  getEvents(filter?: EventFilter): DashboardEvent[];
  count(): number;
  clear(): void;
  setPersistence(persistence: EventPersistence | null): void;
}

// State change event payload
export interface StateChangePayload {
  path: string;
  value: unknown;
}
