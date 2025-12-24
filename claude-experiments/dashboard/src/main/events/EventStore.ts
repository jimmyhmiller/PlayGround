/**
 * EventStore - Core event sourcing store
 *
 * Stores events in memory with optional persistence backend.
 * Supports subscriptions with pattern-based filtering.
 */

import { generateEventId, generateSessionId, matchesPattern } from './utils';
import type {
  DashboardEvent,
  EventMeta,
  EventFilter,
  EventHandler,
  EventPersistence,
  EventStoreOptions,
  Unsubscribe,
} from '../../types/events';

export class EventStore {
  private events: DashboardEvent[] = [];
  private subscribers: Map<string, Set<EventHandler>> = new Map();
  private persistence: EventPersistence | null;
  private maxInMemory: number;
  public sessionId: string;

  constructor(options: EventStoreOptions = {}) {
    this.persistence = options.persistence ?? null;
    this.maxInMemory = options.maxInMemory ?? 10000;
    this.sessionId = generateSessionId();
  }

  /**
   * Emit an event
   *
   * @param type - Event type (hierarchical, e.g., "user.button.clicked")
   * @param payload - Event payload data
   * @param meta - Additional metadata (source, windowId, correlationId)
   * @returns The created event
   */
  emit<T = unknown>(
    type: string,
    payload: T = {} as T,
    meta: Partial<EventMeta> = {}
  ): DashboardEvent<T> {
    const event: DashboardEvent<T> = {
      id: generateEventId(),
      type,
      payload,
      timestamp: Date.now(),
      meta: {
        sessionId: this.sessionId,
        source: 'main',
        version: 1,
        ...meta,
      } as EventMeta,
    };

    // Store in memory (circular buffer)
    this.events.push(event as DashboardEvent);
    if (this.events.length > this.maxInMemory) {
      this.events.shift();
    }

    // Persist if backend configured
    if (this.persistence) {
      try {
        this.persistence.append(event as DashboardEvent);
      } catch (err) {
        console.error('[events] Persistence error:', err);
      }
    }

    // Notify subscribers
    this._notify(event as DashboardEvent);

    return event;
  }

  /**
   * Subscribe to events matching a pattern
   *
   * @param pattern - Pattern to match (e.g., "user.**", "data.*", "*")
   * @param callback - Called with each matching event
   * @returns Unsubscribe function
   */
  subscribe(pattern: string, callback: EventHandler): Unsubscribe {
    if (!this.subscribers.has(pattern)) {
      this.subscribers.set(pattern, new Set());
    }
    this.subscribers.get(pattern)!.add(callback);

    // Return unsubscribe function
    return () => {
      const callbacks = this.subscribers.get(pattern);
      if (callbacks) {
        callbacks.delete(callback);
        if (callbacks.size === 0) {
          this.subscribers.delete(pattern);
        }
      }
    };
  }

  /**
   * Query events with filters
   *
   * @param filter - Filter criteria
   * @returns Matching events
   */
  getEvents(filter: EventFilter = {}): DashboardEvent[] {
    let results = [...this.events];

    if (filter.type) {
      results = results.filter((e) => matchesPattern(e.type, filter.type!));
    }

    if (filter.since) {
      results = results.filter((e) => e.timestamp >= filter.since!);
    }

    if (filter.until) {
      results = results.filter((e) => e.timestamp <= filter.until!);
    }

    if (filter.correlationId) {
      results = results.filter((e) => e.meta.correlationId === filter.correlationId);
    }

    if (filter.limit) {
      results = results.slice(-filter.limit);
    }

    return results;
  }

  /**
   * Get event count
   */
  count(): number {
    return this.events.length;
  }

  /**
   * Clear all events from memory
   */
  clear(): void {
    this.events = [];
  }

  /**
   * Set or update the persistence backend
   */
  setPersistence(persistence: EventPersistence | null): void {
    this.persistence = persistence;
  }

  /**
   * Notify all matching subscribers of an event
   */
  private _notify(event: DashboardEvent): void {
    for (const [pattern, callbacks] of this.subscribers) {
      if (matchesPattern(event.type, pattern)) {
        for (const cb of callbacks) {
          try {
            cb(event);
          } catch (err) {
            console.error('[events] Subscriber error:', err);
          }
        }
      }
    }
  }
}
