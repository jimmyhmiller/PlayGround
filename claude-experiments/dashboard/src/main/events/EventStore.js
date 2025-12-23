/**
 * EventStore - Core event sourcing store
 *
 * Stores events in memory with optional persistence backend.
 * Supports subscriptions with pattern-based filtering.
 */

const { generateEventId, generateSessionId, matchesPattern } = require('./utils');

class EventStore {
  /**
   * @param {Object} options
   * @param {number} options.maxInMemory - Max events to keep in memory (default: 10000)
   * @param {Object} options.persistence - Persistence backend (optional)
   */
  constructor(options = {}) {
    this.events = [];
    this.subscribers = new Map(); // pattern -> Set<callback>
    this.persistence = options.persistence || null;
    this.maxInMemory = options.maxInMemory || 10000;
    this.sessionId = generateSessionId();
  }

  /**
   * Emit an event
   *
   * @param {string} type - Event type (hierarchical, e.g., "user.button.clicked")
   * @param {Object} payload - Event payload data
   * @param {Object} meta - Additional metadata (source, windowId, correlationId)
   * @returns {Object} The created event
   */
  emit(type, payload = {}, meta = {}) {
    const event = {
      id: generateEventId(),
      type,
      payload,
      timestamp: Date.now(),
      meta: {
        sessionId: this.sessionId,
        source: 'main',
        version: 1,
        ...meta,
      },
    };

    // Store in memory (circular buffer)
    this.events.push(event);
    if (this.events.length > this.maxInMemory) {
      this.events.shift();
    }

    // Persist if backend configured
    if (this.persistence) {
      try {
        this.persistence.append(event);
      } catch (err) {
        console.error('[events] Persistence error:', err);
      }
    }

    // Notify subscribers
    this._notify(event);

    return event;
  }

  /**
   * Subscribe to events matching a pattern
   *
   * @param {string} pattern - Pattern to match (e.g., "user.**", "data.*", "*")
   * @param {Function} callback - Called with each matching event
   * @returns {Function} Unsubscribe function
   */
  subscribe(pattern, callback) {
    if (!this.subscribers.has(pattern)) {
      this.subscribers.set(pattern, new Set());
    }
    this.subscribers.get(pattern).add(callback);

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
   * @param {Object} filter
   * @param {string} filter.type - Pattern to match event types
   * @param {number} filter.since - Minimum timestamp
   * @param {number} filter.until - Maximum timestamp
   * @param {number} filter.limit - Max events to return
   * @param {string} filter.correlationId - Filter by correlation ID
   * @returns {Array} Matching events
   */
  getEvents(filter = {}) {
    let results = [...this.events];

    if (filter.type) {
      results = results.filter((e) => matchesPattern(e.type, filter.type));
    }

    if (filter.since) {
      results = results.filter((e) => e.timestamp >= filter.since);
    }

    if (filter.until) {
      results = results.filter((e) => e.timestamp <= filter.until);
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
   * @returns {number}
   */
  count() {
    return this.events.length;
  }

  /**
   * Clear all events from memory
   */
  clear() {
    this.events = [];
  }

  /**
   * Set or update the persistence backend
   * @param {Object} persistence - Persistence backend with append(event) method
   */
  setPersistence(persistence) {
    this.persistence = persistence;
  }

  /**
   * Notify all matching subscribers of an event
   * @private
   */
  _notify(event) {
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

module.exports = { EventStore };
