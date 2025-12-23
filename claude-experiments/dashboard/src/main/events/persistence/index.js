/**
 * Persistence Interface
 *
 * Defines the interface for event persistence backends.
 * Start with in-memory (no-op), can be swapped for file-based or SQLite.
 */

/**
 * In-memory persistence (no-op, events only stay in EventStore memory)
 * This is the default - events are not persisted to disk.
 */
const memoryPersistence = {
  append(event) {
    // No-op: events already stored in EventStore.events
  },

  getEvents(filter) {
    // Return empty - rely on EventStore.events
    return [];
  },

  close() {
    // Nothing to clean up
  },
};

/**
 * Create a JSON file persistence backend
 * Appends events as newline-delimited JSON to a file.
 *
 * @param {string} filePath - Path to the events file
 * @returns {Object} Persistence backend
 */
function createFilePersistence(filePath) {
  const fs = require('fs');
  const path = require('path');

  // Ensure directory exists
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  return {
    append(event) {
      const line = JSON.stringify(event) + '\n';
      fs.appendFileSync(filePath, line);
    },

    getEvents(filter = {}) {
      if (!fs.existsSync(filePath)) {
        return [];
      }

      const content = fs.readFileSync(filePath, 'utf-8');
      const lines = content.trim().split('\n').filter(Boolean);
      const events = lines.map((line) => JSON.parse(line));

      // Apply filters (same logic as EventStore.getEvents)
      let results = events;

      if (filter.type) {
        const { matchesPattern } = require('../utils');
        results = results.filter((e) => matchesPattern(e.type, filter.type));
      }

      if (filter.since) {
        results = results.filter((e) => e.timestamp >= filter.since);
      }

      if (filter.until) {
        results = results.filter((e) => e.timestamp <= filter.until);
      }

      if (filter.limit) {
        results = results.slice(-filter.limit);
      }

      return results;
    },

    close() {
      // Nothing to clean up for sync file writes
    },
  };
}

module.exports = {
  memoryPersistence,
  createFilePersistence,
};
