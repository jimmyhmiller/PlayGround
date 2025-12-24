/**
 * Persistence Interface
 *
 * Defines the interface for event persistence backends.
 * Start with in-memory (no-op), can be swapped for file-based or SQLite.
 */

import * as fs from 'fs';
import * as path from 'path';
import type { DashboardEvent, EventFilter, EventPersistence } from '../../../types/events';
import { matchesPattern } from '../utils';

/**
 * In-memory persistence (no-op, events only stay in EventStore memory)
 * This is the default - events are not persisted to disk.
 */
export const memoryPersistence: EventPersistence = {
  append(_event: DashboardEvent): void {
    // No-op: events already stored in EventStore.events
  },

  getEvents(_filter?: EventFilter): DashboardEvent[] {
    // Return empty - rely on EventStore.events
    return [];
  },

  clear(): void {
    // Nothing to clean up
  },
};

/**
 * Create a JSON file persistence backend
 * Appends events as newline-delimited JSON to a file.
 */
export function createFilePersistence(filePath: string): EventPersistence {
  // Ensure directory exists
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  return {
    append(event: DashboardEvent): void {
      const line = JSON.stringify(event) + '\n';
      fs.appendFileSync(filePath, line);
    },

    getEvents(filter: EventFilter = {}): DashboardEvent[] {
      if (!fs.existsSync(filePath)) {
        return [];
      }

      const content = fs.readFileSync(filePath, 'utf-8');
      const lines = content.trim().split('\n').filter(Boolean);
      const events: DashboardEvent[] = lines.map((line) => JSON.parse(line) as DashboardEvent);

      // Apply filters (same logic as EventStore.getEvents)
      let results = events;

      if (filter.type) {
        results = results.filter((e) => matchesPattern(e.type, filter.type!));
      }

      if (filter.since) {
        results = results.filter((e) => e.timestamp >= filter.since!);
      }

      if (filter.until) {
        results = results.filter((e) => e.timestamp <= filter.until!);
      }

      if (filter.limit) {
        results = results.slice(-filter.limit);
      }

      return results;
    },

    clear(): void {
      // Nothing to clean up for sync file writes
    },
  };
}
