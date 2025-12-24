/**
 * State Persistence
 *
 * Handles loading and saving application state as JSON snapshots
 * to Electron's userData directory.
 */

import * as fs from 'fs';
import * as path from 'path';
import { app } from 'electron';
import type { AppState } from '../../types/state';

const STATE_FILENAME = 'state.json';

/**
 * Default state structure
 */
export const DEFAULT_STATE: AppState = {
  windows: {
    list: [],
    focusedId: null,
    nextId: 1,
  },
  theme: {
    current: 'dark',
    overrides: {},
  },
  settings: {
    fontSize: 'medium',
    fontScale: 1.0,
    spacing: 'normal',
  },
  components: {
    instances: [],
  },
  projects: {
    list: [],
    activeProjectId: null,
    nextProjectId: 1,
    nextDashboardId: 1,
  },
  dashboards: {
    list: [],
  },
};

/**
 * Get the path to the state file
 */
export function getStatePath(): string {
  return path.join(app.getPath('userData'), STATE_FILENAME);
}

/**
 * Load state from disk
 * Returns default state if file doesn't exist or is invalid
 */
export function loadState(): AppState {
  const statePath = getStatePath();

  try {
    if (fs.existsSync(statePath)) {
      const content = fs.readFileSync(statePath, 'utf-8');
      const loaded = JSON.parse(content) as Partial<AppState>;
      // Merge with defaults to handle missing keys from older versions
      return deepMerge(
        DEFAULT_STATE as unknown as Record<string, unknown>,
        loaded as unknown as Partial<Record<string, unknown>>
      ) as unknown as AppState;
    }
  } catch (err) {
    console.error('Failed to load state:', err);
  }

  return JSON.parse(JSON.stringify(DEFAULT_STATE)) as AppState;
}

/**
 * Save state to disk atomically
 * Uses temp file + rename to prevent corruption
 */
export function saveState(state: AppState): void {
  const statePath = getStatePath();
  const tempPath = `${statePath}.tmp`;

  try {
    const content = JSON.stringify(state, null, 2);
    fs.writeFileSync(tempPath, content, 'utf-8');
    fs.renameSync(tempPath, statePath);
  } catch (err) {
    console.error('Failed to save state:', err);
    // Clean up temp file if it exists
    try {
      if (fs.existsSync(tempPath)) {
        fs.unlinkSync(tempPath);
      }
    } catch {
      // Ignore cleanup errors
    }
    throw err;
  }
}

/**
 * Deep merge two objects
 * Target values take precedence over source for existing keys
 */
function deepMerge<T extends Record<string, unknown>>(source: T, target: Partial<T>): T {
  const result = { ...source } as T;

  for (const key of Object.keys(target) as Array<keyof T>) {
    const targetValue = target[key];
    const sourceValue = source[key];

    if (
      targetValue !== null &&
      typeof targetValue === 'object' &&
      !Array.isArray(targetValue) &&
      sourceValue !== null &&
      typeof sourceValue === 'object' &&
      !Array.isArray(sourceValue)
    ) {
      result[key] = deepMerge(
        sourceValue as Record<string, unknown>,
        targetValue as Record<string, unknown>
      ) as T[keyof T];
    } else if (targetValue !== undefined) {
      result[key] = targetValue as T[keyof T];
    }
  }

  return result;
}
