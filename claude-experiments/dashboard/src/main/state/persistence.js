/**
 * State Persistence
 *
 * Handles loading and saving application state as JSON snapshots
 * to Electron's userData directory.
 */

const fs = require('fs');
const path = require('path');
const { app } = require('electron');

const STATE_FILENAME = 'state.json';

/**
 * Default state structure
 */
const DEFAULT_STATE = {
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
};

/**
 * Get the path to the state file
 */
function getStatePath() {
  return path.join(app.getPath('userData'), STATE_FILENAME);
}

/**
 * Load state from disk
 * Returns default state if file doesn't exist or is invalid
 */
function loadState() {
  const statePath = getStatePath();

  try {
    if (fs.existsSync(statePath)) {
      const content = fs.readFileSync(statePath, 'utf-8');
      const loaded = JSON.parse(content);
      // Merge with defaults to handle missing keys from older versions
      return deepMerge(DEFAULT_STATE, loaded);
    }
  } catch (err) {
    console.error('Failed to load state:', err);
  }

  return JSON.parse(JSON.stringify(DEFAULT_STATE));
}

/**
 * Save state to disk atomically
 * Uses temp file + rename to prevent corruption
 */
function saveState(state) {
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
    } catch (cleanupErr) {
      // Ignore cleanup errors
    }
    throw err;
  }
}

/**
 * Deep merge two objects
 * Target values take precedence over source for existing keys
 */
function deepMerge(source, target) {
  const result = { ...source };

  for (const key of Object.keys(target)) {
    if (
      target[key] !== null &&
      typeof target[key] === 'object' &&
      !Array.isArray(target[key]) &&
      source[key] !== null &&
      typeof source[key] === 'object' &&
      !Array.isArray(source[key])
    ) {
      result[key] = deepMerge(source[key], target[key]);
    } else {
      result[key] = target[key];
    }
  }

  return result;
}

module.exports = {
  DEFAULT_STATE,
  loadState,
  saveState,
  getStatePath,
};
