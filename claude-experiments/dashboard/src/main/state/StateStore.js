/**
 * StateStore
 *
 * Central state management for the application.
 * - Maintains state tree in memory
 * - Handles commands to update state
 * - Emits events on state changes
 * - Persists state to disk (debounced)
 */

const { loadState, saveState, DEFAULT_STATE } = require('./persistence');

class StateStore {
  constructor(events) {
    this.events = events;
    this.state = loadState();
    this.saveTimeout = null;
    this.saveDebounceMs = 100;
  }

  /**
   * Get state at a path
   * @param {string} path - Dot-separated path (e.g., 'windows.list') or empty for full state
   * @returns {any} State at path
   */
  getState(path) {
    if (!path) {
      return this.state;
    }

    const parts = path.split('.');
    let current = this.state;

    for (const part of parts) {
      if (current === undefined || current === null) {
        return undefined;
      }
      current = current[part];
    }

    return current;
  }

  /**
   * Set state at a path
   * @param {string} path - Dot-separated path
   * @param {any} value - New value
   */
  setState(path, value) {
    if (!path) {
      this.state = value;
    } else {
      const parts = path.split('.');
      const lastPart = parts.pop();
      let current = this.state;

      for (const part of parts) {
        if (current[part] === undefined) {
          current[part] = {};
        }
        current = current[part];
      }

      current[lastPart] = value;
    }

    // Emit state change event
    this.emitChange(path);

    // Schedule debounced save
    this.scheduleSave();
  }

  /**
   * Emit a state change event
   */
  emitChange(path) {
    const eventType = path ? `state.changed.${path}` : 'state.changed';
    this.events.emit(eventType, { path, value: this.getState(path) });
  }

  /**
   * Schedule a debounced save
   */
  scheduleSave() {
    if (this.saveTimeout) {
      clearTimeout(this.saveTimeout);
    }
    this.saveTimeout = setTimeout(() => {
      this.save();
    }, this.saveDebounceMs);
  }

  /**
   * Save state to disk immediately
   */
  save() {
    if (this.saveTimeout) {
      clearTimeout(this.saveTimeout);
      this.saveTimeout = null;
    }
    try {
      saveState(this.state);
    } catch (err) {
      console.error('StateStore: Failed to save:', err);
    }
  }

  /**
   * Handle a command to update state
   * @param {string} type - Command type (e.g., 'windows.create')
   * @param {object} payload - Command payload
   * @returns {any} Result of the command
   */
  handleCommand(type, payload) {
    const [domain, action] = type.split('.');

    switch (domain) {
      case 'windows':
        return this.handleWindowsCommand(action, payload);
      case 'theme':
        return this.handleThemeCommand(action, payload);
      case 'settings':
        return this.handleSettingsCommand(action, payload);
      case 'components':
        return this.handleComponentsCommand(action, payload);
      default:
        throw new Error(`Unknown command domain: ${domain}`);
    }
  }

  // ========== Windows Commands ==========

  handleWindowsCommand(action, payload) {
    const windows = this.getState('windows');

    switch (action) {
      case 'create': {
        const id = `win_${windows.nextId}`;
        const newWindow = {
          id,
          title: payload.title || 'Window',
          component: payload.component,
          props: payload.props || {},
          x: payload.x ?? 50 + ((windows.nextId - 1) % 10) * 30,
          y: payload.y ?? 50 + ((windows.nextId - 1) % 10) * 30,
          width: payload.width || 500,
          height: payload.height || 350,
          zIndex: windows.list.length + 1,
        };

        this.setState('windows', {
          ...windows,
          list: [...windows.list, newWindow],
          focusedId: id,
          nextId: windows.nextId + 1,
        });

        return { id };
      }

      case 'close': {
        const { id } = payload;
        const newList = windows.list.filter((w) => w.id !== id);
        this.setState('windows', {
          ...windows,
          list: newList,
          focusedId: windows.focusedId === id ? null : windows.focusedId,
        });
        return { success: true };
      }

      case 'focus': {
        const { id } = payload;
        // Skip if already focused
        if (windows.focusedId === id) {
          return { success: true, noChange: true };
        }
        const maxZ = Math.max(0, ...windows.list.map((w) => w.zIndex));
        const newList = windows.list.map((w) =>
          w.id === id ? { ...w, zIndex: maxZ + 1 } : w
        );
        this.setState('windows', {
          ...windows,
          list: newList,
          focusedId: id,
        });
        return { success: true };
      }

      case 'update': {
        const { id, ...updates } = payload;
        const window = windows.list.find((w) => w.id === id);
        if (!window) {
          return { success: false, error: 'Window not found' };
        }
        // Check if anything actually changed
        const hasChanges = Object.keys(updates).some((key) => {
          if (key === 'props') {
            // Deep compare props
            return JSON.stringify(window.props) !== JSON.stringify(updates.props);
          }
          return window[key] !== updates[key];
        });
        if (!hasChanges) {
          return { success: true, noChange: true };
        }
        const newList = windows.list.map((w) =>
          w.id === id ? { ...w, ...updates } : w
        );
        this.setState('windows', {
          ...windows,
          list: newList,
        });
        return { success: true };
      }

      default:
        throw new Error(`Unknown windows action: ${action}`);
    }
  }

  // ========== Theme Commands ==========

  handleThemeCommand(action, payload) {
    const theme = this.getState('theme');

    switch (action) {
      case 'set': {
        this.setState('theme', {
          ...theme,
          current: payload.theme,
        });
        return { success: true };
      }

      case 'setVariable': {
        const { variable, value } = payload;
        this.setState('theme', {
          ...theme,
          overrides: {
            ...theme.overrides,
            [variable]: value,
          },
        });
        return { success: true };
      }

      case 'resetVariable': {
        const { variable } = payload;
        const newOverrides = { ...theme.overrides };
        delete newOverrides[variable];
        this.setState('theme', {
          ...theme,
          overrides: newOverrides,
        });
        return { success: true };
      }

      case 'resetOverrides': {
        this.setState('theme', {
          ...theme,
          overrides: {},
        });
        return { success: true };
      }

      default:
        throw new Error(`Unknown theme action: ${action}`);
    }
  }

  // ========== Settings Commands ==========

  handleSettingsCommand(action, payload) {
    const settings = this.getState('settings');

    switch (action) {
      case 'update': {
        const { key, value } = payload;
        this.setState('settings', {
          ...settings,
          [key]: value,
        });
        return { success: true };
      }

      case 'reset': {
        this.setState('settings', DEFAULT_STATE.settings);
        return { success: true };
      }

      default:
        throw new Error(`Unknown settings action: ${action}`);
    }
  }

  // ========== Components Commands ==========

  handleComponentsCommand(action, payload) {
    const components = this.getState('components');

    switch (action) {
      case 'add': {
        const instance = {
          id: payload.id || `inst_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
          type: payload.type,
          props: payload.props || {},
          createdAt: Date.now(),
        };
        this.setState('components', {
          ...components,
          instances: [...components.instances, instance],
        });
        return { id: instance.id };
      }

      case 'remove': {
        const { id } = payload;
        this.setState('components', {
          ...components,
          instances: components.instances.filter((i) => i.id !== id),
        });
        return { success: true };
      }

      case 'updateProps': {
        const { id, props } = payload;
        this.setState('components', {
          ...components,
          instances: components.instances.map((i) =>
            i.id === id ? { ...i, props: { ...i.props, ...props } } : i
          ),
        });
        return { success: true };
      }

      default:
        throw new Error(`Unknown components action: ${action}`);
    }
  }
}

module.exports = { StateStore };
