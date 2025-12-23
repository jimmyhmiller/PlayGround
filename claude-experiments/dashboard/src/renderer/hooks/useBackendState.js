import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Shallow equality comparison for objects
 */
function shallowEqual(a, b) {
  if (a === b) return true;
  if (!a || !b) return false;
  if (typeof a !== 'object' || typeof b !== 'object') return a === b;
  if (Array.isArray(a) !== Array.isArray(b)) return false;

  const keysA = Object.keys(a);
  const keysB = Object.keys(b);
  if (keysA.length !== keysB.length) return false;
  return keysA.every(key => a[key] === b[key]);
}

/**
 * Selector-based state subscription (like Redux useSelector)
 * Only re-renders when the selected value changes.
 *
 * @param {string} path - State path to subscribe to
 * @param {Function} selector - Function to extract data from state
 * @param {Function} equalityFn - Equality function (default: shallowEqual)
 * @returns {any} Selected state value
 */
export function useBackendStateSelector(path, selector, equalityFn = shallowEqual) {
  const [selectedState, setSelectedState] = useState(null);
  const [loading, setLoading] = useState(true);

  // Keep selector in ref so we always use the latest
  const selectorRef = useRef(selector);
  selectorRef.current = selector;

  const equalityRef = useRef(equalityFn);
  equalityRef.current = equalityFn;

  // Load initial state
  useEffect(() => {
    let mounted = true;

    window.stateAPI.get(path).then(state => {
      if (mounted) {
        setSelectedState(selectorRef.current(state));
        setLoading(false);
      }
    }).catch(err => {
      console.error(`Failed to load state at path "${path}":`, err);
      if (mounted) {
        setLoading(false);
      }
    });

    return () => { mounted = false; };
  }, [path]);

  // Subscribe to changes, only update if selected value changed
  useEffect(() => {
    const unsubscribe = window.stateAPI.subscribe(path, async (event) => {
      // Get full state - either from event or refetch
      let fullState;
      if (event.payload?.path === path) {
        fullState = event.payload.value;
      } else {
        // Child path changed, refetch
        fullState = await window.stateAPI.get(path);
      }

      if (fullState !== undefined) {
        const newSelected = selectorRef.current(fullState);
        setSelectedState(prev => {
          if (equalityRef.current(prev, newSelected)) {
            return prev; // No change, no re-render
          }
          return newSelected;
        });
      }
    });

    return unsubscribe;
  }, [path]);

  return [selectedState, loading];
}

/**
 * Hook for backend-driven state management
 *
 * Loads initial state from backend, subscribes to changes,
 * and provides a dispatch function to send commands.
 *
 * @param {string} path - State path (e.g., 'settings', 'windows', 'theme')
 * @returns {[any, Function, boolean]} [state, dispatch, loading]
 */
export function useBackendState(path) {
  const [state, setState] = useState(null);
  const [loading, setLoading] = useState(true);
  const stateRef = useRef(state);

  // Keep ref in sync
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  // Load initial state
  useEffect(() => {
    let mounted = true;

    async function loadState() {
      try {
        const initialState = await window.stateAPI.get(path);
        if (mounted) {
          setState(initialState);
          setLoading(false);
        }
      } catch (err) {
        console.error(`Failed to load state at path "${path}":`, err);
        if (mounted) {
          setLoading(false);
        }
      }
    }

    loadState();

    return () => {
      mounted = false;
    };
  }, [path]);

  // Subscribe to state changes
  useEffect(() => {
    const unsubscribe = window.stateAPI.subscribe(path, (event) => {
      // Event payload contains { path, value }
      // We need to update our state based on the event
      if (event.payload && event.payload.path) {
        const eventPath = event.payload.path;
        const eventValue = event.payload.value;

        // If the event path matches our path exactly, update the whole state
        if (eventPath === path) {
          setState(eventValue);
        } else if (eventPath.startsWith(path + '.')) {
          // If the event path is a child of our path, we need to refetch
          // This is simpler than trying to do partial updates
          window.stateAPI.get(path).then((newState) => {
            setState(newState);
          });
        }
      }
    });

    return unsubscribe;
  }, [path]);

  // Dispatch function to send commands
  const dispatch = useCallback(async (type, payload) => {
    try {
      const result = await window.stateAPI.command(type, payload);
      return result;
    } catch (err) {
      console.error(`State command "${type}" failed:`, err);
      throw err;
    }
  }, []);

  return [state, dispatch, loading];
}

/**
 * Get dispatch function for sending commands (without subscribing to state)
 */
export function useDispatch() {
  return useCallback(async (type, payload) => {
    try {
      const result = await window.stateAPI.command(type, payload);
      return result;
    } catch (err) {
      console.error(`State command "${type}" failed:`, err);
      throw err;
    }
  }, []);
}

/**
 * Hook for window state (full subscription - use sparingly)
 *
 * @returns {Object} { windows, focusedId, createWindow, closeWindow, focusWindow, updateWindow, loading }
 */
export function useWindowState() {
  const [state, dispatch, loading] = useBackendState('windows');

  const createWindow = useCallback(
    (options) => dispatch('windows.create', options),
    [dispatch]
  );

  const closeWindow = useCallback(
    (id) => dispatch('windows.close', { id }),
    [dispatch]
  );

  const focusWindow = useCallback(
    (id) => dispatch('windows.focus', { id }),
    [dispatch]
  );

  const updateWindow = useCallback(
    (id, updates) => dispatch('windows.update', { id, ...updates }),
    [dispatch]
  );

  return {
    windows: state?.list ?? [],
    focusedId: state?.focusedId ?? null,
    createWindow,
    closeWindow,
    focusWindow,
    updateWindow,
    loading,
  };
}

/**
 * Hook for window list only (doesn't re-render on focus changes)
 */
export function useWindowList() {
  const [list, loading] = useBackendStateSelector(
    'windows',
    state => state?.list ?? []
  );
  return [list ?? [], loading];
}

/**
 * Hook for focused window ID only
 */
export function useWindowFocus() {
  const [focusedId, loading] = useBackendStateSelector(
    'windows',
    state => state?.focusedId ?? null
  );
  return [focusedId, loading];
}

/**
 * Hook for a single window by ID - only re-renders when THAT window changes
 */
export function useWindow(windowId) {
  const [window, loading] = useBackendStateSelector(
    'windows',
    state => state?.list?.find(w => w.id === windowId) ?? null
  );
  return [window, loading];
}

/**
 * Hook for window commands without subscribing to state
 */
export function useWindowCommands() {
  const dispatch = useDispatch();

  const createWindow = useCallback(
    (options) => dispatch('windows.create', options),
    [dispatch]
  );

  const closeWindow = useCallback(
    (id) => dispatch('windows.close', { id }),
    [dispatch]
  );

  const focusWindow = useCallback(
    (id) => dispatch('windows.focus', { id }),
    [dispatch]
  );

  const updateWindow = useCallback(
    (id, updates) => dispatch('windows.update', { id, ...updates }),
    [dispatch]
  );

  return { createWindow, closeWindow, focusWindow, updateWindow };
}

/**
 * Hook for theme state
 *
 * @returns {Object} { currentTheme, overrides, setTheme, setVariable, resetVariable, resetOverrides, loading }
 */
export function useThemeState() {
  const [state, dispatch, loading] = useBackendState('theme');

  const setTheme = useCallback(
    (theme) => dispatch('theme.set', { theme }),
    [dispatch]
  );

  const setVariable = useCallback(
    (variable, value) => dispatch('theme.setVariable', { variable, value }),
    [dispatch]
  );

  const resetVariable = useCallback(
    (variable) => dispatch('theme.resetVariable', { variable }),
    [dispatch]
  );

  const resetOverrides = useCallback(
    () => dispatch('theme.resetOverrides', {}),
    [dispatch]
  );

  return {
    currentTheme: state?.current ?? 'dark',
    overrides: state?.overrides ?? {},
    setTheme,
    setVariable,
    resetVariable,
    resetOverrides,
    loading,
  };
}

/**
 * Hook for settings state
 *
 * @returns {Object} { settings, updateSetting, resetSettings, loading }
 */
export function useSettingsState() {
  const [state, dispatch, loading] = useBackendState('settings');

  const updateSetting = useCallback(
    (key, value) => dispatch('settings.update', { key, value }),
    [dispatch]
  );

  const resetSettings = useCallback(
    () => dispatch('settings.reset', {}),
    [dispatch]
  );

  return {
    settings: state ?? { fontSize: 'medium', fontScale: 1.0, spacing: 'normal' },
    updateSetting,
    resetSettings,
    loading,
  };
}

/**
 * Hook for component registry state
 *
 * @returns {Object} { instances, addInstance, removeInstance, updateInstanceProps, loading }
 */
export function useComponentsState() {
  const [state, dispatch, loading] = useBackendState('components');

  const addInstance = useCallback(
    (type, props) => dispatch('components.add', { type, props }),
    [dispatch]
  );

  const removeInstance = useCallback(
    (id) => dispatch('components.remove', { id }),
    [dispatch]
  );

  const updateInstanceProps = useCallback(
    (id, props) => dispatch('components.updateProps', { id, props }),
    [dispatch]
  );

  return {
    instances: state?.instances ?? [],
    addInstance,
    removeInstance,
    updateInstanceProps,
    loading,
  };
}
