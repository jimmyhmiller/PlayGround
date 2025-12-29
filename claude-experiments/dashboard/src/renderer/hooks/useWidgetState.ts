import { useState, useEffect, useCallback, useRef } from 'react';
import { useDispatch } from './useBackendState';
import { useWidgetId } from './useWidgetId';

/**
 * Result type for widgetState.get command
 */
interface WidgetStateGetResult {
  success: boolean;
  state?: unknown;
  error?: string;
}

/**
 * Hook for widget state that persists across dashboard switches
 *
 * Widget state is stored per-dashboard and automatically persisted to disk
 * via the existing StateStore debounced persistence (100ms).
 *
 * @param widgetId - Unique identifier for this widget instance
 * @param initialState - Default state if no persisted state exists
 * @returns [state, updateState, loading]
 */
export function useWidgetState<T extends Record<string, unknown>>(
  widgetId: string,
  initialState: T
): [T, (updates: Partial<T>) => void, boolean] {
  const [state, setState] = useState<T>(initialState);
  const [loading, setLoading] = useState(true);
  const dispatch = useDispatch();
  const initialStateRef = useRef(initialState);
  const widgetIdRef = useRef(widgetId);

  // Keep refs updated
  initialStateRef.current = initialState;
  widgetIdRef.current = widgetId;

  // Load persisted state on mount and when widgetId changes
  useEffect(() => {
    let mounted = true;

    async function loadState(): Promise<void> {
      try {
        const result = await dispatch('widgetState.get', { widgetId }) as WidgetStateGetResult;
        if (mounted) {
          if (result.state !== undefined && result.state !== null) {
            // Merge with initial state to handle new fields
            setState(prev => ({
              ...initialStateRef.current,
              ...(result.state as Partial<T>),
            }));
          }
          setLoading(false);
        }
      } catch (err) {
        console.error(`Failed to load widget state for "${widgetId}":`, err);
        if (mounted) {
          setLoading(false);
        }
      }
    }

    loadState();

    return () => {
      mounted = false;
    };
  }, [widgetId, dispatch]);

  // Update state and persist to backend
  const updateState = useCallback(
    (updates: Partial<T>) => {
      // Compute new state
      const newState = { ...state, ...updates };

      // Update local state
      setState(newState);

      // Persist to backend
      dispatch('widgetState.set', {
        widgetId: widgetIdRef.current,
        state: newState,
      }).catch((err) => {
        console.error(`Failed to save widget state for "${widgetIdRef.current}":`, err);
      });
    },
    [dispatch, state]
  );

  return [state, updateState, loading];
}

/**
 * Hook for simple single-value widget state
 *
 * @param widgetId - Unique identifier for this widget instance
 * @param key - Key for this value within the widget's state
 * @param initialValue - Default value if not persisted
 * @returns [value, setValue, loading]
 */
export function useWidgetValue<T>(
  widgetId: string,
  key: string,
  initialValue: T
): [T, (value: T) => void, boolean] {
  const [state, updateState, loading] = useWidgetState(widgetId, { [key]: initialValue } as Record<string, unknown>);

  const value = (state[key] ?? initialValue) as T;

  const setValue = useCallback(
    (newValue: T) => {
      updateState({ [key]: newValue } as Partial<Record<string, unknown>>);
    },
    [updateState, key]
  );

  return [value, setValue, loading];
}

/**
 * Drop-in replacement for useState that automatically persists to backend.
 *
 * This is the GENERIC solution - any widget can use this instead of useState
 * and the state will automatically persist across dashboard switches.
 *
 * @param key - Unique key for this piece of state within the widget
 * @param initialValue - Default value if not persisted
 * @returns [value, setValue] - Same signature as useState
 *
 * @example
 * // Instead of:
 * const [code, setCode] = useState('');
 *
 * // Use:
 * const [code, setCode] = usePersistentState('code', '');
 */
export function usePersistentState<T>(
  key: string,
  initialValue: T
): [T, (value: T | ((prev: T) => T)) => void] {
  // Auto-get widget ID from context
  const widgetId = useWidgetId();
  const fullKey = `${widgetId}::${key}`;

  const [value, setValue] = useState<T>(initialValue);
  const [loaded, setLoaded] = useState(false);
  const dispatch = useDispatch();
  const valueRef = useRef(value);
  const loadedRef = useRef(loaded);
  valueRef.current = value;
  loadedRef.current = loaded;

  // Debug logging
  console.log(`[usePersistentState] Hook called: key="${key}", fullKey="${fullKey}"`);

  // Load persisted value on mount
  useEffect(() => {
    let mounted = true;
    console.log(`[usePersistentState] Loading: "${fullKey}"`);

    dispatch('widgetState.get', { widgetId: fullKey })
      .then((result: unknown) => {
        const r = result as WidgetStateGetResult;
        console.log(`[usePersistentState] Loaded "${fullKey}": state=`, r.state);
        if (mounted && r.state !== undefined && r.state !== null) {
          console.log(`[usePersistentState] Applying loaded state for "${key}":`, r.state);
          setValue(r.state as T);
        } else {
          console.log(`[usePersistentState] No state found for "${fullKey}", using initial:`, initialValue);
        }
        if (mounted) setLoaded(true);
      })
      .catch((err) => {
        console.error(`[usePersistentState] Load error "${fullKey}":`, err);
        if (mounted) setLoaded(true);
      });

    return () => { mounted = false; };
  }, [fullKey, dispatch]);

  // Persist on change - use ref to avoid stale closure
  const setPersistedValue = useCallback(
    (newValue: T | ((prev: T) => T)) => {
      const resolvedValue = typeof newValue === 'function'
        ? (newValue as (prev: T) => T)(valueRef.current)
        : newValue;

      setValue(resolvedValue);

      // Always persist - use ref to get current loaded state
      console.log(`[usePersistentState] Setting "${fullKey}" (loaded=${loadedRef.current}):`, resolvedValue);
      dispatch('widgetState.set', {
        widgetId: fullKey,
        state: resolvedValue,
      }).then(() => {
        console.log(`[usePersistentState] Saved "${fullKey}" successfully`);
      }).catch((err) => {
        console.error(`[usePersistentState] Failed to save "${fullKey}":`, err);
      });
    },
    [dispatch, fullKey]
  );

  return [value, setPersistedValue];
}
