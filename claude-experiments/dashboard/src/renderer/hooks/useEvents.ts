import { useState, useEffect, useCallback, useRef } from 'react';
import type { DashboardEvent, EventFilter } from '../../types/events';
import { useWidgetId } from './useWidgetId';
import { useDispatch } from './useBackendState';

interface UseEventSubscriptionOptions {
  maxEvents?: number;
}

interface WidgetStateGetResult {
  success: boolean;
  state?: unknown;
}

/**
 * Subscribe to events matching a pattern
 * Events are automatically persisted and restored across dashboard switches
 */
export function useEventSubscription(
  pattern: string,
  options: UseEventSubscriptionOptions = {}
): DashboardEvent[] {
  const { maxEvents = 100 } = options;
  const [events, setEvents] = useState<DashboardEvent[]>([]);
  const [loaded, setLoaded] = useState(false);
  const eventsRef = useRef(events);
  eventsRef.current = events;

  // Get widget ID for persistence
  const widgetId = useWidgetId();
  const persistKey = `${widgetId}::events::${pattern}`;
  const dispatch = useDispatch();

  // Load persisted events on mount
  useEffect(() => {
    let mounted = true;

    dispatch('widgetState.get', { widgetId: persistKey })
      .then((result: unknown) => {
        const r = result as WidgetStateGetResult;
        if (mounted && r.state && Array.isArray(r.state)) {
          console.log(`[useEventSubscription] Restored ${(r.state as DashboardEvent[]).length} events for "${pattern}"`);
          setEvents(r.state as DashboardEvent[]);
        }
        if (mounted) setLoaded(true);
      })
      .catch(() => {
        if (mounted) setLoaded(true);
      });

    return () => { mounted = false; };
  }, [persistKey, dispatch, pattern]);

  // Subscribe to new events and persist
  useEffect(() => {
    const unsubscribe = window.eventAPI.subscribe(pattern, (event) => {
      setEvents((prev) => {
        const next = [...prev, event];
        const trimmed = next.length > maxEvents ? next.slice(-maxEvents) : next;

        // Persist events (debounced by StateStore)
        if (loaded) {
          dispatch('widgetState.set', {
            widgetId: persistKey,
            state: trimmed,
          }).catch(() => {});
        }

        return trimmed;
      });
    });

    return unsubscribe;
  }, [pattern, maxEvents, persistKey, dispatch, loaded]);

  // Also persist when loaded becomes true (in case events came in before load finished)
  useEffect(() => {
    if (loaded && events.length > 0) {
      dispatch('widgetState.set', {
        widgetId: persistKey,
        state: events,
      }).catch(() => {});
    }
  }, [loaded]); // eslint-disable-line react-hooks/exhaustive-deps

  return events;
}

/**
 * Get the latest event matching a pattern
 * Automatically persisted and restored across dashboard switches
 */
export function useLatestEvent(pattern: string): DashboardEvent | null {
  const [event, setEvent] = useState<DashboardEvent | null>(null);
  const [loaded, setLoaded] = useState(false);

  // Get widget ID for persistence
  const widgetId = useWidgetId();
  const persistKey = `${widgetId}::latestEvent::${pattern}`;
  const dispatch = useDispatch();

  // Load persisted event on mount
  useEffect(() => {
    let mounted = true;

    dispatch('widgetState.get', { widgetId: persistKey })
      .then((result: unknown) => {
        const r = result as WidgetStateGetResult;
        if (mounted && r.state) {
          setEvent(r.state as DashboardEvent);
        }
        if (mounted) setLoaded(true);
      })
      .catch(() => {
        if (mounted) setLoaded(true);
      });

    return () => { mounted = false; };
  }, [persistKey, dispatch]);

  // Subscribe and persist new events
  useEffect(() => {
    const unsubscribe = window.eventAPI.subscribe(pattern, (evt) => {
      setEvent(evt);
      if (loaded) {
        dispatch('widgetState.set', {
          widgetId: persistKey,
          state: evt,
        }).catch(() => {});
      }
    });

    return unsubscribe;
  }, [pattern, persistKey, dispatch, loaded]);

  return event;
}

/**
 * Get a memoized emit function
 */
export function useEmit(): (type: string, payload?: unknown) => Promise<DashboardEvent> {
  return useCallback((type: string, payload?: unknown) => {
    return window.eventAPI.emit(type, payload);
  }, []);
}

interface UseEventQueryResult {
  events: DashboardEvent[];
  loading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

/**
 * Query historical events
 */
export function useEventQuery(
  filter: EventFilter,
  deps: unknown[] = []
): UseEventQueryResult {
  const [events, setEvents] = useState<DashboardEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const refetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await window.eventAPI.query(filter);
      setEvents(result);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, [JSON.stringify(filter)]);

  useEffect(() => {
    refetch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refetch, ...deps]);

  return { events, loading, error, refetch };
}

/**
 * Subscribe and accumulate state from events
 * Automatically persisted and restored across dashboard switches
 */
export function useEventReducer<T>(
  pattern: string,
  reducer: (state: T, event: DashboardEvent) => T,
  initialState: T
): T {
  const [state, setState] = useState<T>(initialState);
  const [loaded, setLoaded] = useState(false);
  const stateRef = useRef(state);

  // Get widget ID for persistence
  const widgetId = useWidgetId();
  const persistKey = `${widgetId}::eventReducer::${pattern}`;
  const dispatch = useDispatch();

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  // Load persisted state on mount
  useEffect(() => {
    let mounted = true;

    dispatch('widgetState.get', { widgetId: persistKey })
      .then((result: unknown) => {
        const r = result as WidgetStateGetResult;
        if (mounted && r.state !== undefined && r.state !== null) {
          setState(r.state as T);
        }
        if (mounted) setLoaded(true);
      })
      .catch(() => {
        if (mounted) setLoaded(true);
      });

    return () => { mounted = false; };
  }, [persistKey, dispatch]);

  // Subscribe and persist
  useEffect(() => {
    const unsubscribe = window.eventAPI.subscribe(pattern, (event) => {
      setState((prev) => {
        const next = reducer(prev, event);
        if (loaded) {
          dispatch('widgetState.set', {
            widgetId: persistKey,
            state: next,
          }).catch(() => {});
        }
        return next;
      });
    });

    return unsubscribe;
  }, [pattern, reducer, persistKey, dispatch, loaded]);

  return state;
}
