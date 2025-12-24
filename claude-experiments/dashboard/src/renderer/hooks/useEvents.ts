import { useState, useEffect, useCallback, useRef } from 'react';
import type { DashboardEvent, EventFilter } from '../../types/events';

interface UseEventSubscriptionOptions {
  maxEvents?: number;
}

/**
 * Subscribe to events matching a pattern
 */
export function useEventSubscription(
  pattern: string,
  options: UseEventSubscriptionOptions = {}
): DashboardEvent[] {
  const { maxEvents = 100 } = options;
  const [events, setEvents] = useState<DashboardEvent[]>([]);

  useEffect(() => {
    const unsubscribe = window.eventAPI.subscribe(pattern, (event) => {
      setEvents((prev) => {
        const next = [...prev, event];
        return next.length > maxEvents ? next.slice(-maxEvents) : next;
      });
    });

    return unsubscribe;
  }, [pattern, maxEvents]);

  return events;
}

/**
 * Get the latest event matching a pattern
 */
export function useLatestEvent(pattern: string): DashboardEvent | null {
  const [event, setEvent] = useState<DashboardEvent | null>(null);

  useEffect(() => {
    const unsubscribe = window.eventAPI.subscribe(pattern, (evt) => {
      setEvent(evt);
    });

    return unsubscribe;
  }, [pattern]);

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
 */
export function useEventReducer<T>(
  pattern: string,
  reducer: (state: T, event: DashboardEvent) => T,
  initialState: T
): T {
  const [state, setState] = useState<T>(initialState);
  const stateRef = useRef(state);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    const unsubscribe = window.eventAPI.subscribe(pattern, (event) => {
      setState((prev) => reducer(prev, event));
    });

    return unsubscribe;
  }, [pattern, reducer]);

  return state;
}
