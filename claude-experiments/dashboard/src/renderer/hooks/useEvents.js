import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Subscribe to events matching a pattern
 *
 * @param {string} pattern - Event pattern to match (e.g., "user.**", "data.*")
 * @param {Object} options
 * @param {number} options.maxEvents - Max events to keep (default: 100)
 * @returns {Array} Array of matching events (newest last)
 */
export function useEventSubscription(pattern, options = {}) {
  const { maxEvents = 100 } = options;
  const [events, setEvents] = useState([]);

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
 *
 * @param {string} pattern - Event pattern to match
 * @returns {Object|null} The latest matching event, or null
 */
export function useLatestEvent(pattern) {
  const [event, setEvent] = useState(null);

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
 *
 * @returns {Function} Emit function (type, payload) => Promise<event>
 */
export function useEmit() {
  return useCallback((type, payload) => {
    return window.eventAPI.emit(type, payload);
  }, []);
}

/**
 * Query historical events
 *
 * @param {Object} filter - Filter criteria
 * @param {Array} deps - Dependency array to trigger re-query
 * @returns {Object} { events, loading, error, refetch }
 */
export function useEventQuery(filter, deps = []) {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await window.eventAPI.query(filter);
      setEvents(result);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, [JSON.stringify(filter)]);

  useEffect(() => {
    refetch();
  }, [refetch, ...deps]);

  return { events, loading, error, refetch };
}

/**
 * Subscribe and accumulate state from events
 *
 * @param {string} pattern - Event pattern to match
 * @param {Function} reducer - (state, event) => newState
 * @param {any} initialState - Initial state value
 * @returns {any} Current accumulated state
 */
export function useEventReducer(pattern, reducer, initialState) {
  const [state, setState] = useState(initialState);
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
