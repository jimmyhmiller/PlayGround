/**
 * useCustomWidgets Hook
 *
 * Loads custom widgets from backend state on startup and
 * listens for registration/update/unregistration events.
 */

import { useEffect, useState, useCallback } from 'react';
import { useEventSubscription } from './useEvents';
import {
  registerCustomWidget,
  unregisterCustomWidget,
  updateCustomWidget,
} from '../widgets/DynamicWidget';
import type { CustomWidgetDefinition } from '../../types/state';

/**
 * Load all custom widgets from backend state and register them
 */
async function loadCustomWidgetsFromState(): Promise<CustomWidgetDefinition[]> {
  try {
    const result = await window.stateAPI.command('customWidgets.list', {}) as {
      success?: boolean;
      widgets?: CustomWidgetDefinition[];
    };

    if (result.success && result.widgets) {
      console.log(`[useCustomWidgets] Loading ${result.widgets.length} custom widgets from state`);

      for (const widget of result.widgets) {
        const success = registerCustomWidget(widget);
        if (!success) {
          console.warn(`[useCustomWidgets] Failed to register widget: ${widget.name}`);
        }
      }

      return result.widgets;
    }
  } catch (err) {
    console.error('[useCustomWidgets] Failed to load custom widgets:', err);
  }

  return [];
}

/**
 * Hook to manage custom widgets lifecycle
 *
 * - Loads custom widgets from state on mount
 * - Subscribes to registration/update/unregistration events
 * - Provides list of loaded widgets
 */
export function useCustomWidgets(): {
  widgets: CustomWidgetDefinition[];
  loading: boolean;
  reload: () => Promise<void>;
} {
  const [widgets, setWidgets] = useState<CustomWidgetDefinition[]>([]);
  const [loading, setLoading] = useState(true);

  // Subscribe to customWidgets events
  const registeredEvents = useEventSubscription('customWidgets.registered', { maxEvents: 100 });
  const updatedEvents = useEventSubscription('customWidgets.updated', { maxEvents: 100 });
  const unregisteredEvents = useEventSubscription('customWidgets.unregistered', { maxEvents: 100 });

  // Load widgets on mount
  useEffect(() => {
    loadCustomWidgetsFromState().then((loaded) => {
      setWidgets(loaded);
      setLoading(false);
    });
  }, []);

  // Handle registered events
  useEffect(() => {
    if (registeredEvents.length === 0) return;

    const latestEvent = registeredEvents[registeredEvents.length - 1];
    const payload = latestEvent?.payload as { widget?: CustomWidgetDefinition } | undefined;
    const widget = payload?.widget;

    if (widget) {
      console.log(`[useCustomWidgets] Widget registered: ${widget.name}`);
      registerCustomWidget(widget);
      setWidgets((prev) => {
        const existing = prev.findIndex((w) => w.name === widget.name);
        if (existing !== -1) {
          const updated = [...prev];
          updated[existing] = widget;
          return updated;
        }
        return [...prev, widget];
      });
    }
  }, [registeredEvents.length]);

  // Handle updated events
  useEffect(() => {
    if (updatedEvents.length === 0) return;

    const latestEvent = updatedEvents[updatedEvents.length - 1];
    const payload = latestEvent?.payload as { widget?: CustomWidgetDefinition } | undefined;
    const widget = payload?.widget;

    if (widget) {
      console.log(`[useCustomWidgets] Widget updated: ${widget.name}`);
      updateCustomWidget(widget);
      setWidgets((prev) => {
        const existing = prev.findIndex((w) => w.name === widget.name);
        if (existing !== -1) {
          const updated = [...prev];
          updated[existing] = widget;
          return updated;
        }
        return prev;
      });
    }
  }, [updatedEvents.length]);

  // Handle unregistered events
  useEffect(() => {
    if (unregisteredEvents.length === 0) return;

    const latestEvent = unregisteredEvents[unregisteredEvents.length - 1];
    const payload = latestEvent?.payload as { name?: string } | undefined;
    const name = payload?.name;

    if (name) {
      console.log(`[useCustomWidgets] Widget unregistered: ${name}`);
      unregisterCustomWidget(name);
      setWidgets((prev) => prev.filter((w) => w.name !== name));
    }
  }, [unregisteredEvents.length]);

  // Reload function
  const reload = useCallback(async () => {
    setLoading(true);
    const loaded = await loadCustomWidgetsFromState();
    setWidgets(loaded);
    setLoading(false);
  }, []);

  return { widgets, loading, reload };
}

/**
 * Initialize custom widgets system
 * Call this once on app startup
 */
export async function initCustomWidgets(): Promise<void> {
  await loadCustomWidgetsFromState();
}

export default useCustomWidgets;
