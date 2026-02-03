/**
 * DynamicWidget
 *
 * Renders custom widgets that are registered at runtime.
 * Compiles code strings into React components with the same context as EvalWidget.
 */

import React, { memo, useState, useEffect, useMemo, useCallback, useRef, type ReactElement } from 'react';
import { useBackendStateSelector, useDispatch, useBackendState } from '../hooks/useBackendState';
import { useEventSubscription, useEmit } from '../hooks/useEvents';
import { usePersistentState } from '../hooks/useWidgetState';
import { WidgetErrorBoundary } from '../components/ErrorBoundary';
import type { CustomWidgetDefinition } from '../../types/state';

// Re-export the hooks for use in dynamic widgets
export { useBackendStateSelector, useDispatch, useBackendState };
export { useEventSubscription, useEmit };
export { usePersistentState };

// Base style for widgets
const baseWidgetStyle: React.CSSProperties = {
  padding: 'var(--theme-spacing-sm, 8px)',
  background: 'var(--theme-bg-elevated, #252540)',
  borderRadius: 'var(--theme-radius-sm, 4px)',
  fontSize: '0.85em',
};

/**
 * Context provided to dynamic widgets - same as EvalWidget context
 */
const createWidgetContext = () => ({
  // React
  React,
  useState,
  useEffect,
  useMemo,
  useCallback,
  useRef,
  memo,

  // State hooks
  useBackendState,
  useBackendStateSelector,
  useDispatch,

  // Event hooks
  useEventSubscription,
  useEmit,

  // Persistent state
  usePersistentState,

  // Styling helpers
  baseWidgetStyle,

  // Convenience: dispatch commands directly
  command: async (type: string, payload?: unknown) => {
    return window.stateAPI.command(type, payload);
  },

  // Get current theme variable value
  getThemeVar: (varName: string) => {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  },
});

/**
 * Compile code string into a React component
 */
function compileWidgetCode(code: string): React.ComponentType<Record<string, unknown>> | null {
  try {
    const context = createWidgetContext();
    const contextKeys = Object.keys(context);
    const contextValues = Object.values(context);

    // The code can either:
    // 1. Be a function component directly: (props) => <div>...</div>
    // 2. Return a component: return (props) => <div>...</div>
    // 3. Return JSX directly: return <div>...</div>
    const wrappedCode = `
      "use strict";
      return (function(props) {
        const __result__ = (function() {
          ${code}
        })();

        // If result is a function, it's a component - call it with props
        if (typeof __result__ === 'function') {
          return __result__(props);
        }
        // If result is a React element, return it
        if (__result__ && typeof __result__ === 'object' && '__proto__' in __result__) {
          return __result__;
        }
        // Otherwise return null
        return null;
      });
    `;

    // eslint-disable-next-line @typescript-eslint/no-implied-eval
    const factory = new Function(...contextKeys, wrappedCode);
    return factory(...contextValues) as React.ComponentType<Record<string, unknown>>;
  } catch (err) {
    console.error('[DynamicWidget] Failed to compile code:', err);
    return null;
  }
}

// ========== Dynamic Widget Wrapper ==========

interface DynamicWidgetWrapperProps {
  /** The widget definition */
  definition: CustomWidgetDefinition;
  /** Additional props passed to the widget */
  [key: string]: unknown;
}

/**
 * DynamicWidgetWrapper - Wrapper that compiles and renders a custom widget
 */
export const DynamicWidgetWrapper = memo(function DynamicWidgetWrapper({
  definition,
  ...props
}: DynamicWidgetWrapperProps): ReactElement {
  const [Component, setComponent] = useState<React.ComponentType<Record<string, unknown>> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const codeRef = useRef<string>(definition.code);

  // Compile the widget code
  useEffect(() => {
    codeRef.current = definition.code;
    try {
      const compiled = compileWidgetCode(definition.code);
      if (compiled) {
        setComponent(() => compiled);
        setError(null);
      } else {
        setError('Failed to compile widget code');
        setComponent(null);
      }
    } catch (err) {
      setError(`Compilation error: ${(err as Error).message}`);
      setComponent(null);
    }
  }, [definition.code, definition.updatedAt]);

  if (error) {
    return (
      <div style={{
        ...baseWidgetStyle,
        color: 'var(--theme-status-error, #f44)',
        fontSize: '0.8em',
        maxWidth: 300,
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: 4 }}>
          Widget Error: {definition.name}
        </div>
        {error}
      </div>
    );
  }

  if (!Component) {
    return (
      <div style={{
        ...baseWidgetStyle,
        color: 'var(--theme-text-muted)',
        fontSize: '0.8em',
      }}>
        Loading widget: {definition.name}...
      </div>
    );
  }

  // Merge default props with passed props
  const mergedProps = { ...definition.defaultProps, ...props };

  // Wrap in error boundary (try/catch for render errors)
  try {
    return <Component {...mergedProps} />;
  } catch (err) {
    return (
      <div style={{
        ...baseWidgetStyle,
        color: 'var(--theme-status-error, #f44)',
        fontSize: '0.8em',
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: 4 }}>
          Render Error: {definition.name}
        </div>
        {(err as Error).message}
      </div>
    );
  }
});

// ========== Widget Registry ==========

import { WIDGET_TYPES } from './BuiltinWidgets';

// Cache of compiled widget factories
const compiledWidgetCache = new Map<string, {
  code: string;
  component: React.ComponentType<Record<string, unknown>>;
  version: number;
}>();

// Listeners for widget updates (used to trigger re-renders)
const widgetUpdateListeners = new Map<string, Set<() => void>>();

function subscribeToWidgetUpdates(name: string, callback: () => void): () => void {
  if (!widgetUpdateListeners.has(name)) {
    widgetUpdateListeners.set(name, new Set());
  }
  widgetUpdateListeners.get(name)!.add(callback);
  return () => {
    widgetUpdateListeners.get(name)?.delete(callback);
  };
}

function notifyWidgetUpdate(name: string): void {
  widgetUpdateListeners.get(name)?.forEach(callback => callback());
}

/**
 * Register a custom widget type in the registry
 * Returns true if successful, false otherwise
 */
export function registerCustomWidget(definition: CustomWidgetDefinition): boolean {
  try {
    // Check if we have a cached version with the same code
    const cached = compiledWidgetCache.get(definition.name);
    if (cached && cached.code === definition.code) {
      // Already registered with same code
      return true;
    }

    // Compile the code
    const compiled = compileWidgetCode(definition.code);
    if (!compiled) {
      console.error(`[DynamicWidget] Failed to compile widget: ${definition.name}`);
      return false;
    }

    // Cache the compilation
    compiledWidgetCache.set(definition.name, {
      code: definition.code,
      component: compiled,
      version: 1,
    });

    // Create a wrapper component that uses the definition
    const widgetName = definition.name;
    const WidgetComponent = memo(function CustomWidget(props: Record<string, unknown>): ReactElement {
      // Subscribe to updates for this widget to trigger re-renders
      const [, forceUpdate] = useState(0);
      useEffect(() => {
        return subscribeToWidgetUpdates(widgetName, () => forceUpdate(v => v + 1));
      }, []);

      // Get the latest definition from cache
      const cachedEntry = compiledWidgetCache.get(widgetName);
      if (!cachedEntry) {
        return (
          <div style={{
            ...baseWidgetStyle,
            color: 'var(--theme-status-error)',
          }}>
            Widget not found: {widgetName}
          </div>
        );
      }

      const Component = cachedEntry.component;
      return (
        <WidgetErrorBoundary>
          <Component {...props} />
        </WidgetErrorBoundary>
      );
    });

    // Register in the WIDGET_TYPES registry
    WIDGET_TYPES[definition.name] = {
      component: WidgetComponent as unknown as React.ComponentType<Record<string, unknown>>,
      defaultProps: definition.defaultProps,
    };

    console.log(`[DynamicWidget] Registered widget: ${definition.name}`);
    return true;
  } catch (err) {
    console.error(`[DynamicWidget] Failed to register widget ${definition.name}:`, err);
    return false;
  }
}

/**
 * Unregister a custom widget type from the registry
 */
export function unregisterCustomWidget(name: string): boolean {
  // Remove from cache
  compiledWidgetCache.delete(name);

  // Remove from registry
  if (name in WIDGET_TYPES) {
    delete WIDGET_TYPES[name];
    console.log(`[DynamicWidget] Unregistered widget: ${name}`);
    return true;
  }

  return false;
}

/**
 * Update a custom widget's code (hot-reload)
 */
export function updateCustomWidget(definition: CustomWidgetDefinition): boolean {
  // Re-compile and update cache
  const compiled = compileWidgetCode(definition.code);
  if (!compiled) {
    console.error(`[DynamicWidget] Failed to recompile widget: ${definition.name}`);
    return false;
  }

  // Get current version
  const currentEntry = compiledWidgetCache.get(definition.name);
  const currentVersion = currentEntry?.version ?? 0;

  // Update cache with incremented version
  compiledWidgetCache.set(definition.name, {
    code: definition.code,
    component: compiled,
    version: currentVersion + 1,
  });

  // Update default props in registry if present
  if (definition.name in WIDGET_TYPES) {
    WIDGET_TYPES[definition.name]!.defaultProps = definition.defaultProps;
  }

  // Notify all instances of this widget to re-render
  notifyWidgetUpdate(definition.name);

  console.log(`[DynamicWidget] Updated widget: ${definition.name} (version ${currentVersion + 1})`);
  return true;
}

/**
 * Check if a widget is a custom (dynamically registered) widget
 */
export function isCustomWidget(name: string): boolean {
  return compiledWidgetCache.has(name);
}

/**
 * Get all registered custom widget names
 */
export function getCustomWidgetNames(): string[] {
  return Array.from(compiledWidgetCache.keys());
}

// ========== Code Execution Handler ==========

/**
 * Context for code execution (non-hook subset)
 * This is used for one-shot code execution from the MCP server
 */
const createExecutionContext = (emit: (type: string, payload?: unknown) => void) => ({
  // Basic React (no hooks)
  React,

  // Emit events
  emit,

  // Dispatch commands
  command: async (type: string, payload?: unknown) => {
    return window.stateAPI.command(type, payload);
  },

  // Get current theme variable value
  getThemeVar: (varName: string) => {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  },

  // Helper to get state asynchronously
  getState: (path?: string) => {
    return window.stateAPI.get(path);
  },

  // Styling helper
  baseWidgetStyle,

  // Console access
  console,

  // JSON access
  JSON,

  // Math access
  Math,

  // Date access
  Date,

  // Promise access
  Promise,

  // Fetch access
  fetch: window.fetch.bind(window),
});

/**
 * Execute code in the execution context
 * Returns the result or an error
 */
export function executeCode(code: string, emit: (type: string, payload?: unknown) => void): { result?: unknown; error?: string } {
  try {
    const context = createExecutionContext(emit);
    const contextKeys = Object.keys(context);
    const contextValues = Object.values(context);

    // Wrap the code to return the result
    const wrappedCode = `
      "use strict";
      ${code}
    `;

    // eslint-disable-next-line @typescript-eslint/no-implied-eval
    const fn = new Function(...contextKeys, wrappedCode);
    const result = fn(...contextValues);

    return { result };
  } catch (err) {
    return { error: err instanceof Error ? err.message : String(err) };
  }
}

/**
 * CodeExecutionHandler - Invisible component that handles code.execute events
 * Should be rendered once at the top level
 */
export const CodeExecutionHandler = memo(function CodeExecutionHandler(): null {
  const emit = useEmit();

  useEffect(() => {
    const unsubscribe = window.eventAPI.subscribe('code.execute', (event: { payload: unknown }) => {
      const { executionId, code } = event.payload as { executionId: string; code: string };

      // Execute the code
      const result = executeCode(code, emit);

      // Emit the result
      emit(`code.result.${executionId}`, result);
    });

    return unsubscribe;
  }, [emit]);

  return null;
});
