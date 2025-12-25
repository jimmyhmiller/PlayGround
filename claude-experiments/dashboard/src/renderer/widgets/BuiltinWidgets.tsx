/**
 * Built-in Widget Types
 *
 * Generic, configurable widget components that can be instantiated at runtime.
 * Each widget type accepts props that configure its behavior.
 */

import React, { memo, useCallback, useState, useEffect, useMemo, useRef, type ReactElement } from 'react';
import { useBackendStateSelector, useDispatch, useBackendState } from '../hooks/useBackendState';

// ========== Selector Functions ==========

/**
 * Built-in selector functions that can be referenced by name
 */
export const SELECTORS: Record<string, (state: unknown, args?: Record<string, unknown>) => unknown> = {
  // Identity - return the whole state at path
  'identity': (state) => state,

  // Count items in a list
  'count': (state) => {
    if (Array.isArray(state)) return state.length;
    if (state && typeof state === 'object' && 'list' in state) {
      return (state as { list: unknown[] }).list?.length ?? 0;
    }
    return 0;
  },

  // Get list from state
  'list': (state) => {
    if (Array.isArray(state)) return state;
    if (state && typeof state === 'object' && 'list' in state) {
      return (state as { list: unknown[] }).list ?? [];
    }
    return [];
  },

  // Get active item ID
  'activeId': (state) => {
    if (state && typeof state === 'object') {
      const s = state as Record<string, unknown>;
      return s.activeProjectId ?? s.activeDashboardId ?? s.activeId ?? s.focusedId ?? null;
    }
    return null;
  },

  // Get active project with its dashboards
  'activeProjectDashboards': (state) => {
    const s = state as { list?: Array<{ id: string; activeDashboardId?: string | null }>, activeProjectId?: string | null } | null;
    if (!s?.list || !s.activeProjectId) return { dashboards: [], activeDashboardId: null };
    const project = s.list.find(p => p.id === s.activeProjectId);
    return {
      activeProjectId: s.activeProjectId,
      activeDashboardId: project?.activeDashboardId ?? null,
    };
  },

  // Get dashboards for active project
  'dashboardsForActiveProject': (state, args) => {
    const dashboards = args?.dashboards as Array<{ id: string; projectId: string; name: string }> | undefined;
    const projects = state as { list?: Array<{ id: string; name: string; activeDashboardId?: string | null }>, activeProjectId?: string | null } | null;

    if (!projects?.list || !projects.activeProjectId || !dashboards) {
      return { dashboards: [], activeDashboardId: null, projectName: null };
    }

    const project = projects.list.find(p => p.id === projects.activeProjectId);
    const projectDashboards = dashboards.filter(d => d.projectId === projects.activeProjectId);

    return {
      dashboards: projectDashboards,
      activeDashboardId: project?.activeDashboardId ?? null,
      projectName: project?.name ?? null,
    };
  },
};

// ========== Widget Styles ==========

const baseWidgetStyle: React.CSSProperties = {
  padding: 'var(--theme-spacing-sm, 8px)',
  background: 'var(--theme-bg-elevated, #252540)',
  borderRadius: 'var(--theme-radius-sm, 4px)',
  fontSize: '0.85em',
};

// ========== StateValue Widget ==========

export interface StateValueProps {
  /** State path to subscribe to */
  path: string;
  /** Selector name from SELECTORS, or 'identity' */
  selector?: string;
  /** Label to show before the value */
  label?: string;
  /** Suffix to show after the value (e.g., "projects") */
  suffix?: string;
  /** Format: 'text' | 'number' | 'badge' */
  format?: 'text' | 'number' | 'badge';
}

/**
 * StateValue - Displays a single value from state
 */
export const StateValue = memo(function StateValue({
  path,
  selector = 'identity',
  label,
  suffix,
  format = 'text',
}: StateValueProps): ReactElement | null {
  const selectorFn = SELECTORS[selector] ?? SELECTORS.identity!;

  const [value, loading] = useBackendStateSelector(
    path,
    (state) => selectorFn(state)
  );

  if (loading) return null;

  const displayValue = value?.toString() ?? '';

  return (
    <div className="widget-state-value" style={{
      ...baseWidgetStyle,
      display: 'flex',
      alignItems: 'center',
      gap: 'var(--theme-spacing-xs, 4px)',
      color: 'var(--theme-text-secondary, #aaa)',
    }}>
      {label && <span>{label}</span>}
      <span style={{
        fontWeight: format === 'badge' ? 'bold' : 'normal',
        color: format === 'badge' ? 'var(--theme-accent-primary, #6366f1)' : 'inherit',
        fontSize: format === 'badge' ? '1.1em' : 'inherit',
      }}>
        {displayValue}
      </span>
      {suffix && <span>{suffix}</span>}
    </div>
  );
});

// ========== StateList Widget ==========

export interface StateListProps {
  /** State path to subscribe to */
  path: string;
  /** Selector name to get the list */
  selector?: string;
  /** Property name for item label */
  labelKey?: string;
  /** Property name for item ID */
  idKey?: string;
  /** Command to dispatch on item click */
  onClickCommand?: string;
  /** Header text */
  header?: string;
  /** Selector to get active item ID (relative to same path) */
  activeSelector?: string;
}

/**
 * StateList - Displays a list of items from state with optional click handling
 */
export const StateList = memo(function StateList({
  path,
  selector = 'list',
  labelKey = 'name',
  idKey = 'id',
  onClickCommand,
  header,
  activeSelector,
}: StateListProps): ReactElement | null {
  const selectorFn = SELECTORS[selector] ?? SELECTORS.list!;
  const activeSelectorFn = activeSelector ? SELECTORS[activeSelector] : null;

  const [items, loading] = useBackendStateSelector(
    path,
    (state) => selectorFn(state) as Array<Record<string, unknown>>
  );

  const [activeId] = useBackendStateSelector(
    path,
    (state) => activeSelectorFn ? activeSelectorFn(state) : null
  );

  const dispatch = useDispatch();

  const handleClick = useCallback(async (id: string) => {
    if (onClickCommand) {
      await dispatch(onClickCommand, { id });
    }
  }, [dispatch, onClickCommand]);

  if (loading || !items) return null;

  return (
    <div className="widget-state-list" style={{
      ...baseWidgetStyle,
      display: 'flex',
      flexDirection: 'column',
      gap: 'var(--theme-spacing-xs, 4px)',
      minWidth: 150,
    }}>
      {header && (
        <div style={{
          fontSize: '0.7em',
          color: 'var(--theme-text-muted, #888)',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          marginBottom: 'var(--theme-spacing-xs, 4px)',
          paddingBottom: 'var(--theme-spacing-xs, 4px)',
          borderBottom: '1px solid var(--theme-border-primary, #333)',
        }}>
          {header}
        </div>
      )}
      {items.map((item) => {
        const id = String(item[idKey] ?? '');
        const label = String(item[labelKey] ?? '');
        const isActive = id === activeId;

        return (
          <button
            key={id}
            onClick={() => handleClick(id)}
            disabled={!onClickCommand}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 'var(--theme-spacing-xs, 4px)',
              padding: 'var(--theme-spacing-xs, 4px) var(--theme-spacing-sm, 8px)',
              background: isActive ? 'var(--theme-accent-primary, #6366f1)' : 'transparent',
              color: isActive ? 'var(--theme-bg-primary, #0f0f1a)' : 'var(--theme-text-primary, #e0e0e0)',
              border: 'none',
              borderRadius: 'var(--theme-radius-sm, 4px)',
              cursor: onClickCommand ? 'pointer' : 'default',
              fontSize: '0.85em',
              textAlign: 'left',
              fontFamily: 'inherit',
              opacity: onClickCommand ? 1 : 0.8,
            }}
          >
            <span style={{ opacity: 0.6 }}>●</span>
            {label}
          </button>
        );
      })}
      {items.length === 0 && (
        <span style={{ color: 'var(--theme-text-muted, #888)', fontStyle: 'italic' }}>
          Empty
        </span>
      )}
    </div>
  );
});

// ========== DashboardList Widget (Special Case) ==========

export interface DashboardListWidgetProps {
  /** Optional header override */
  header?: string;
}

/**
 * DashboardList - Special widget that shows dashboards for the active project
 * This requires combining data from both projects and dashboards state
 */
export const DashboardList = memo(function DashboardList({
  header,
}: DashboardListWidgetProps): ReactElement | null {
  // Get active project info
  const [projectInfo, projectsLoading] = useBackendStateSelector(
    'projects',
    (state) => {
      const s = state as { list?: Array<{ id: string; name: string; activeDashboardId?: string | null }>, activeProjectId?: string | null } | null;
      if (!s?.list || !s.activeProjectId) return null;
      const project = s.list.find(p => p.id === s.activeProjectId);
      return {
        projectId: s.activeProjectId,
        projectName: project?.name ?? null,
        activeDashboardId: project?.activeDashboardId ?? null,
      };
    }
  );

  // Get dashboards
  const [allDashboards, dashboardsLoading] = useBackendStateSelector(
    'dashboards',
    (state) => {
      const s = state as { list?: Array<{ id: string; projectId: string; name: string }> } | null;
      return s?.list ?? [];
    }
  );

  const dispatch = useDispatch();

  const handleClick = useCallback(async (id: string) => {
    await dispatch('dashboards.switch', { id });
  }, [dispatch]);

  if (projectsLoading || dashboardsLoading || !projectInfo || !allDashboards) return null;

  const dashboards = allDashboards.filter(d => d.projectId === projectInfo.projectId);

  return (
    <div className="widget-dashboard-list" style={{
      ...baseWidgetStyle,
      display: 'flex',
      flexDirection: 'column',
      gap: 'var(--theme-spacing-xs, 4px)',
      minWidth: 150,
    }}>
      <div style={{
        fontSize: '0.7em',
        color: 'var(--theme-text-muted, #888)',
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        marginBottom: 'var(--theme-spacing-xs, 4px)',
        paddingBottom: 'var(--theme-spacing-xs, 4px)',
        borderBottom: '1px solid var(--theme-border-primary, #333)',
      }}>
        {header ?? projectInfo.projectName ?? 'Dashboards'}
      </div>
      {dashboards.map((dashboard) => {
        const isActive = dashboard.id === projectInfo.activeDashboardId;

        return (
          <button
            key={dashboard.id}
            onClick={() => handleClick(dashboard.id)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 'var(--theme-spacing-xs, 4px)',
              padding: 'var(--theme-spacing-xs, 4px) var(--theme-spacing-sm, 8px)',
              background: isActive ? 'var(--theme-accent-primary, #6366f1)' : 'transparent',
              color: isActive ? 'var(--theme-bg-primary, #0f0f1a)' : 'var(--theme-text-primary, #e0e0e0)',
              border: 'none',
              borderRadius: 'var(--theme-radius-sm, 4px)',
              cursor: 'pointer',
              fontSize: '0.85em',
              textAlign: 'left',
              fontFamily: 'inherit',
            }}
          >
            <span style={{ opacity: 0.6 }}>●</span>
            {dashboard.name}
          </button>
        );
      })}
      {dashboards.length === 0 && (
        <span style={{ color: 'var(--theme-text-muted, #888)', fontStyle: 'italic' }}>
          No dashboards
        </span>
      )}
    </div>
  );
});

// ========== Eval Widget ==========

export interface EvalWidgetProps {
  /** React component code as a string */
  code?: string;
  /** Path to a file containing React component code */
  file?: string;
  /** Additional props to pass to the evaluated component */
  componentProps?: Record<string, unknown>;
}

/**
 * Context provided to evaluated components
 */
const createEvalContext = () => ({
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

  // Selectors
  SELECTORS,

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
 * EvalWidget - Renders arbitrary React code passed as a string
 *
 * The code should export a default component or return a React element.
 *
 * Available in scope:
 * - React, useState, useEffect, useMemo, useCallback, useRef, memo
 * - useBackendState, useBackendStateSelector, useDispatch
 * - SELECTORS (built-in selector functions)
 * - baseWidgetStyle (common widget styling)
 * - command(type, payload) - dispatch state commands
 * - getThemeVar(name) - get computed CSS variable value
 * - props - any componentProps passed in
 */
export const EvalWidget = memo(function EvalWidget({
  code,
  file,
  componentProps = {},
}: EvalWidgetProps): ReactElement | null {
  const [fileCode, setFileCode] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [Component, setComponent] = useState<React.ComponentType<Record<string, unknown>> | null>(null);

  // Load code from file if specified
  useEffect(() => {
    if (file) {
      window.fileAPI?.load(file)
        .then((result) => {
          setFileCode(result.content);
          setError(null);
        })
        .catch((err: Error) => {
          setError(`Failed to load file: ${err.message}`);
          setFileCode(null);
        });
    }
  }, [file]);

  // Compile the code into a component
  useEffect(() => {
    const sourceCode = code ?? fileCode;
    if (!sourceCode) {
      setComponent(null);
      return;
    }

    try {
      const context = createEvalContext();

      // Create a function that has all context vars in scope
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
            ${sourceCode}
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
      const EvaluatedComponent = factory(...contextValues);

      setComponent(() => EvaluatedComponent);
      setError(null);
    } catch (err) {
      setError(`Eval error: ${(err as Error).message}`);
      setComponent(null);
    }
  }, [code, fileCode]);

  if (error) {
    return (
      <div style={{
        ...baseWidgetStyle,
        color: 'var(--theme-status-error, #f44)',
        fontSize: '0.8em',
        maxWidth: 300,
      }}>
        {error}
      </div>
    );
  }

  if (!Component) {
    return null;
  }

  // Wrap in error boundary
  try {
    return <Component {...componentProps} />;
  } catch (err) {
    return (
      <div style={{
        ...baseWidgetStyle,
        color: 'var(--theme-status-error, #f44)',
        fontSize: '0.8em',
      }}>
        Render error: {(err as Error).message}
      </div>
    );
  }
});

// ========== Widget Type Registry ==========

export interface WidgetTypeConfig {
  component: React.ComponentType<Record<string, unknown>>;
  defaultProps: Record<string, unknown>;
}

/**
 * Registry of built-in widget types
 */
export const WIDGET_TYPES: Record<string, WidgetTypeConfig> = {
  'state-value': {
    component: StateValue as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { path: 'projects', selector: 'count', suffix: 'projects' },
  },
  'state-list': {
    component: StateList as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { path: 'projects', selector: 'list', labelKey: 'name', idKey: 'id' },
  },
  'dashboard-list': {
    component: DashboardList as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: {},
  },
  'eval': {
    component: EvalWidget as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: {},
  },
};
