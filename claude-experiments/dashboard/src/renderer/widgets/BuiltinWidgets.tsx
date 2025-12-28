/**
 * Built-in Widget Types
 *
 * Generic, configurable widget components that can be instantiated at runtime.
 * Each widget type accepts props that configure its behavior.
 */

import React, { memo, useCallback, useState, useEffect, useMemo, useRef, type ReactElement } from 'react';
import { useBackendStateSelector, useDispatch, useBackendState } from '../hooks/useBackendState';
import { useEventSubscription, useEmit, useEventReducer } from '../hooks/useEvents';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
// DashboardEvent type used indirectly via event hooks

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

// ========== Event-Driven Widgets ==========

/**
 * ChartWidget - Renders a chart from event data
 *
 * Subscribes to events matching `subscribePattern`, extracts data using `dataKey`,
 * and renders as a bar or line chart.
 */
export interface ChartWidgetProps {
  /** Event pattern to subscribe to */
  subscribePattern: string;
  /** Key to extract numeric value from event payload */
  dataKey?: string;
  /** Key to extract label from event payload */
  labelKey?: string;
  /** Chart type */
  chartType?: 'bar' | 'line';
  /** Max data points to show */
  maxPoints?: number;
  /** Chart height */
  height?: number;
  /** Title */
  title?: string;
}

export const ChartWidget = memo(function ChartWidget({
  subscribePattern,
  dataKey = 'value',
  labelKey = 'name',
  chartType = 'bar',
  maxPoints = 20,
  height = 150,
  title,
}: ChartWidgetProps): ReactElement {
  const events = useEventSubscription(subscribePattern, { maxEvents: maxPoints });

  const chartData = useMemo(() => {
    return events.map((event, index) => {
      const payload = event.payload as Record<string, unknown>;
      return {
        name: payload[labelKey] ?? `#${index + 1}`,
        value: typeof payload[dataKey] === 'number' ? payload[dataKey] : 0,
      };
    });
  }, [events, dataKey, labelKey]);

  if (chartData.length === 0) {
    return (
      <div style={{ ...baseWidgetStyle, color: 'var(--theme-text-muted)', textAlign: 'center' }}>
        Waiting for {subscribePattern} events...
      </div>
    );
  }

  return (
    <div style={{ ...baseWidgetStyle, padding: 0 }}>
      {title && (
        <div style={{ padding: '8px 12px', fontSize: '0.8em', color: 'var(--theme-text-muted)', borderBottom: '1px solid var(--theme-border-primary)' }}>
          {title}
        </div>
      )}
      <ResponsiveContainer width="100%" height={height}>
        {chartType === 'bar' ? (
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--theme-text-muted)' }} />
            <YAxis tick={{ fontSize: 10, fill: 'var(--theme-text-muted)' }} />
            <Tooltip contentStyle={{ background: 'var(--theme-bg-elevated)', border: '1px solid var(--theme-border-primary)' }} />
            <Bar dataKey="value" fill="var(--theme-accent-primary)" />
          </BarChart>
        ) : (
          <LineChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--theme-text-muted)' }} />
            <YAxis tick={{ fontSize: 10, fill: 'var(--theme-text-muted)' }} />
            <Tooltip contentStyle={{ background: 'var(--theme-bg-elevated)', border: '1px solid var(--theme-border-primary)' }} />
            <Line type="monotone" dataKey="value" stroke="var(--theme-accent-primary)" dot={false} />
          </LineChart>
        )}
      </ResponsiveContainer>
    </div>
  );
});

/**
 * TableWidget - Renders a table from event data
 *
 * Subscribes to events and displays payload fields as columns.
 */
export interface TableWidgetProps {
  /** Event pattern to subscribe to */
  subscribePattern: string;
  /** Columns to display (keys from payload) */
  columns: string[];
  /** Column headers (optional, defaults to column keys) */
  headers?: string[];
  /** Max rows to show */
  maxRows?: number;
  /** Title */
  title?: string;
}

export const TableWidget = memo(function TableWidget({
  subscribePattern,
  columns,
  headers,
  maxRows = 10,
  title,
}: TableWidgetProps): ReactElement {
  const events = useEventSubscription(subscribePattern, { maxEvents: maxRows });

  const displayHeaders = headers ?? columns;

  return (
    <div style={{ ...baseWidgetStyle, padding: 0 }}>
      {title && (
        <div style={{ padding: '8px 12px', fontSize: '0.8em', color: 'var(--theme-text-muted)', borderBottom: '1px solid var(--theme-border-primary)' }}>
          {title}
        </div>
      )}
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85em' }}>
        <thead>
          <tr>
            {displayHeaders.map((h, i) => (
              <th key={i} style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid var(--theme-border-primary)', color: 'var(--theme-text-muted)', fontWeight: 500 }}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {events.length === 0 ? (
            <tr>
              <td colSpan={columns.length} style={{ padding: '12px', textAlign: 'center', color: 'var(--theme-text-muted)' }}>
                Waiting for events...
              </td>
            </tr>
          ) : (
            events.map((event, i) => {
              const payload = event.payload as Record<string, unknown>;
              return (
                <tr key={event.id ?? i}>
                  {columns.map((col, j) => (
                    <td key={j} style={{ padding: '6px 8px', borderBottom: '1px solid var(--theme-border-primary)' }}>
                      {formatCellValue(payload[col])}
                    </td>
                  ))}
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
});

function formatCellValue(value: unknown): string {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'number') return value.toFixed(3);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return String(value);
}

/**
 * StatsWidget - Computes statistics from numeric events and emits results
 *
 * Subscribes to events, computes mean/min/max/count, and emits stats events.
 */
export interface StatsWidgetProps {
  /** Event pattern to subscribe to */
  subscribePattern: string;
  /** Key to extract numeric value from payload */
  dataKey?: string;
  /** Event type to emit with computed stats */
  emitAs?: string;
  /** Window size for stats computation */
  windowSize?: number;
  /** Show stats inline */
  showStats?: boolean;
}

interface StatsState {
  values: number[];
  count: number;
  sum: number;
  min: number;
  max: number;
}

export const StatsWidget = memo(function StatsWidget({
  subscribePattern,
  dataKey = 'value',
  emitAs,
  windowSize = 100,
  showStats = true,
}: StatsWidgetProps): ReactElement | null {
  const emit = useEmit();

  const stats = useEventReducer<StatsState>(
    subscribePattern,
    (state, event) => {
      const payload = event.payload as Record<string, unknown>;
      const value = typeof payload[dataKey] === 'number' ? payload[dataKey] as number : NaN;

      if (isNaN(value)) return state;

      const values = [...state.values, value].slice(-windowSize);
      const count = state.count + 1;
      const sum = state.sum + value;
      const min = Math.min(state.min, value);
      const max = Math.max(state.max, value);

      return { values, count, sum, min, max };
    },
    { values: [], count: 0, sum: 0, min: Infinity, max: -Infinity }
  );

  const mean = stats.count > 0 ? stats.sum / stats.count : 0;
  const windowMean = stats.values.length > 0
    ? stats.values.reduce((a, b) => a + b, 0) / stats.values.length
    : 0;

  // Emit stats when they change
  useEffect(() => {
    if (emitAs && stats.count > 0) {
      emit(emitAs, {
        count: stats.count,
        mean,
        windowMean,
        min: stats.min === Infinity ? 0 : stats.min,
        max: stats.max === -Infinity ? 0 : stats.max,
        windowSize: stats.values.length,
      });
    }
  }, [emitAs, emit, stats.count, mean, windowMean, stats.min, stats.max, stats.values.length]);

  if (!showStats) return null;

  return (
    <div style={{ ...baseWidgetStyle, display: 'flex', gap: '16px', fontSize: '0.85em' }}>
      <div>
        <span style={{ color: 'var(--theme-text-muted)' }}>n:</span> {stats.count}
      </div>
      <div>
        <span style={{ color: 'var(--theme-text-muted)' }}>mean:</span> {mean.toFixed(3)}
      </div>
      <div>
        <span style={{ color: 'var(--theme-text-muted)' }}>min:</span> {stats.min === Infinity ? '-' : stats.min.toFixed(3)}
      </div>
      <div>
        <span style={{ color: 'var(--theme-text-muted)' }}>max:</span> {stats.max === -Infinity ? '-' : stats.max.toFixed(3)}
      </div>
    </div>
  );
});

/**
 * TransformWidget - Subscribes to events, transforms payload, emits new events
 *
 * A pure data transformer - no UI.
 */
export interface TransformWidgetProps {
  /** Event pattern to subscribe to */
  subscribePattern: string;
  /** Event type to emit */
  emitAs: string;
  /** Transform function as a string (receives payload, returns new payload) */
  transform: string;
}

export const TransformWidget = memo(function TransformWidget({
  subscribePattern,
  emitAs,
  transform,
}: TransformWidgetProps): ReactElement | null {
  const emit = useEmit();

  useEffect(() => {
    // Create transform function
    let transformFn: (payload: unknown) => unknown;
    try {
      // eslint-disable-next-line @typescript-eslint/no-implied-eval
      transformFn = new Function('payload', `return (${transform})(payload)`) as (payload: unknown) => unknown;
    } catch (err) {
      console.error('[TransformWidget] Invalid transform:', err);
      return;
    }

    const unsubscribe = window.eventAPI.subscribe(subscribePattern, (event) => {
      try {
        const result = transformFn(event.payload);
        if (result !== undefined) {
          emit(emitAs, result);
        }
      } catch (err) {
        console.error('[TransformWidget] Transform error:', err);
      }
    });

    return unsubscribe;
  }, [subscribePattern, emitAs, transform, emit]);

  return null; // No UI - pure transformer
});

// ========== Layout Container Widget ==========

/**
 * LayoutContainer - Composes multiple widgets with layout
 *
 * Allows declarative composition of widgets with horizontal/vertical splits.
 */
export interface LayoutChildConfig {
  type: string;
  props?: Record<string, unknown>;
  flex?: number;
}

export interface LayoutContainerProps {
  /** Layout direction */
  direction?: 'horizontal' | 'vertical';
  /** Gap between children */
  gap?: number;
  /** Child widget configurations */
  children: LayoutChildConfig[];
}

export const LayoutContainer = memo(function LayoutContainer({
  direction = 'horizontal',
  gap = 8,
  children,
}: LayoutContainerProps): ReactElement {
  return (
    <div style={{
      display: 'flex',
      flexDirection: direction === 'horizontal' ? 'row' : 'column',
      gap,
      height: '100%',
      width: '100%',
    }}>
      {children.map((child, index) => {
        const typeConfig = WIDGET_TYPES[child.type];
        if (!typeConfig) {
          return (
            <div key={index} style={{ flex: child.flex ?? 1, color: 'var(--theme-status-error)' }}>
              Unknown widget: {child.type}
            </div>
          );
        }

        const Component = typeConfig.component;
        const props = { ...typeConfig.defaultProps, ...child.props };

        return (
          <div key={index} style={{ flex: child.flex ?? 1, minWidth: 0, minHeight: 0, overflow: 'auto' }}>
            <Component {...props} />
          </div>
        );
      })}
    </div>
  );
});

// ========== Eval Code Editor Widget ==========

/**
 * EvalCodeEditor - Simple code editor that emits eval events
 *
 * A textarea-based code editor that emits events when code is evaluated.
 * For simple use cases - use InlineEvalEditor for full CodeMirror experience.
 */
export interface EvalCodeEditorProps {
  /** Initial code content */
  initialCode?: string;
  /** Event type to emit for eval requests */
  emitAs?: string;
  /** Placeholder text */
  placeholder?: string;
  /** Label/title */
  title?: string;
  /** Number of iterations for benchmark mode */
  iterations?: number;
}

export const EvalCodeEditor = memo(function EvalCodeEditor({
  initialCode = '',
  emitAs = 'eval.request',
  placeholder = 'Enter code to evaluate...',
  title,
  iterations = 1,
}: EvalCodeEditorProps): ReactElement {
  const [code, setCode] = useState(initialCode);
  const [isRunning, setIsRunning] = useState(false);
  const emit = useEmit();
  const hasEvalAPI = typeof window !== 'undefined' && !!window.evalAPI;

  const handleRun = useCallback(async () => {
    if (!code.trim() || isRunning) return;
    if (!window.evalAPI) {
      console.error('[EvalCodeEditor] evalAPI not available');
      return;
    }
    setIsRunning(true);

    try {
      for (let i = 0; i < iterations; i++) {
        const iterationId = `${Date.now()}-${i}`;
        emit(emitAs, { code, iteration: i, id: iterationId });

        // Actually execute the code
        const result = await window.evalAPI.execute(code, 'javascript');
        // The eval service already emits eval.result, but we can also emit here
        emit('eval.result', { ...result, iteration: i });
      }
    } finally {
      setIsRunning(false);
    }
  }, [code, iterations, emit, emitAs, isRunning]);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: 'var(--theme-bg-elevated)',
      borderRadius: 'var(--theme-radius-sm)',
      overflow: 'hidden',
    }}>
      {title && (
        <div style={{
          padding: '8px 12px',
          fontSize: '0.8em',
          color: 'var(--theme-text-muted)',
          borderBottom: '1px solid var(--theme-border-primary)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <span>{title}</span>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={handleRun}
              disabled={isRunning || !hasEvalAPI}
              title={hasEvalAPI ? 'Run code (⌘+Enter)' : 'Eval API not available - run in Electron'}
              style={{
                padding: '4px 12px',
                background: hasEvalAPI ? 'var(--theme-accent-primary)' : 'var(--theme-bg-tertiary)',
                color: hasEvalAPI ? 'var(--theme-bg-primary)' : 'var(--theme-text-muted)',
                border: 'none',
                borderRadius: 'var(--theme-radius-sm)',
                cursor: isRunning || !hasEvalAPI ? 'not-allowed' : 'pointer',
                fontSize: '0.85em',
                opacity: hasEvalAPI ? 1 : 0.6,
              }}
            >
              {isRunning ? 'Running...' : hasEvalAPI ? 'Run' : 'No Eval API'}
            </button>
          </div>
        </div>
      )}
      <textarea
        value={code}
        onChange={(e) => setCode(e.target.value)}
        placeholder={placeholder}
        style={{
          flex: 1,
          padding: '12px',
          background: 'var(--theme-code-bg, #1e1e2e)',
          color: 'var(--theme-code-text, #e0e0e0)',
          border: 'none',
          fontFamily: 'var(--theme-font-mono)',
          fontSize: 'var(--theme-font-size-sm)',
          resize: 'none',
          outline: 'none',
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            handleRun();
          }
        }}
      />
    </div>
  );
});

// ========== Event Display Widget ==========

/**
 * EventDisplay - Simple event stream display
 *
 * Shows recent events matching a pattern as a log.
 */
export interface EventDisplayProps {
  /** Event pattern to subscribe to */
  subscribePattern: string;
  /** Max events to show */
  maxEvents?: number;
  /** Title */
  title?: string;
  /** Fields to display from payload */
  fields?: string[];
}

export const EventDisplay = memo(function EventDisplay({
  subscribePattern,
  maxEvents = 10,
  title,
  fields,
}: EventDisplayProps): ReactElement {
  const events = useEventSubscription(subscribePattern, { maxEvents });

  return (
    <div style={{ ...baseWidgetStyle, padding: 0 }}>
      {title && (
        <div style={{ padding: '8px 12px', fontSize: '0.8em', color: 'var(--theme-text-muted)', borderBottom: '1px solid var(--theme-border-primary)' }}>
          {title}
        </div>
      )}
      <div style={{ maxHeight: 200, overflow: 'auto' }}>
        {events.length === 0 ? (
          <div style={{ padding: '12px', color: 'var(--theme-text-muted)', textAlign: 'center', fontSize: '0.85em' }}>
            Waiting for {subscribePattern} events...
          </div>
        ) : (
          events.map((event, i) => {
            const payload = event.payload as Record<string, unknown>;
            const displayFields = fields ?? Object.keys(payload).slice(0, 4);
            return (
              <div key={event.id ?? i} style={{
                padding: '6px 12px',
                borderBottom: '1px solid var(--theme-border-primary)',
                fontSize: '0.8em',
                fontFamily: 'var(--theme-font-mono)',
              }}>
                {displayFields.map((f) => (
                  <span key={f} style={{ marginRight: '12px' }}>
                    <span style={{ color: 'var(--theme-text-muted)' }}>{f}:</span>{' '}
                    {formatCellValue(payload[f])}
                  </span>
                ))}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
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
  'chart': {
    component: ChartWidget as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { subscribePattern: 'data.**', dataKey: 'value', chartType: 'bar' },
  },
  'table': {
    component: TableWidget as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { subscribePattern: 'data.**', columns: ['value'] },
  },
  'stats': {
    component: StatsWidget as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { subscribePattern: 'data.**', dataKey: 'value', showStats: true },
  },
  'transform': {
    component: TransformWidget as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { subscribePattern: 'data.**', emitAs: 'transformed', transform: '(p) => p' },
  },
  'layout': {
    component: LayoutContainer as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { direction: 'horizontal', gap: 8, children: [] },
  },
  'eval-editor': {
    component: EvalCodeEditor as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { title: 'Code', iterations: 1 },
  },
  'event-display': {
    component: EventDisplay as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { subscribePattern: '**', maxEvents: 10 },
  },
};
