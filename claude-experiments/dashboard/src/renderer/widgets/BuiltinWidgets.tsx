/**
 * Built-in Widget Types
 *
 * Generic, configurable widget components that can be instantiated at runtime.
 * Each widget type accepts props that configure its behavior.
 */

import React, { memo, useCallback, useState, useEffect, useMemo, useRef, useId, type ReactElement } from 'react';
import { useBackendStateSelector, useDispatch, useBackendState } from '../hooks/useBackendState';
import { useEventSubscription, useEmit, useEventReducer } from '../hooks/useEvents';
import { usePersistentState } from '../hooks/useWidgetState';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';

// CodeMirror imports for syntax highlighting
import { EditorView, lineNumbers, highlightActiveLineGutter, highlightSpecialChars } from '@codemirror/view';
import { EditorState, Extension } from '@codemirror/state';
import { syntaxHighlighting, defaultHighlightStyle } from '@codemirror/language';
import { javascript } from '@codemirror/lang-javascript';
import { html } from '@codemirror/lang-html';
import { python } from '@codemirror/lang-python';
import { oneDark } from '@codemirror/theme-one-dark';

// Pipeline widgets
import {
  FileDrop,
  Pipeline,
  PipelineStatus,
  ProcessorList,
  InlinePipeline,
} from './PipelineWidgets';

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
  /** Key to extract numeric value from event payload (auto-detected if not specified) */
  dataKey?: string;
  /** Key to extract label from event payload (auto-detected if not specified) */
  labelKey?: string;
  /** Chart type */
  chartType?: 'bar' | 'line';
  /** Max data points to show */
  maxPoints?: number;
  /** Chart height */
  height?: number;
  /** Title */
  title?: string;
  /** Show column selectors */
  showSelectors?: boolean;
}

// Helper to detect if a value looks numeric (including string numbers)
function isNumericValue(val: unknown): boolean {
  if (typeof val === 'number') return true;
  if (typeof val === 'string') {
    const trimmed = val.replace(/[$%,]/g, '').trim();
    // Must be entirely numeric (not just start with a number like "2023-08")
    return trimmed !== '' && !isNaN(Number(trimmed)) && isFinite(Number(trimmed));
  }
  return false;
}

// Helper to parse numeric value from string or number
function parseNumericValue(val: unknown): number {
  if (typeof val === 'number') return val;
  if (typeof val === 'string') {
    const trimmed = val.replace(/[$%,]/g, '').trim();
    return parseFloat(trimmed) || 0;
  }
  return 0;
}

export const ChartWidget = memo(function ChartWidget({
  subscribePattern,
  dataKey: configDataKey,
  labelKey: configLabelKey,
  chartType = 'bar',
  maxPoints = 20,
  height = 150,
  title,
  showSelectors = true,
}: ChartWidgetProps): ReactElement {
  const events = useEventSubscription(subscribePattern, { maxEvents: maxPoints });

  // Persist selected keys - usePersistentState is a drop-in for useState
  const [selectedDataKey, setSelectedDataKey] = usePersistentState<string | null>('selectedDataKey', null);
  const [selectedLabelKey, setSelectedLabelKey] = usePersistentState<string | null>('selectedLabelKey', null);

  // Detect available columns from first event
  const { numericColumns, labelColumns } = useMemo(() => {
    if (events.length === 0) {
      return { numericColumns: [] as string[], labelColumns: [] as string[] };
    }

    const payload = events[0].payload as Record<string, unknown>;
    if (!payload || typeof payload !== 'object') {
      return { numericColumns: [] as string[], labelColumns: [] as string[] };
    }

    const numeric: string[] = [];
    const labels: string[] = [];

    for (const [key, value] of Object.entries(payload)) {
      if (isNumericValue(value)) {
        numeric.push(key);
      } else if (typeof value === 'string') {
        labels.push(key);
      }
    }

    return { numericColumns: numeric, labelColumns: labels };
  }, [events]);

  // Determine active keys (config > selected > auto-detected)
  const dataKey = configDataKey || selectedDataKey || numericColumns[0] || 'value';
  const labelKey = configLabelKey || selectedLabelKey || labelColumns[0] || 'name';

  const chartData = useMemo(() => {
    return events.map((event, index) => {
      const payload = event.payload as Record<string, unknown>;
      return {
        name: payload[labelKey] ?? `#${index + 1}`,
        value: parseNumericValue(payload[dataKey]),
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

  const selectStyle: React.CSSProperties = {
    padding: '2px 6px',
    fontSize: '0.75em',
    background: 'var(--theme-bg-tertiary)',
    color: 'var(--theme-text-primary)',
    border: '1px solid var(--theme-border-primary)',
    borderRadius: 'var(--theme-radius-sm)',
    cursor: 'pointer',
  };

  return (
    <div style={{ ...baseWidgetStyle, padding: 0 }}>
      <div style={{
        padding: '8px 12px',
        fontSize: '0.8em',
        color: 'var(--theme-text-muted)',
        borderBottom: '1px solid var(--theme-border-primary)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        gap: 8,
      }}>
        <span>{title || 'Chart'}</span>
        {showSelectors && numericColumns.length > 0 && (
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ fontSize: '0.9em' }}>Y:</span>
              <select
                value={dataKey}
                onChange={(e) => setSelectedDataKey(e.target.value)}
                style={selectStyle}
              >
                {numericColumns.map((col) => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </label>
            {labelColumns.length > 0 && (
              <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ fontSize: '0.9em' }}>X:</span>
                <select
                  value={labelKey}
                  onChange={(e) => setSelectedLabelKey(e.target.value)}
                  style={selectStyle}
                >
                  {labelColumns.map((col) => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </label>
            )}
          </div>
        )}
      </div>
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
  /** Columns to display (keys from payload). If empty/omitted, auto-detects from data. */
  columns?: string[];
  /** Column headers (optional, defaults to column keys) */
  headers?: string[];
  /** Max rows to show */
  maxRows?: number;
  /** Title */
  title?: string;
}

export const TableWidget = memo(function TableWidget({
  subscribePattern,
  columns: configColumns,
  headers,
  maxRows = 10,
  title,
}: TableWidgetProps): ReactElement {
  const events = useEventSubscription(subscribePattern, { maxEvents: maxRows });

  // Auto-detect columns from first event if not specified
  const columns = useMemo(() => {
    if (configColumns && configColumns.length > 0) {
      return configColumns;
    }
    // Detect from first event's payload
    if (events.length > 0) {
      const firstPayload = events[0].payload as Record<string, unknown>;
      if (firstPayload && typeof firstPayload === 'object') {
        return Object.keys(firstPayload);
      }
    }
    return [];
  }, [configColumns, events]);

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
      const rawValue = payload[dataKey];
      const value = parseNumericValue(rawValue);

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
  /** Placeholder text */
  placeholder?: string;
  /** Label/title */
  title?: string;
  /** Number of iterations for benchmark mode */
  iterations?: number;
  /** Language for evaluation (defaults to 'javascript') */
  language?: string;
  /** Command to run for this language (registers executor automatically) */
  command?: string;
  /** Arguments for the command */
  args?: string[];
  /** Unique channel for this editor's results (e.g., "editor1" -> emits to "eval.result.editor1") */
  channel?: string;
}

export const EvalCodeEditor = memo(function EvalCodeEditor({
  initialCode = '',
  placeholder = 'Enter code to evaluate...',
  title,
  iterations = 1,
  language = 'javascript',
  command,
  args,
  channel,
}: EvalCodeEditorProps): ReactElement {
  // Persist code - usePersistentState is a drop-in for useState
  const [code, setCode] = usePersistentState('code', initialCode);
  const [isRunning, setIsRunning] = useState(false);
  const [executorReady, setExecutorReady] = useState(false);
  const emit = useEmit();
  const hasEvalAPI = typeof window !== 'undefined' && !!window.evalAPI;

  // Auto-generate unique instance ID for this editor
  const instanceId = useId();
  const effectiveChannel = channel ?? `auto-${instanceId.replace(/:/g, '')}`;

  // Register executor on mount if command is provided
  useEffect(() => {
    if (!command || !window.evalAPI) {
      // No command needed (built-in JS) or no API
      setExecutorReady(language === 'javascript' || language === 'typescript');
      return;
    }

    window.evalAPI.registerExecutor({ language, command, args })
      .then(() => {
        setExecutorReady(true);
        console.log(`[EvalCodeEditor] Registered executor for ${language}: ${command}`);
      })
      .catch((err) => {
        console.error(`[EvalCodeEditor] Failed to register executor:`, err);
      });
  }, [language, command, args]);

  // Determine the event channel for results
  const resultChannel = `eval.result.${effectiveChannel}`;

  const handleRun = useCallback(async () => {
    if (!code.trim() || isRunning) return;
    if (!window.evalAPI) {
      console.error('[EvalCodeEditor] evalAPI not available');
      return;
    }
    setIsRunning(true);

    try {
      for (let i = 0; i < iterations; i++) {
        // Actually execute the code
        const result = await window.evalAPI.execute(code, language);
        // Emit to the specific channel for this editor
        emit(resultChannel, { ...result, iteration: i });
      }
    } finally {
      setIsRunning(false);
    }
  }, [code, iterations, emit, isRunning, language, resultChannel]);

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
              disabled={isRunning || !hasEvalAPI || !executorReady}
              title={!hasEvalAPI ? 'Eval API not available - run in Electron' : !executorReady ? `Registering ${language} executor...` : 'Run code (⌘+Enter)'}
              style={{
                padding: '4px 12px',
                background: hasEvalAPI && executorReady ? 'var(--theme-accent-primary)' : 'var(--theme-bg-tertiary)',
                color: hasEvalAPI && executorReady ? 'var(--theme-bg-primary)' : 'var(--theme-text-muted)',
                border: 'none',
                borderRadius: 'var(--theme-radius-sm)',
                cursor: isRunning || !hasEvalAPI || !executorReady ? 'not-allowed' : 'pointer',
                fontSize: '0.85em',
                opacity: hasEvalAPI && executorReady ? 1 : 0.6,
              }}
            >
              {isRunning ? 'Running...' : !hasEvalAPI ? 'No Eval API' : !executorReady ? 'Loading...' : 'Run'}
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

// ========== Selector Widget ==========

/**
 * Selector - Clickable item list that emits selection events
 *
 * Items can come from:
 * 1. Static `items` prop
 * 2. Subscribed events (uses payload as items array)
 */
export interface SelectorProps {
  /** Static items */
  items?: Array<Record<string, unknown>>;
  /** Subscribe to events for items (payload should be an array) */
  subscribePattern?: string;
  /** Path to extract items from payload (e.g., "0.data" for payload[0].data) */
  dataPath?: string;
  /** Key for item label */
  labelKey?: string;
  /** Template for label using ${key} syntax */
  labelTemplate?: string;
  /** Key for unique item ID */
  idKey?: string;
  /** Channel to emit selection to */
  channel: string;
  /** Title */
  title?: string;
  /** Layout direction */
  direction?: 'horizontal' | 'vertical';
}

export const Selector = memo(function Selector({
  items: staticItems,
  subscribePattern,
  dataPath,
  labelKey = 'label',
  labelTemplate,
  idKey = 'id',
  channel,
  title,
  direction = 'vertical',
}: SelectorProps): ReactElement {
  // Persist selection - usePersistentState is a drop-in for useState
  const [selectedId, setSelectedId] = usePersistentState<string | null>('selectedId', null);
  const events = useEventSubscription(subscribePattern ?? '__none__', { maxEvents: 1 });
  const emit = useEmit();

  // Get items from static prop or most recent event
  const items = useMemo(() => {
    if (staticItems) return staticItems;
    if (events.length > 0) {
      let payload: unknown = events[0]?.payload;
      // Navigate to nested data if dataPath is specified
      if (dataPath && payload) {
        const parts = dataPath.split('.');
        for (const part of parts) {
          if (payload && typeof payload === 'object') {
            payload = (payload as Record<string, unknown>)[part];
          } else {
            payload = undefined;
            break;
          }
        }
      }
      if (Array.isArray(payload)) return payload;
    }
    return [];
  }, [staticItems, events, dataPath]);

  const handleSelect = useCallback((item: Record<string, unknown>) => {
    const id = String(item[idKey] ?? '');
    setSelectedId(id);
    emit(`selection.${channel}`, item);
  }, [idKey, channel, emit]);

  const renderLabel = useCallback((item: Record<string, unknown>) => {
    if (labelTemplate) {
      return labelTemplate.replace(/\$\{(\w+)\}/g, (_, key) => String(item[key] ?? ''));
    }
    return String(item[labelKey] ?? '');
  }, [labelTemplate, labelKey]);

  return (
    <div style={{ ...baseWidgetStyle, padding: 0 }}>
      {title && (
        <div style={{ padding: '8px 12px', fontSize: '0.8em', color: 'var(--theme-text-muted)', borderBottom: '1px solid var(--theme-border-primary)' }}>
          {title}
        </div>
      )}
      <div style={{
        display: 'flex',
        flexDirection: direction === 'horizontal' ? 'row' : 'column',
        gap: 4,
        padding: 8,
        flexWrap: direction === 'horizontal' ? 'wrap' : 'nowrap',
        overflow: 'auto',
      }}>
        {items.length === 0 ? (
          <div style={{ padding: '12px', color: 'var(--theme-text-muted)', textAlign: 'center', fontSize: '0.85em' }}>
            No items
          </div>
        ) : (
          items.map((item, i) => {
            const id = String(item[idKey] ?? i);
            const isSelected = selectedId === id;
            return (
              <button
                key={id}
                onClick={() => handleSelect(item)}
                style={{
                  padding: '6px 12px',
                  background: isSelected ? 'var(--theme-accent-primary)' : 'var(--theme-bg-tertiary)',
                  color: isSelected ? 'var(--theme-bg-primary)' : 'var(--theme-text-primary)',
                  border: '1px solid var(--theme-border-primary)',
                  borderRadius: 'var(--theme-radius-sm)',
                  cursor: 'pointer',
                  fontSize: '0.85em',
                  textAlign: 'left',
                }}
              >
                {renderLabel(item)}
              </button>
            );
          })
        )}
      </div>
    </div>
  );
});

// ========== CodeBlock Widget ==========

/**
 * CodeBlock - Read-only code display
 *
 * Content can come from:
 * 1. Static `code` prop
 * 2. File path via `file` prop
 * 3. Subscribed event payload
 */
export interface CodeBlockProps {
  /** Static code content */
  code?: string;
  /** File path to load */
  file?: string;
  /** Subscribe to events for file path to load */
  filePattern?: string;
  /** Key in event payload containing file path */
  fileKey?: string;
  /** Subscribe to events for code (uses payload[codeKey] or payload as string) */
  subscribePattern?: string;
  /** Key in event payload containing code */
  codeKey?: string;
  /** Title */
  title?: string;
  /** Show line numbers */
  lineNumbers?: boolean;
  /** Background color */
  background?: string;
  /** Language hint for display */
  language?: string;
}

// Get language extension based on language name
function getLanguageExtension(lang?: string): Extension {
  const langLower = lang?.toLowerCase() ?? '';
  if (langLower === 'python' || langLower === 'py') return python();
  if (langLower === 'javascript' || langLower === 'js') return javascript();
  if (langLower === 'typescript' || langLower === 'ts') return javascript({ typescript: true });
  if (langLower === 'html' || langLower === 'jinja' || langLower === 'jinja2') return html();
  return [];
}

export const CodeBlock = memo(function CodeBlock({
  code: staticCode,
  file: staticFile,
  filePattern,
  fileKey = 'file',
  subscribePattern,
  codeKey,
  title,
  lineNumbers: showLineNumbers = true,
  background,
  language,
}: CodeBlockProps): ReactElement {
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Persist current file - usePersistentState is a drop-in for useState
  const [currentFile, setCurrentFile] = usePersistentState<string | null>('currentFile', null);

  const codeEvents = useEventSubscription(subscribePattern ?? '__none__', { maxEvents: 1 });
  const fileEvents = useEventSubscription(filePattern ?? '__none__', { maxEvents: 1 });
  const editorRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<EditorView | null>(null);

  // Determine which file to load (from static prop or event)
  const fileToLoad = useMemo(() => {
    if (staticFile) return staticFile;
    if (fileEvents.length > 0) {
      const payload = fileEvents[0]?.payload;
      if (typeof payload === 'string') return payload;
      if (payload && typeof payload === 'object') {
        return String((payload as Record<string, unknown>)[fileKey] ?? '');
      }
    }
    return null;
  }, [staticFile, fileEvents, fileKey]);

  // Load file content when file path changes
  useEffect(() => {
    if (!fileToLoad || fileToLoad === currentFile) return;
    setLoading(true);
    setCurrentFile(fileToLoad);
    window.fileAPI?.load(fileToLoad)
      .then((result) => {
        if (result?.content) {
          setFileContent(result.content);
        } else {
          setFileContent(null);
        }
      })
      .catch(() => setFileContent(null))
      .finally(() => setLoading(false));
  }, [fileToLoad, currentFile]);

  // Determine code to display
  const code = useMemo(() => {
    if (staticCode) return staticCode;
    if (fileContent) return fileContent;
    if (codeEvents.length > 0) {
      const payload = codeEvents[0]?.payload;
      if (typeof payload === 'string') return payload;
      if (payload && typeof payload === 'object' && codeKey) {
        return String((payload as Record<string, unknown>)[codeKey] ?? '');
      }
    }
    return '';
  }, [staticCode, fileContent, codeEvents, codeKey]);

  // Initialize/update CodeMirror
  useEffect(() => {
    if (!editorRef.current || !code) return;

    // Destroy previous view if exists
    if (viewRef.current) {
      viewRef.current.destroy();
      viewRef.current = null;
    }

    const extensions: Extension[] = [
      EditorView.editable.of(false),
      EditorState.readOnly.of(true),
      highlightSpecialChars(),
      syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
      oneDark,
      getLanguageExtension(language),
      EditorView.theme({
        '&': { height: '100%', fontSize: '13px' },
        '.cm-scroller': { overflow: 'auto' },
        '.cm-content': { padding: '8px 0' },
        '.cm-line': { padding: '0 12px' },
      }),
    ];

    if (showLineNumbers) {
      extensions.push(lineNumbers());
    }

    const state = EditorState.create({
      doc: code,
      extensions,
    });

    viewRef.current = new EditorView({
      state,
      parent: editorRef.current,
    });

    return () => {
      if (viewRef.current) {
        viewRef.current.destroy();
        viewRef.current = null;
      }
    };
  }, [code, language, showLineNumbers]);

  // Display title with file path if available
  const displayTitle = title ?? (currentFile ? currentFile.split('/').pop() : undefined);

  return (
    <div style={{
      ...baseWidgetStyle,
      padding: 0,
      background: background ?? 'var(--theme-bg-elevated)',
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      {displayTitle && (
        <div style={{
          padding: '8px 12px',
          fontSize: '0.8em',
          color: 'var(--theme-text-muted)',
          borderBottom: '1px solid var(--theme-border-primary)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexShrink: 0,
        }}>
          <span>{displayTitle}</span>
          {language && <span style={{ opacity: 0.6, fontSize: '0.9em' }}>{language}</span>}
        </div>
      )}
      <div style={{ flex: 1, overflow: 'hidden', minHeight: 0 }}>
        {loading ? (
          <div style={{ padding: '12px', color: 'var(--theme-text-muted)', textAlign: 'center' }}>Loading...</div>
        ) : !code ? (
          <div style={{ padding: '12px', color: 'var(--theme-text-muted)', textAlign: 'center', fontSize: '0.85em' }}>
            No code to display
          </div>
        ) : (
          <div ref={editorRef} style={{ height: '100%' }} />
        )}
      </div>
    </div>
  );
});

// ========== WebView Widget ==========

/**
 * WebView - Electron webview for embedded web content
 * Uses the Electron webview tag for full browser capabilities
 */
export interface WebViewProps {
  /** Base URL */
  url?: string;
  /** Subscribe to events for URL path (appended to base url) */
  subscribePattern?: string;
  /** Key in event payload containing path */
  pathKey?: string;
  /** Title */
  title?: string;
  /** Height in pixels (ignored if flex parent) */
  height?: number;
}

export const WebView = memo(function WebView({
  url: baseUrl,
  subscribePattern,
  pathKey = 'path',
  title,
  height = 300,
}: WebViewProps): ReactElement {
  const webviewRef = useRef<HTMLWebViewElement>(null);
  const events = useEventSubscription(subscribePattern ?? '__none__', { maxEvents: 1 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Persist current URL - usePersistentState is a drop-in for useState
  const [currentUrl, setCurrentUrl] = usePersistentState('currentUrl', '');

  // Determine URL
  const url = useMemo(() => {
    let path = '';
    if (events.length > 0) {
      const payload = events[0]?.payload;
      if (typeof payload === 'string') path = payload;
      else if (payload && typeof payload === 'object') {
        path = String((payload as Record<string, unknown>)[pathKey] ?? '');
      }
    }
    if (baseUrl && path) return `${baseUrl}${path}`;
    if (baseUrl) return baseUrl;
    return '';
  }, [baseUrl, events, pathKey]);

  // Handle webview events
  useEffect(() => {
    const webview = webviewRef.current;
    if (!webview) return;

    const handleLoadStart = () => {
      setLoading(true);
      setError(null);
    };
    const handleLoadStop = () => setLoading(false);
    const handleDidNavigate = (e: Event) => {
      const evt = e as Event & { url: string };
      setCurrentUrl(evt.url);
      setError(null);
    };
    const handleDidFail = (e: Event) => {
      const evt = e as Event & { errorCode: number; errorDescription: string; validatedURL: string };
      setLoading(false);
      if (evt.errorCode !== -3) { // -3 is ERR_ABORTED, often happens during redirects
        setError(`Failed to load: ${evt.errorDescription || 'Connection refused'}`);
      }
    };

    webview.addEventListener('did-start-loading', handleLoadStart);
    webview.addEventListener('did-stop-loading', handleLoadStop);
    webview.addEventListener('did-navigate', handleDidNavigate);
    webview.addEventListener('did-navigate-in-page', handleDidNavigate);
    webview.addEventListener('did-fail-load', handleDidFail);

    return () => {
      webview.removeEventListener('did-start-loading', handleLoadStart);
      webview.removeEventListener('did-stop-loading', handleLoadStop);
      webview.removeEventListener('did-navigate', handleDidNavigate);
      webview.removeEventListener('did-navigate-in-page', handleDidNavigate);
      webview.removeEventListener('did-fail-load', handleDidFail);
    };
  }, []);

  const handleRefresh = useCallback(() => {
    const webview = webviewRef.current;
    if (webview) {
      webview.reload();
    }
  }, []);

  return (
    <div style={{
      ...baseWidgetStyle,
      padding: 0,
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
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
          gap: 8,
        }}>
          <span>{title} {loading && '(loading...)'}</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {currentUrl && <span style={{ opacity: 0.5, fontSize: '0.85em', maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{currentUrl}</span>}
            <button
              onClick={handleRefresh}
              disabled={loading}
              style={{
                padding: '2px 8px',
                background: 'var(--theme-bg-tertiary)',
                border: '1px solid var(--theme-border-primary)',
                borderRadius: 'var(--theme-radius-sm)',
                cursor: loading ? 'not-allowed' : 'pointer',
                fontSize: '0.85em',
                color: 'var(--theme-text-secondary)',
              }}
            >
              Refresh
            </button>
          </div>
        </div>
      )}
      {url ? (
        <div
          style={{ flex: 1, position: 'relative', minHeight: height, background: '#fff' }}
        >
          <webview
            ref={webviewRef as React.RefObject<HTMLWebViewElement>}
            src={url}
            style={{
              width: '100%',
              height: '100%',
              minHeight: height,
              display: 'flex',
            }}
          />
          {error && (
            <div style={{
              position: 'absolute',
              inset: 0,
              background: 'var(--theme-bg-elevated)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 12,
              padding: 20,
            }}>
              <div style={{ color: 'var(--theme-status-error)', fontSize: '0.9em', textAlign: 'center' }}>
                {error}
              </div>
              <div style={{ color: 'var(--theme-text-muted)', fontSize: '0.8em' }}>
                Make sure the server is running
              </div>
              <button
                onClick={handleRefresh}
                style={{
                  padding: '6px 16px',
                  background: 'var(--theme-accent-primary)',
                  color: 'white',
                  border: 'none',
                  borderRadius: 'var(--theme-radius-sm)',
                  cursor: 'pointer',
                  fontSize: '0.85em',
                }}
              >
                Retry
              </button>
            </div>
          )}
        </div>
      ) : (
        <div style={{ padding: '12px', color: 'var(--theme-text-muted)', textAlign: 'center', fontSize: '0.85em', flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          No URL specified
        </div>
      )}
    </div>
  );
});

// Legacy alias
export const WebFrame = WebView;

// ========== FileLoader Widget ==========

/**
 * FileLoader - Loads files and emits their content as events
 *
 * Useful for loading config files, source files, etc.
 * Can apply a transform to parse the content.
 */
export interface FileLoaderProps {
  /** File paths to load */
  files: string[];
  /** Channel to emit loaded content to */
  channel: string;
  /** Transform function (receives content string, returns parsed data) */
  transform?: string;
  /** Reload interval in ms (0 = no auto-reload) */
  reloadInterval?: number;
}

export const FileLoader = memo(function FileLoader({
  files,
  channel,
  transform,
  reloadInterval = 0,
}: FileLoaderProps): ReactElement {
  const [loading, setLoading] = useState(false);
  const [lastLoad, setLastLoad] = useState<Date | null>(null);
  const emit = useEmit();

  // Parse transform function - receives (content, filePath, allFiles)
  const transformFn = useMemo(() => {
    if (!transform) return null;
    try {
      // eslint-disable-next-line no-new-func
      return new Function('content', 'filePath', 'allFiles', `return (${transform})(content, filePath, allFiles)`) as (content: string, filePath: string, allFiles: Array<{ file: string; content: string }>) => unknown;
    } catch (err) {
      console.error('[FileLoader] Failed to parse transform:', err);
      return null;
    }
  }, [transform]);

  const loadFiles = useCallback(async () => {
    setLoading(true);
    try {
      // First, load all files
      const loadedFiles: Array<{ file: string; content: string }> = [];
      for (const file of files) {
        try {
          const result = await window.fileAPI?.load(file);
          if (result?.content) {
            loadedFiles.push({ file, content: result.content });
          }
        } catch (err) {
          console.error('[FileLoader] File load error:', file, err);
        }
      }

      // Then transform each file, passing all files for cross-reference
      const results: Array<{ file: string; content: string; data?: unknown }> = [];
      for (const { file, content } of loadedFiles) {
        let data: unknown;
        try {
          data = transformFn ? transformFn(content, file, loadedFiles) : content;
        } catch (err) {
          console.error('[FileLoader] Transform error:', err);
          data = content;
        }
        results.push({ file, content, data });
      }

      emit(`loaded.${channel}`, results);
      setLastLoad(new Date());
    } finally {
      setLoading(false);
    }
  }, [files, channel, emit, transformFn]);

  // Initial load
  useEffect(() => {
    loadFiles();
  }, [loadFiles]);

  // Auto-reload
  useEffect(() => {
    if (reloadInterval <= 0) return;
    const interval = setInterval(loadFiles, reloadInterval);
    return () => clearInterval(interval);
  }, [reloadInterval, loadFiles]);

  return (
    <div style={{
      padding: '4px 8px',
      fontSize: '0.75em',
      color: 'var(--theme-text-muted)',
      background: 'var(--theme-bg-elevated)',
      borderRadius: 'var(--theme-radius-sm)',
      display: 'inline-flex',
      alignItems: 'center',
      gap: 8,
    }}>
      <span>{loading ? 'Loading...' : `${files.length} file(s)`}</span>
      {lastLoad && <span>@ {lastLoad.toLocaleTimeString()}</span>}
      <button
        onClick={loadFiles}
        disabled={loading}
        style={{
          padding: '2px 6px',
          background: 'var(--theme-bg-tertiary)',
          border: '1px solid var(--theme-border-primary)',
          borderRadius: 'var(--theme-radius-sm)',
          cursor: 'pointer',
          fontSize: '0.9em',
        }}
      >
        Reload
      </button>
    </div>
  );
});

// ========== ProcessRunner Widget ==========

/**
 * ProcessRunner - Start/stop a process with a button
 */
export interface ProcessRunnerProps {
  /** Unique ID for this process */
  id: string;
  /** Command to run */
  command: string;
  /** Command arguments */
  args?: string[];
  /** Working directory */
  cwd?: string;
  /** Button label when stopped */
  startLabel?: string;
  /** Button label when running */
  stopLabel?: string;
  /** Title/description */
  title?: string;
  /** Show output log */
  showOutput?: boolean;
  /** Max output lines to keep */
  maxOutputLines?: number;
}

export const ProcessRunner = memo(function ProcessRunner({
  id,
  command,
  args = [],
  cwd,
  startLabel = 'Start',
  stopLabel = 'Stop',
  title,
  showOutput = true,
  maxOutputLines = 100,
}: ProcessRunnerProps): ReactElement {
  const [running, setRunning] = useState(false);
  const [output, setOutput] = useState<string[]>([]);
  const outputRef = useRef<HTMLDivElement>(null);

  // Check initial running state
  useEffect(() => {
    window.shellAPI?.isRunning(id).then(({ running: isRunning }) => {
      setRunning(isRunning);
    });
  }, [id]);

  // Subscribe to process output
  useEffect(() => {
    const unsubStdout = window.eventAPI?.subscribe(`shell.stdout.${id}`, (event) => {
      const { data } = event.payload as { data: string };
      setOutput(prev => [...prev.slice(-(maxOutputLines - 1)), ...data.split('\n').filter(Boolean)]);
    });

    const unsubStderr = window.eventAPI?.subscribe(`shell.stderr.${id}`, (event) => {
      const { data } = event.payload as { data: string };
      setOutput(prev => [...prev.slice(-(maxOutputLines - 1)), ...data.split('\n').filter(Boolean).map(l => `[err] ${l}`)]);
    });

    const unsubExit = window.eventAPI?.subscribe(`shell.exit.${id}`, (event) => {
      const { code } = event.payload as { code: number | null };
      setRunning(false);
      setOutput(prev => [...prev, `[Process exited with code ${code}]`]);
    });

    const unsubError = window.eventAPI?.subscribe(`shell.error.${id}`, (event) => {
      const { error } = event.payload as { error: string };
      setRunning(false);
      setOutput(prev => [...prev, `[Error: ${error}]`]);
    });

    return () => {
      unsubStdout?.();
      unsubStderr?.();
      unsubExit?.();
      unsubError?.();
    };
  }, [id, maxOutputLines]);

  // Auto-scroll output
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [output]);

  const handleToggle = useCallback(async () => {
    if (running) {
      await window.shellAPI?.kill(id);
      setRunning(false);
    } else {
      setOutput([`$ ${command} ${args.join(' ')}`]);
      const result = await window.shellAPI?.spawn(id, command, args, { cwd });
      if (result?.success) {
        setRunning(true);
      }
    }
  }, [running, id, command, args, cwd]);

  return (
    <div style={{
      ...baseWidgetStyle,
      padding: 0,
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
    }}>
      <div style={{
        padding: '8px 12px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: showOutput ? '1px solid var(--theme-border-primary)' : 'none',
      }}>
        <span style={{ fontSize: '0.85em', color: 'var(--theme-text-muted)' }}>
          {title ?? command}
        </span>
        <button
          onClick={handleToggle}
          style={{
            padding: '4px 12px',
            background: running ? 'var(--theme-status-error)' : 'var(--theme-accent-primary)',
            color: 'white',
            border: 'none',
            borderRadius: 'var(--theme-radius-sm)',
            cursor: 'pointer',
            fontSize: '0.8em',
            fontWeight: 500,
          }}
        >
          {running ? stopLabel : startLabel}
        </button>
      </div>
      {showOutput && (
        <div
          ref={outputRef}
          style={{
            flex: 1,
            overflow: 'auto',
            padding: '8px 12px',
            fontFamily: 'var(--theme-font-mono)',
            fontSize: '0.75em',
            lineHeight: 1.4,
            background: 'var(--theme-bg-primary)',
            color: 'var(--theme-text-secondary)',
          }}
        >
          {output.length === 0 ? (
            <span style={{ color: 'var(--theme-text-muted)' }}>No output yet</span>
          ) : (
            output.map((line, i) => (
              <div key={i} style={{ color: line.startsWith('[err]') ? 'var(--theme-status-error)' : 'inherit' }}>
                {line}
              </div>
            ))
          )}
        </div>
      )}
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
    defaultProps: { subscribePattern: 'data.**', chartType: 'bar' },
  },
  'table': {
    component: TableWidget as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { subscribePattern: 'data.**' },
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
  'selector': {
    component: Selector as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { direction: 'vertical', labelKey: 'label', idKey: 'id' },
  },
  'code-block': {
    component: CodeBlock as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { lineNumbers: true },
  },
  'webview': {
    component: WebView as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { height: 300 },
  },
  'web-frame': {
    component: WebView as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { height: 300 },
  },
  'file-loader': {
    component: FileLoader as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { files: [], reloadInterval: 0 },
  },
  'process-runner': {
    component: ProcessRunner as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { showOutput: true, maxOutputLines: 100 },
  },
  // Pipeline widgets
  'file-drop': {
    component: FileDrop as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { title: 'Drop file here', showInfo: true },
  },
  'pipeline': {
    component: Pipeline as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { autoStart: true, showStatus: false },
  },
  'pipeline-status': {
    component: PipelineStatus as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { showAll: true },
  },
  'processor-list': {
    component: ProcessorList as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: {},
  },
  'inline-pipeline': {
    component: InlinePipeline as unknown as React.ComponentType<Record<string, unknown>>,
    defaultProps: { autoStart: true },
  },
};
