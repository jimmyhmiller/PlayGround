/**
 * WidgetLayout
 *
 * Runtime-loadable widget layout renderer.
 * Takes a JSON configuration and renders a tree of composable widgets.
 *
 * Configuration format:
 * {
 *   "type": "layout",
 *   "direction": "horizontal" | "vertical",
 *   "gap": 8,
 *   "children": [
 *     { "type": "widget-type", "props": { ... }, "flex": 1 },
 *     { "type": "layout", "direction": "vertical", "children": [...] }
 *   ]
 * }
 *
 * Or a simple widget:
 * { "type": "chart", "props": { "subscribePattern": "eval.result" } }
 *
 * Or floating panes:
 * {
 *   "type": "floating",
 *   "panes": [
 *     { "id": "routes", "title": "Routes", "x": 20, "y": 20, "width": 200, "height": 400, "widget": { ... } }
 *   ]
 * }
 */

import { memo, useState, useEffect, useId, useRef, useCallback, createContext, useContext, type ReactElement, type CSSProperties } from 'react';
import { WIDGET_TYPES } from '../widgets/BuiltinWidgets';
import { WidgetIdContext, type WidgetIdContextValue } from '../hooks/useWidgetId';

// Re-export for backwards compatibility
export { useWidgetId } from '../hooks/useWidgetId';

// ========== Scope Context ==========
// Each WidgetLayout gets a unique scope ID that child widgets can use

const ScopeContext = createContext<string>('');

export function useScope(): string {
  return useContext(ScopeContext);
}

/**
 * Replace $scope placeholders in string values
 */
function replaceScope(value: unknown, scope: string): unknown {
  if (typeof value === 'string') {
    return value.replace(/\$scope/g, scope);
  }
  if (Array.isArray(value)) {
    return value.map(v => replaceScope(v, scope));
  }
  if (value && typeof value === 'object') {
    const result: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) {
      result[k] = replaceScope(v, scope);
    }
    return result;
  }
  return value;
}

// ========== Configuration Types ==========

export interface WidgetNodeConfig {
  type: string;
  props?: Record<string, unknown>;
  flex?: number;
  style?: CSSProperties;
}

export interface LayoutNodeConfig {
  type: 'layout';
  direction?: 'horizontal' | 'vertical';
  gap?: number;
  padding?: number;
  children: LayoutConfig[];
  flex?: number;
  style?: CSSProperties;
}

export interface FloatingPaneConfig {
  id: string;
  title?: string;
  x: number;
  y: number;
  width: number;
  height: number;
  widget: WidgetNodeConfig | LayoutNodeConfig;
  minWidth?: number;
  minHeight?: number;
}

export interface FloatingLayoutConfig {
  type: 'floating';
  panes: FloatingPaneConfig[];
}

export interface WindowConfig {
  id?: string;
  title: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  widget: WidgetNodeConfig | LayoutNodeConfig;
}

export interface WindowGroupConfig {
  type: 'window-group';
  scope?: string;
  windows: WindowConfig[];
}

export type LayoutConfig = WidgetNodeConfig | LayoutNodeConfig | FloatingLayoutConfig | WindowGroupConfig;

export interface DashboardConfig {
  name: string;
  description?: string;
  layout: LayoutConfig;
}

// ========== Type Guards ==========

function isLayoutNode(config: LayoutConfig): config is LayoutNodeConfig {
  return config.type === 'layout';
}

function isFloatingLayout(config: LayoutConfig): config is FloatingLayoutConfig {
  return config.type === 'floating';
}

function isWindowGroup(config: LayoutConfig): config is WindowGroupConfig {
  return config.type === 'window-group';
}

// ========== Widget Node Renderer ==========

interface WidgetNodeProps {
  config: WidgetNodeConfig;
  path: string;
}

const WidgetNode = memo(function WidgetNode({ config, path }: WidgetNodeProps): ReactElement {
  const scope = useContext(ScopeContext);
  const typeConfig = WIDGET_TYPES[config.type];

  if (!typeConfig) {
    return (
      <div style={{
        padding: 12,
        background: 'var(--theme-bg-elevated)',
        color: 'var(--theme-status-error)',
        borderRadius: 'var(--theme-radius-sm)',
        fontSize: '0.85em',
      }}>
        Unknown widget type: {config.type}
      </div>
    );
  }

  const Component = typeConfig.component;
  // Replace $scope placeholders in props
  const rawProps = { ...typeConfig.defaultProps, ...config.props };
  const props = replaceScope(rawProps, scope) as Record<string, unknown>;

  // Provide widget ID context for state persistence
  const widgetIdValue: WidgetIdContextValue = { path, scope };

  return (
    <WidgetIdContext.Provider value={widgetIdValue}>
      <Component {...props} />
    </WidgetIdContext.Provider>
  );
});

// ========== Layout Node Renderer ==========

interface LayoutNodeProps {
  config: LayoutNodeConfig;
  path: string;
}

const LayoutNode = memo(function LayoutNode({ config, path }: LayoutNodeProps): ReactElement {
  const {
    direction = 'horizontal',
    gap = 8,
    padding = 0,
    children = [],
    style = {},
  } = config;

  return (
    <div style={{
      display: 'flex',
      flexDirection: direction === 'horizontal' ? 'row' : 'column',
      gap,
      padding,
      height: '100%',
      width: '100%',
      ...style,
    }}>
      {children.map((child, index) => {
        // flex: 0 means "fit content" (0 0 auto), flex > 0 means "grow" (N 1 0%)
        const flexValue = child.flex ?? 1;
        const flexStyle = flexValue === 0
          ? { flex: '0 0 auto' }  // Don't grow, don't shrink, use content size
          : { flex: flexValue, minWidth: 0, minHeight: 0 };  // Grow, can shrink

        // Build child path
        const childPath = `${path}.children.${index}`;

        return (
          <div
            key={index}
            style={{
              ...flexStyle,
              overflow: 'auto',
              ...child.style,
            }}
          >
            <ConfigNode config={child} path={childPath} />
          </div>
        );
      })}
    </div>
  );
});

// ========== Floating Pane Component ==========

interface PaneState {
  x: number;
  y: number;
  width: number;
  height: number;
  zIndex: number;
}

interface FloatingPaneProps {
  config: FloatingPaneConfig;
  state: PaneState;
  path: string;
  onMove: (id: string, x: number, y: number) => void;
  onResize: (id: string, width: number, height: number) => void;
  onFocus: (id: string) => void;
}

const FloatingPane = memo(function FloatingPane({
  config,
  state,
  path,
  onMove,
  onResize,
  onFocus,
}: FloatingPaneProps): ReactElement {
  const paneRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const dragOffset = useRef({ x: 0, y: 0 });
  const resizeStart = useRef({ x: 0, y: 0, width: 0, height: 0 });

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('.pane-resize-handle')) return;
    e.preventDefault();
    onFocus(config.id);
    setIsDragging(true);
    dragOffset.current = {
      x: e.clientX - state.x,
      y: e.clientY - state.y,
    };
  }, [config.id, state.x, state.y, onFocus]);

  const handleResizeMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onFocus(config.id);
    setIsResizing(true);
    resizeStart.current = {
      x: e.clientX,
      y: e.clientY,
      width: state.width,
      height: state.height,
    };
  }, [config.id, state.width, state.height, onFocus]);

  useEffect(() => {
    if (!isDragging && !isResizing) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        const newX = Math.max(0, e.clientX - dragOffset.current.x);
        const newY = Math.max(0, e.clientY - dragOffset.current.y);
        onMove(config.id, newX, newY);
      } else if (isResizing) {
        const deltaX = e.clientX - resizeStart.current.x;
        const deltaY = e.clientY - resizeStart.current.y;
        const newWidth = Math.max(config.minWidth ?? 150, resizeStart.current.width + deltaX);
        const newHeight = Math.max(config.minHeight ?? 100, resizeStart.current.height + deltaY);
        onResize(config.id, newWidth, newHeight);
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      setIsResizing(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, isResizing, config.id, config.minWidth, config.minHeight, onMove, onResize]);

  return (
    <div
      ref={paneRef}
      onMouseDown={() => onFocus(config.id)}
      style={{
        position: 'absolute',
        left: state.x,
        top: state.y,
        width: state.width,
        height: state.height,
        zIndex: state.zIndex,
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--theme-bg-elevated)',
        borderRadius: 'var(--theme-radius-md, 8px)',
        boxShadow: '0 4px 20px rgba(0,0,0,0.3), 0 0 1px rgba(255,255,255,0.1)',
        overflow: 'hidden',
        border: '1px solid var(--theme-border-primary)',
      }}
    >
      {/* Title bar */}
      <div
        onMouseDown={handleMouseDown}
        style={{
          padding: '8px 12px',
          background: 'var(--theme-bg-tertiary)',
          borderBottom: '1px solid var(--theme-border-primary)',
          cursor: isDragging ? 'grabbing' : 'grab',
          userSelect: 'none',
          fontSize: '0.85em',
          fontWeight: 500,
          color: 'var(--theme-text-secondary)',
          flexShrink: 0,
        }}
      >
        {config.title ?? config.id}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'hidden', minHeight: 0 }}>
        <ConfigNode config={config.widget} path={`${path}.widget`} />
      </div>

      {/* Resize handle */}
      <div
        className="pane-resize-handle"
        onMouseDown={handleResizeMouseDown}
        style={{
          position: 'absolute',
          right: 0,
          bottom: 0,
          width: 16,
          height: 16,
          cursor: 'nwse-resize',
          background: 'linear-gradient(135deg, transparent 50%, var(--theme-text-muted) 50%)',
          opacity: 0.5,
          borderRadius: '0 0 var(--theme-radius-md, 8px) 0',
        }}
      />
    </div>
  );
});

// ========== Floating Layout Renderer ==========

interface FloatingLayoutProps {
  config: FloatingLayoutConfig;
  path: string;
}

const FloatingLayout = memo(function FloatingLayout({ config, path }: FloatingLayoutProps): ReactElement {
  const [paneStates, setPaneStates] = useState<Record<string, PaneState>>(() => {
    const initial: Record<string, PaneState> = {};
    config.panes.forEach((pane, index) => {
      initial[pane.id] = {
        x: pane.x,
        y: pane.y,
        width: pane.width,
        height: pane.height,
        zIndex: index + 1,
      };
    });
    return initial;
  });

  const [maxZ, setMaxZ] = useState(config.panes.length);

  const handleMove = useCallback((id: string, x: number, y: number) => {
    setPaneStates(prev => ({
      ...prev,
      [id]: { ...prev[id]!, x, y },
    }));
  }, []);

  const handleResize = useCallback((id: string, width: number, height: number) => {
    setPaneStates(prev => ({
      ...prev,
      [id]: { ...prev[id]!, width, height },
    }));
  }, []);

  const handleFocus = useCallback((id: string) => {
    setMaxZ(prev => {
      const newZ = prev + 1;
      setPaneStates(states => ({
        ...states,
        [id]: { ...states[id]!, zIndex: newZ },
      }));
      return newZ;
    });
  }, []);

  return (
    <div style={{
      position: 'relative',
      width: '100%',
      height: '100%',
      overflow: 'hidden',
    }}>
      {config.panes.map(pane => (
        <FloatingPane
          key={pane.id}
          config={pane}
          state={paneStates[pane.id]!}
          path={`${path}.pane.${pane.id}`}
          onMove={handleMove}
          onResize={handleResize}
          onFocus={handleFocus}
        />
      ))}
    </div>
  );
});

// ========== Generic Config Node Renderer ==========

interface ConfigNodeProps {
  config: LayoutConfig;
  path: string;
}

const ConfigNode = memo(function ConfigNode({ config, path }: ConfigNodeProps): ReactElement {
  if (isWindowGroup(config)) {
    // window-group is handled at the command/loader level, not here
    return (
      <div style={{ padding: 20, color: 'var(--theme-text-muted)', textAlign: 'center' }}>
        window-group must be loaded via command palette
      </div>
    );
  }
  if (isFloatingLayout(config)) {
    return <FloatingLayout config={config} path={path} />;
  }
  if (isLayoutNode(config)) {
    return <LayoutNode config={config} path={path} />;
  }
  return <WidgetNode config={config} path={path} />;
});

// ========== Main WidgetLayout Component ==========

export interface WidgetLayoutProps {
  /** Inline layout configuration */
  config?: LayoutConfig;
  /** Path to JSON config file (loaded via fileAPI) */
  configPath?: string;
  /** Dashboard config with metadata */
  dashboard?: DashboardConfig;
  /** Background color */
  background?: string;
  /** Padding around the layout */
  padding?: number;
  /** Instance/window IDs (passed by window manager) */
  instanceId?: string;
  windowId?: string;
  /** Shared scope ID - if provided, uses this instead of generating a unique one.
   *  Multiple WidgetLayouts with the same scope will share event channels. */
  scope?: string;
}

/**
 * WidgetLayout - Renders a widget tree from configuration
 *
 * Can load config from:
 * 1. Inline `config` prop
 * 2. JSON file via `configPath` prop
 * 3. Dashboard object via `dashboard` prop
 */
export const WidgetLayout = memo(function WidgetLayout({
  config,
  configPath,
  dashboard,
  background = 'var(--theme-bg-secondary)',
  padding = 8,
  scope,
  windowId,
}: WidgetLayoutProps): ReactElement {
  const [loadedConfig, setLoadedConfig] = useState<LayoutConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Use provided scope, or derive stable scope from configPath, or generate one
  const generatedScope = useId().replace(/:/g, '');
  // For file-based dashboards: use configPath as stable scope (same file = same state)
  // For dynamic dashboards without configPath: use windowId if available
  // This ensures state persists when reopening the same dashboard file
  const stableScope = configPath
    ? `file:${configPath}`
    : windowId
      ? `window:${windowId}`
      : generatedScope;
  const scopeId = scope ?? stableScope;

  // Debug logging
  console.log(`[WidgetLayout] configPath="${configPath}", windowId="${windowId}", scopeId="${scopeId}"`);

  // Load config from file if configPath is provided
  useEffect(() => {
    if (!configPath) {
      setLoadedConfig(null);
      return;
    }

    setLoading(true);
    setError(null);

    window.fileAPI?.load(configPath)
      .then((result) => {
        try {
          const parsed = JSON.parse(result.content) as DashboardConfig | LayoutConfig;
          // Handle both DashboardConfig and raw LayoutConfig
          if ('layout' in parsed) {
            setLoadedConfig(parsed.layout);
          } else {
            setLoadedConfig(parsed);
          }
          setError(null);
        } catch (parseErr) {
          setError(`Invalid JSON: ${(parseErr as Error).message}`);
        }
      })
      .catch((err: Error) => {
        setError(`Failed to load config: ${err.message}`);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [configPath]);

  // Determine which config to use
  const activeConfig = config ?? dashboard?.layout ?? loadedConfig;

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        color: 'var(--theme-text-muted)',
        background,
      }}>
        Loading configuration...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        padding: 20,
        color: 'var(--theme-status-error)',
        background,
      }}>
        {error}
      </div>
    );
  }

  if (!activeConfig) {
    return (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        gap: 12,
        color: 'var(--theme-text-muted)',
        background,
        padding: 20,
        textAlign: 'center',
      }}>
        <div style={{ fontSize: '1.2em' }}>No layout configuration</div>
        <div style={{ fontSize: '0.85em', maxWidth: 400 }}>
          Provide a layout via the <code>config</code>, <code>configPath</code>, or <code>dashboard</code> prop.
        </div>
      </div>
    );
  }

  return (
    <ScopeContext.Provider value={scopeId}>
      <div style={{
        height: '100%',
        width: '100%',
        background,
        padding,
        boxSizing: 'border-box',
      }}>
        <ConfigNode config={activeConfig} path="root" />
      </div>
    </ScopeContext.Provider>
  );
});

export default WidgetLayout;
