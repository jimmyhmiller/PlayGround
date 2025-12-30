import { createContext, useContext, useCallback, memo, useRef, ReactNode, ComponentType, RefObject } from 'react';
import { useWindowList, useWindowFocus, useWindowCommands, useWindow } from '../hooks/useBackendState';
import Window from './Window';
import type { WindowUpdates } from '../../types/components';
import type { WindowState } from '../../types/state';

/**
 * Window Manager Context
 * Manages window state via backend, z-ordering, and focus
 */

interface ComponentRegistryEntry {
  component: ComponentType<any>;
}

type ComponentRegistry = Record<string, ComponentRegistryEntry>;

interface WindowManagerContextValue {
  windows: WindowState[];
  focusedId: string | null;
  createWindow: (options: CreateWindowOptions) => Promise<string>;
  closeWindow: (id: string) => Promise<unknown>;
  focusWindow: (id: string) => Promise<unknown>;
  updateWindow: (id: string, updates: WindowUpdates) => Promise<unknown>;
  loading: boolean;
  componentRegistry: RefObject<ComponentRegistry>;
}

interface CreateWindowOptions {
  title?: string;
  componentType?: string;
  component?: ComponentType<any> & { displayName?: string };
  props?: Record<string, unknown>;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

const WindowManagerContext = createContext<WindowManagerContextValue | null>(null);

export interface WindowManagerProviderProps {
  children: ReactNode;
  componentRegistry: ComponentRegistry;
}

export function WindowManagerProvider({ children, componentRegistry }: WindowManagerProviderProps) {
  const [windows, listLoading] = useWindowList();
  const [focusedId, focusLoading] = useWindowFocus();
  const { createWindow: backendCreateWindow, closeWindow, focusWindow, updateWindow } = useWindowCommands();

  // Store component registry for looking up components by type
  const registryRef = useRef(componentRegistry);
  registryRef.current = componentRegistry;

  const createWindow = useCallback(async (options: CreateWindowOptions): Promise<string> => {
    // Store the component type string, not the component itself
    const result = await backendCreateWindow({
      title: options.title,
      component: options.componentType || options.component?.displayName || 'Unknown',
      props: options.props || {},
      x: options.x,
      y: options.y,
      width: options.width,
      height: options.height,
    });
    return result.id ?? '';
  }, [backendCreateWindow]);

  const value: WindowManagerContextValue = {
    windows,
    focusedId,
    createWindow,
    closeWindow,
    focusWindow,
    updateWindow,
    loading: listLoading || focusLoading,
    componentRegistry: registryRef,
  };

  return (
    <WindowManagerContext.Provider value={value}>
      {children}
    </WindowManagerContext.Provider>
  );
}

export function useWindowManager(): WindowManagerContextValue {
  const context = useContext(WindowManagerContext);
  if (!context) {
    throw new Error('useWindowManager must be used within WindowManagerProvider');
  }
  return context;
}

/**
 * Individual window wrapper - subscribes only to its own state
 */
interface WindowWrapperProps {
  windowId: string;
  componentRegistry: RefObject<ComponentRegistry>;
  closeWindow: (id: string) => Promise<unknown>;
  focusWindow: (id: string) => Promise<unknown>;
  updateWindow: (id: string, updates: WindowUpdates) => Promise<unknown>;
}

interface DynamicComponentProps {
  windowId: string;
  onUpdateProps: (newProps: Record<string, unknown>) => void;
  [key: string]: any;
}

const WindowWrapper = memo(function WindowWrapper({
  windowId,
  componentRegistry,
  closeWindow,
  focusWindow,
  updateWindow,
}: WindowWrapperProps) {
  // Subscribe only to THIS window's state
  const [win] = useWindow(windowId);
  const [focusedId] = useWindowFocus();

  // Update the callback's closure over win
  const winRef = useRef(win);
  winRef.current = win;

  const onUpdateProps = useCallback((newProps: Record<string, unknown>) => {
    const currentWin = winRef.current;
    if (currentWin) {
      updateWindow(windowId, { props: { ...currentWin.props, ...newProps } });
    }
  }, [windowId, updateWindow]);

  if (!win) return null;

  // Look up component from registry by type string
  const registry = componentRegistry.current || {};
  const ComponentEntry = registry[win.component];
  const Component: ComponentType<DynamicComponentProps> = ComponentEntry?.component || (() => (
    <div style={{ padding: '16px', color: 'var(--theme-text-muted)' }}>
      Unknown component: {win.component}
    </div>
  ));

  return (
    <Window
      id={win.id}
      title={win.title}
      x={win.x}
      y={win.y}
      width={win.width}
      height={win.height}
      zIndex={win.zIndex}
      isFocused={win.id === focusedId}
      pinned={win.pinned}
      onClose={closeWindow}
      onFocus={focusWindow}
      onUpdate={updateWindow}
    >
      <Component
        {...win.props}
        windowId={win.id}
        onUpdateProps={onUpdateProps}
      />
    </Window>
  );
});

/**
 * Renders all managed windows
 * Only re-renders when the window LIST changes (add/remove), not when individual windows update
 */
export const WindowContainer = memo(function WindowContainer() {
  const { windows, closeWindow, focusWindow, updateWindow, componentRegistry } = useWindowManager();

  // Just get the list of window IDs - each WindowWrapper handles its own state
  const windowIds = windows.map(w => w.id);

  return (
    <div style={{
      position: 'relative',
      width: '100%',
      height: '100%',
      overflow: 'hidden',
    }}>
      {windowIds.map((windowId) => (
        <WindowWrapper
          key={windowId}
          windowId={windowId}
          componentRegistry={componentRegistry}
          closeWindow={closeWindow}
          focusWindow={focusWindow}
          updateWindow={updateWindow}
        />
      ))}
    </div>
  );
});

/**
 * Toolbar for creating windows
 */
interface ComponentTypeInfo {
  type: string;
  label: string;
  defaultProps?: Record<string, unknown>;
  icon?: string;
}

export interface WindowToolbarProps {
  componentTypes: ComponentTypeInfo[];
}

export function WindowToolbar({ componentTypes }: WindowToolbarProps) {
  const { createWindow } = useWindowManager();

  return (
    <div style={{
      display: 'flex',
      gap: '8px',
      padding: '12px',
      background: '#2d2d2d',
      borderBottom: '1px solid #3d3d3d',
    }}>
      {componentTypes.map(({ type, label, defaultProps, icon }) => (
        <button
          key={type}
          onClick={() => createWindow({
            title: label,
            componentType: type,
            props: defaultProps,
          })}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '8px 12px',
            background: '#3d3d3d',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            fontSize: '12px',
            cursor: 'pointer',
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = '#4d4d4d')}
          onMouseLeave={(e) => (e.currentTarget.style.background = '#3d3d3d')}
        >
          {icon && <span>{icon}</span>}
          {label}
        </button>
      ))}
    </div>
  );
}
