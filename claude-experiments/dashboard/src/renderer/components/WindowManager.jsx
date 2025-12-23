import { createContext, useContext, useCallback, memo, useRef } from 'react';
import { useWindowList, useWindowFocus, useWindowCommands, useWindow } from '../hooks/useBackendState';
import Window from './Window';

/**
 * Window Manager Context
 * Manages window state via backend, z-ordering, and focus
 */

const WindowManagerContext = createContext(null);

export function WindowManagerProvider({ children, componentRegistry }) {
  const [windows, listLoading] = useWindowList();
  const [focusedId, focusLoading] = useWindowFocus();
  const { createWindow: backendCreateWindow, closeWindow, focusWindow, updateWindow } = useWindowCommands();

  // Store component registry for looking up components by type
  const registryRef = useRef(componentRegistry);
  registryRef.current = componentRegistry;

  const createWindow = useCallback(async (options) => {
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
    return result.id;
  }, [backendCreateWindow]);

  const value = {
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

export function useWindowManager() {
  const context = useContext(WindowManagerContext);
  if (!context) {
    throw new Error('useWindowManager must be used within WindowManagerProvider');
  }
  return context;
}

/**
 * Individual window wrapper - subscribes only to its own state
 */
const WindowWrapper = memo(function WindowWrapper({
  windowId,
  componentRegistry,
  closeWindow,
  focusWindow,
  updateWindow,
}) {
  // Subscribe only to THIS window's state
  const [win] = useWindow(windowId);
  const [focusedId] = useWindowFocus();

  // Stable callback for updating props
  const updatePropsRef = useRef(null);
  if (!updatePropsRef.current) {
    updatePropsRef.current = (newProps) => {
      if (win) {
        updateWindow(windowId, { props: { ...win.props, ...newProps } });
      }
    };
  }

  // Update the callback's closure over win
  const winRef = useRef(win);
  winRef.current = win;

  const onUpdateProps = useCallback((newProps) => {
    const currentWin = winRef.current;
    if (currentWin) {
      updateWindow(windowId, { props: { ...currentWin.props, ...newProps } });
    }
  }, [windowId, updateWindow]);

  if (!win) return null;

  // Look up component from registry by type string
  const registry = componentRegistry.current || {};
  const ComponentEntry = registry[win.component];
  const Component = ComponentEntry?.component || (() => (
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
export function WindowToolbar({ componentTypes }) {
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
          onMouseEnter={(e) => e.target.style.background = '#4d4d4d'}
          onMouseLeave={(e) => e.target.style.background = '#3d3d3d'}
        >
          {icon && <span>{icon}</span>}
          {label}
        </button>
      ))}
    </div>
  );
}
