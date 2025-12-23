import { memo, useState, useMemo } from 'react';
import { WindowManagerProvider, WindowContainer, useWindowManager } from './WindowManager';
import { ThemeProvider } from '../theme/ThemeProvider';
import { SettingsProvider } from '../settings/SettingsProvider';
import CodeMirrorEditor from './CodeMirrorEditor';
import GitDiffViewer from './GitDiffViewer';
import EventLogPanel from './EventLogPanel';
import ThemeEditor from './ThemeEditor';
import SettingsEditor from './SettingsEditor';

/**
 * Component registry - maps type strings to components
 * This is needed because we store component types as strings in backend state
 */
const COMPONENT_REGISTRY = {
  'codemirror': {
    component: CodeMirrorEditor,
    label: 'Code Editor',
    icon: '📝',
    defaultProps: { subscribePattern: 'file.**' },
    width: 600,
    height: 400,
  },
  'git-diff': {
    component: GitDiffViewer,
    label: 'Git Diff',
    icon: '📊',
    defaultProps: { subscribePattern: 'git.**' },
    width: 500,
    height: 400,
  },
  'event-log': {
    component: EventLogPanel,
    label: 'Event Log',
    icon: '📋',
    defaultProps: { subscribePattern: '**' },
    width: 450,
    height: 350,
  },
  'theme-editor': {
    component: ThemeEditor,
    label: 'Theme Editor',
    icon: '🎨',
    defaultProps: {},
    width: 320,
    height: 500,
  },
  'settings': {
    component: SettingsEditor,
    label: 'Settings',
    icon: '⚙️',
    defaultProps: {},
    width: 300,
    height: 280,
  },
};

/**
 * Window types for toolbar (derived from registry)
 */
const WINDOW_TYPES = Object.entries(COMPONENT_REGISTRY).map(([type, config]) => ({
  type,
  ...config,
}));

/**
 * Toolbar component inside the provider
 */
const DesktopToolbar = memo(function DesktopToolbar() {
  const { createWindow, windows } = useWindowManager();
  const [filePath, setFilePath] = useState('');

  const handleCreateWindow = (type) => {
    const config = COMPONENT_REGISTRY[type];
    if (!config) return;

    createWindow({
      title: config.label,
      componentType: type,
      props: {
        ...config.defaultProps,
        ...(type === 'codemirror' && filePath ? { filePath } : {}),
      },
      width: config.width,
      height: config.height,
    });
  };

  const handleLoadFile = async () => {
    if (!filePath) return;
    await window.fileAPI.load(filePath);
  };

  const handleWatchFile = async () => {
    if (!filePath) return;
    await window.fileAPI.watch(filePath);
  };

  const handleRefreshGit = async () => {
    await window.gitAPI.refresh();
  };

  const handleStartPolling = async () => {
    await window.gitAPI.startPolling(2000);
  };

  return (
    <div
      className="desktop-toolbar"
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        gap: 'var(--theme-spacing-sm)',
        padding: 'var(--theme-spacing-sm) var(--theme-spacing-md)',
        background: 'var(--theme-toolbar-bg)',
        borderBottom: '1px solid var(--theme-toolbar-border)',
      }}
    >
      {/* Window creation buttons */}
      <div style={{ display: 'flex', gap: '6px' }}>
        {WINDOW_TYPES.map((config) => (
          <button
            key={config.type}
            className="toolbar-button"
            onClick={() => handleCreateWindow(config.type)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '5px',
              padding: '6px 10px',
              background: 'var(--theme-toolbar-button-bg)',
              border: '1px solid var(--theme-toolbar-button-border)',
              borderRadius: 'var(--theme-radius-sm)',
              color: 'var(--theme-text-secondary)',
              fontSize: 'var(--theme-font-size-md)',
              cursor: 'pointer',
              transition: 'background var(--theme-transition-fast)',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'var(--theme-toolbar-button-hover)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'var(--theme-toolbar-button-bg)';
            }}
          >
            <span>{config.icon}</span>
            <span>{config.label}</span>
          </button>
        ))}
      </div>

      <div
        className="toolbar-divider"
        style={{
          width: '1px',
          height: '24px',
          background: 'var(--theme-border-secondary)',
          margin: '0 var(--theme-spacing-sm)',
        }}
      />

      {/* File input */}
      <input
        type="text"
        className="toolbar-input"
        value={filePath}
        onChange={(e) => setFilePath(e.target.value)}
        placeholder="File path..."
        style={{
          padding: '6px 10px',
          background: 'var(--theme-bg-input)',
          border: '1px solid var(--theme-border-secondary)',
          borderRadius: 'var(--theme-radius-sm)',
          color: 'var(--theme-text-primary)',
          fontSize: 'var(--theme-font-size-md)',
          width: '200px',
        }}
      />
      <button
        className="toolbar-action"
        onClick={handleLoadFile}
        style={{
          padding: '6px 10px',
          background: 'var(--theme-accent-primary)',
          border: 'none',
          borderRadius: 'var(--theme-radius-sm)',
          color: '#fff',
          fontSize: 'var(--theme-font-size-md)',
          cursor: 'pointer',
        }}
      >
        Load
      </button>
      <button
        className="toolbar-action"
        onClick={handleWatchFile}
        style={{
          padding: '6px 10px',
          background: 'var(--theme-accent-warning)',
          border: 'none',
          borderRadius: 'var(--theme-radius-sm)',
          color: '#fff',
          fontSize: 'var(--theme-font-size-md)',
          cursor: 'pointer',
        }}
      >
        Watch
      </button>

      <div
        className="toolbar-divider"
        style={{
          width: '1px',
          height: '24px',
          background: 'var(--theme-border-secondary)',
          margin: '0 var(--theme-spacing-sm)',
        }}
      />

      {/* Git controls */}
      <button
        className="toolbar-action"
        onClick={handleRefreshGit}
        style={{
          padding: '6px 10px',
          background: 'var(--theme-bg-tertiary)',
          border: '1px solid var(--theme-border-secondary)',
          borderRadius: 'var(--theme-radius-sm)',
          color: 'var(--theme-text-secondary)',
          fontSize: 'var(--theme-font-size-md)',
          cursor: 'pointer',
        }}
      >
        Git Refresh
      </button>
      <button
        className="toolbar-action"
        onClick={handleStartPolling}
        style={{
          padding: '6px 10px',
          background: 'var(--theme-accent-info)',
          border: 'none',
          borderRadius: 'var(--theme-radius-sm)',
          color: '#fff',
          fontSize: 'var(--theme-font-size-md)',
          cursor: 'pointer',
        }}
      >
        Poll Git
      </button>

      {/* Window count */}
      <div style={{
        marginLeft: 'auto',
        fontSize: 'var(--theme-font-size-sm)',
        color: 'var(--theme-text-disabled)',
      }}>
        {windows.length} window{windows.length !== 1 ? 's' : ''}
      </div>
    </div>
  );
});

/**
 * Desktop environment with windowing and theming
 */
function Desktop() {
  return (
    <SettingsProvider>
      <ThemeProvider>
        <WindowManagerProvider componentRegistry={COMPONENT_REGISTRY}>
          <div
            className="desktop"
            style={{
              display: 'flex',
              flexDirection: 'column',
              height: '100vh',
              background: 'var(--theme-bg-primary)',
              fontFamily: 'var(--theme-font-family)',
              color: 'var(--theme-text-primary)',
            }}
          >
            <DesktopToolbar />
            <div
              className="desktop-workspace"
              style={{ flex: 1, position: 'relative' }}
            >
              <WindowContainer />
            </div>
          </div>
        </WindowManagerProvider>
      </ThemeProvider>
    </SettingsProvider>
  );
}

export default Desktop;
