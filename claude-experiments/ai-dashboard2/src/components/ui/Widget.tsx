import { FC, useState, useEffect, MouseEvent } from 'react';
import type { Theme, WidgetConfig, Dashboard, LayoutSettings, WidgetDimensions } from '../../types';
import { ErrorBoundary } from './ErrorBoundary';
import { GridItem } from '../GridItem';

export interface BaseWidgetComponentProps {
  theme: Theme;
  config: WidgetConfig;
  dashboardId: string;
  dashboard?: Dashboard;
  layout?: LayoutSettings;
  widgetKey?: string;
  currentConversationId?: string | null;
  setCurrentConversationId?: (id: string | null) => void;
  widgetConversations?: Record<string, string | null>;
  reloadTrigger?: number;
}

interface WidgetProps {
  theme: Theme;
  config: WidgetConfig;
  clipPath?: string;
  onResize?: (dashboardId: string, widgetId: string, dimensions: Partial<WidgetDimensions>) => void;
  onDelete?: (dashboardId: string, widgetId: string) => void;
  dashboardId: string;
  dashboard?: Dashboard;
  layout?: LayoutSettings;
  allWidgets?: WidgetConfig[];
  widgetConversations: Record<string, string | null>;
  setWidgetConversations: (update: (prev: Record<string, string | null>) => Record<string, string | null>) => void;
  reloadTrigger?: number;
  widgetComponents: Record<string, FC<BaseWidgetComponentProps>>;
}

const defaultWidgetDimensions: Record<string, { w: number; h: number }> = {
  chat: { w: 400, h: 500 },
  'bar-chart': { w: 300, h: 200 },
  stat: { w: 200, h: 150 },
  progress: { w: 250, h: 100 },
  'file-list': { w: 250, h: 200 },
  'todo-list': { w: 250, h: 200 },
  'claude-todo-list': { w: 250, h: 200 },
  'key-value': { w: 200, h: 150 },
  'diff-list': { w: 300, h: 200 },
  'layout-settings': { w: 250, h: 200 },
  'command-runner': { w: 350, h: 250 },
  'code-editor': { w: 600, h: 500 },
  webview: { w: 400, h: 400 },
};

function parseSize(size: string | number | undefined, defaultSize: number): number {
  if (!size) return defaultSize;
  if (typeof size === 'string') {
    return parseInt(size.replace('px', ''));
  }
  return size;
}

export const Widget: FC<WidgetProps> = ({
  theme,
  config,
  clipPath,
  onResize,
  onDelete,
  dashboardId,
  dashboard,
  layout,
  widgetConversations,
  setWidgetConversations,
  reloadTrigger,
  widgetComponents
}) => {
  const Component = widgetComponents[config.type];
  const [showContextMenu, setShowContextMenu] = useState(false);
  const [contextMenuPos, setContextMenuPos] = useState({ x: 0, y: 0 });
  const [isRegenerating, setIsRegenerating] = useState(false);

  const unknownWidgetContent = !Component ? (
    <div
      className="widget"
      style={{
        background: theme.widgetBg,
        border: '2px solid ' + theme.negative,
        borderRadius: theme.widgetRadius,
        clipPath: clipPath,
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px',
        textAlign: 'center',
      }}
    >
      <div style={{ fontSize: '48px', marginBottom: '16px' }}>⚠️</div>
      <div style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '8px', color: theme.negative }}>
        Unknown Widget Type
      </div>
      <div style={{ fontSize: '14px', opacity: 0.7, marginBottom: '8px' }}>
        Type: "{config.type}"
      </div>
      <div style={{ fontSize: '12px', opacity: 0.5 }}>
        Widget ID: {config.id}
      </div>
    </div>
  ) : null;

  const isChat = config.type === 'chat';
  const widgetKey = `${dashboardId}-${config.id}`;
  const hasRegenerate =
    ('regenerateCommand' in config && !!config.regenerateCommand) ||
    ('regenerateScript' in config && !!config.regenerateScript) ||
    ('regenerate' in config && !!config.regenerate);

  const handleDrag = ({ x, y }: { x?: number; y?: number; width?: number; height?: number }) => {
    if (onResize && dashboardId && (x !== undefined || y !== undefined)) {
      onResize(dashboardId, config.id, { x, y });
    }
  };

  const handleResizeEnd = ({ width, height }: { x?: number; y?: number; width?: number; height?: number }) => {
    if (onResize && dashboardId && (width !== undefined || height !== undefined)) {
      onResize(dashboardId, config.id, { width, height });
    }
  };

  const handleMouseDown = (e: MouseEvent<HTMLDivElement>) => {
    // Check for ctrl + option + command (Mac)
    if (e.metaKey && e.ctrlKey && e.altKey) {
      e.preventDefault();
      e.stopPropagation();
      if (onDelete && dashboardId) {
        onDelete(dashboardId, config.id);
      }
    }
  };

  const handleContextMenu = (e: MouseEvent<HTMLDivElement>) => {
    console.log('Context menu triggered, hasRegenerate:', hasRegenerate, 'config:', config.id);
    if (!hasRegenerate) return;
    e.preventDefault();
    e.stopPropagation();
    setContextMenuPos({ x: e.clientX, y: e.clientY });
    setShowContextMenu(true);
    console.log('Context menu shown at:', e.clientX, e.clientY);
  };

  const handleRegenerate = async () => {
    console.log('Regenerating widget:', config.id);
    setShowContextMenu(false);
    setIsRegenerating(true);
    try {
      if (!window.dashboardAPI || !window.dashboardAPI.regenerateWidget) {
        console.error('dashboardAPI.regenerateWidget not available');
        return;
      }
      await window.dashboardAPI.regenerateWidget(dashboardId, config.id);
      console.log('Regenerate succeeded');
    } catch (error: any) {
      console.error('Regenerate error:', error);
      alert(`Regenerate error: ${error.message}`);
    } finally {
      setIsRegenerating(false);
    }
  };

  // Close context menu when clicking outside
  useEffect(() => {
    if (showContextMenu) {
      const handleClick = () => setShowContextMenu(false);
      document.addEventListener('click', handleClick);
      return () => document.removeEventListener('click', handleClick);
    }
  }, [showContextMenu]);

  const widgetContent = (
    <div
      className={`widget ${isChat ? 'chat-widget' : ''}`}
      onMouseDown={handleMouseDown}
      onContextMenu={handleContextMenu}
      style={{
        background: theme.widgetBg,
        border: theme.widgetBorder,
        borderRadius: theme.widgetRadius,
        clipPath: clipPath,
        width: '100%',
        height: '100%',
        position: 'relative',
        opacity: isRegenerating ? 0.6 : 1,
        transition: 'opacity 0.2s',
      }}
    >
      {unknownWidgetContent || (
        <ErrorBoundary theme={theme}>
          <Component
            theme={theme}
            config={config}
            dashboardId={dashboardId}
            dashboard={dashboard}
            layout={layout}
            widgetKey={widgetKey}
            currentConversationId={widgetConversations[widgetKey] || null}
            setCurrentConversationId={(id: string | null) => setWidgetConversations(prev => ({ ...prev, [widgetKey]: id }))}
            widgetConversations={widgetConversations}
            reloadTrigger={reloadTrigger}
          />
        </ErrorBoundary>
      )}
      {/* Context Menu */}
      {showContextMenu && (
        <div
          style={{
            position: 'fixed',
            left: contextMenuPos.x,
            top: contextMenuPos.y,
            background: 'rgba(30, 30, 30, 0.95)',
            border: `1px solid ${theme.accent}`,
            borderRadius: 6,
            padding: 4,
            zIndex: 10000,
            minWidth: 150,
            boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <div
            onClick={handleRegenerate}
            style={{
              padding: '8px 12px',
              cursor: 'pointer',
              color: theme.accent,
              fontSize: '0.85rem',
              fontFamily: theme.textBody,
              transition: 'background-color 0.2s',
              borderRadius: 4,
            }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.1)')}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
          >
            ⟳ Regenerate
          </div>
        </div>
      )}
    </div>
  );

  const defaults = defaultWidgetDimensions[config.type] || { w: 250, h: 200 };
  const dims = config.dimensions || config;

  // Always use GridItem for draggable/resizable widgets
  return (
    <GridItem
      x={dims.x || 0}
      y={dims.y || 0}
      width={parseSize(dims.w || dims.width, defaults.w)}
      height={parseSize(dims.h || dims.height, defaults.h)}
      resizable={true}
      draggable={true}
      onDrag={handleDrag}
      onDragEnd={handleDrag}
      onResize={handleResizeEnd}
    >
      {widgetContent}
    </GridItem>
  );
};
