import { FC } from 'react';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';
import type { NestedDashboardConfig } from '../types/dashboard';
import { Grid } from '../components';
import { Widget } from '../components/ui/Widget';
import { WIDGET_REGISTRY } from './index';
import { DashboardProvider, useDashboardContext } from '../contexts/DashboardContext';

export const NestedDashboard: FC<BaseWidgetComponentProps & { isDropTarget?: boolean; onResize?: any; onDelete?: any; onTransfer?: any }> = (props) => {
  const { theme, config, dashboardId, widgetConversations, reloadTrigger, isDropTarget = false, onResize, onDelete, onTransfer } = props;
  const nestedConfig = config as NestedDashboardConfig;
  const parentContext = useDashboardContext();

  // Get the nested dashboard configuration
  const nestedDashboard = nestedConfig.dashboard;

  if (!nestedDashboard) {
    return (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: theme.textBody,
          color: '#fff',
          opacity: 0.6,
        }}
      >
        No nested dashboard configured
      </div>
    );
  }

  // Merge parent theme with nested dashboard theme
  const effectiveTheme = nestedDashboard.theme
    ? { ...theme, ...nestedDashboard.theme }
    : theme;

  // Get layout settings
  const nestedLayout = nestedDashboard.layout || nestedDashboard.layoutSettings || {};
  const cellSize = nestedLayout?.gridSize || 16;
  const gapX = nestedLayout?.widgetGap || 8;
  const gapY = nestedLayout?.widgetGap || 8;
  const layoutMode = (nestedLayout?.mode || 'single-pane') as 'single-pane' | 'vertical-scroll' | 'horizontal-scroll' | 'infinite-canvas';

  // Calculate nesting depth
  const nestingDepth = (parentContext?.nestingDepth ?? 0) + 1;

  // Build widget path for scoping
  const widgetPath = parentContext?.widgetPath
    ? `${parentContext.widgetPath}.${config.id}`
    : config.id;

  // Prevent excessive nesting
  if (nestingDepth > 5) {
    return (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: theme.textBody,
          color: theme.negative,
          padding: '20px',
          textAlign: 'center',
        }}
      >
        ⚠️ Maximum nesting depth (5) exceeded
      </div>
    );
  }

  const handleDoubleClick = () => {
    console.log('🎯 Double-click detected on nested dashboard!', {
      nestedDashboard,
      parentDashboardId: dashboardId,
      widgetId: config.id,
    });

    // Emit custom event to tell App.tsx to navigate to this nested dashboard
    const event = new CustomEvent('navigate-to-nested-dashboard', {
      detail: {
        dashboard: nestedDashboard,
        parentDashboardId: dashboardId,
        widgetId: config.id,
        parentTheme: theme,
      },
      bubbles: true,
    });
    window.dispatchEvent(event);
    console.log('✅ Event dispatched:', event);
  };

  // Drop zone styles with pulse animation
  const dropTargetStyles = isDropTarget ? {
    border: `2px solid ${effectiveTheme.accent || '#00d9ff'}`,
    boxShadow: `0 0 20px ${effectiveTheme.accent || '#00d9ff'}44`,
    animation: 'pulse 1s ease-in-out infinite',
  } : {
    border: '1px solid rgba(255,255,255,0.1)',
  };

  return (
    <DashboardProvider
      theme={effectiveTheme}
      layout={nestedLayout}
      nestingDepth={nestingDepth}
      widgetPath={widgetPath}
    >
      <div
        data-nested-dashboard="true"
        data-widget-id={config.id}
        onDoubleClick={handleDoubleClick}
        style={{
          width: '100%',
          height: '100%',
          position: 'relative',
          overflow: layoutMode === 'infinite-canvas' ? 'hidden' : 'auto',
          backgroundColor: effectiveTheme.bgApp || 'transparent',
          cursor: 'pointer',
          ...dropTargetStyles,
          transition: 'border 200ms ease-in-out, box-shadow 200ms ease-in-out',
        }}
      >
        {isDropTarget && (
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              padding: '20px 40px',
              backgroundColor: `${effectiveTheme.accent || '#00d9ff'}22`,
              border: `2px dashed ${effectiveTheme.accent || '#00d9ff'}`,
              borderRadius: '12px',
              color: effectiveTheme.accent || '#00d9ff',
              fontFamily: effectiveTheme.textBody,
              fontSize: '18px',
              fontWeight: 'bold',
              zIndex: 1000,
              pointerEvents: 'none',
              backdropFilter: 'blur(4px)',
            }}
          >
            Drop here to move widget
          </div>
        )}
        <Grid
          cellSize={cellSize}
          gapX={gapX}
          gapY={gapY}
          mode={layoutMode}
          width="100%"
          height="100%"
        >
          {nestedDashboard.widgets?.map((widget) => {
            // Create scoped widget key
            const scopedWidgetKey = `${dashboardId}.${widgetPath}.${widget.id}`;

            return (
              <Widget
                key={scopedWidgetKey}
                theme={effectiveTheme}
                config={widget}
                dashboardId={dashboardId}
                dashboard={nestedDashboard}
                layout={nestedLayout}
                widgetConversations={widgetConversations || {}}
                setWidgetConversations={props.setCurrentConversationId ? () => {} : () => {}}
                reloadTrigger={reloadTrigger}
                onResize={onResize}
                onDelete={onDelete}
                onTransfer={onTransfer}
                widgetComponents={WIDGET_REGISTRY}
              />
            );
          })}
        </Grid>
      </div>
    </DashboardProvider>
  );
};
