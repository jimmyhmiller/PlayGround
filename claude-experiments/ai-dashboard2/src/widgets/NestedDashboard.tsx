import { FC } from 'react';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';
import type { NestedDashboardConfig } from '../types/dashboard';
import { Grid } from '../components';
import { Widget } from '../components/ui/Widget';
import { WIDGET_REGISTRY } from './index';
import { DashboardProvider, useDashboardContext } from '../contexts/DashboardContext';

export const NestedDashboard: FC<BaseWidgetComponentProps> = (props) => {
  const { theme, config, dashboardId, widgetConversations, reloadTrigger } = props;
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
        onDoubleClick={handleDoubleClick}
        style={{
          width: '100%',
          height: '100%',
          position: 'relative',
          overflow: layoutMode === 'infinite-canvas' ? 'hidden' : 'auto',
          backgroundColor: effectiveTheme.bgApp || 'transparent',
          cursor: 'pointer',
        }}
      >
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
                widgetComponents={WIDGET_REGISTRY}
              />
            );
          })}
        </Grid>
      </div>
    </DashboardProvider>
  );
};
