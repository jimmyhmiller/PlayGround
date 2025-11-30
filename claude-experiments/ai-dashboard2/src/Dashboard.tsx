import { FC } from 'react';
import type { Dashboard as DashboardType, WidgetConfig } from './types';
import { DEFAULT_THEME } from './types/theme';
import { Grid } from './components';
import { Widget } from './components/ui/Widget';
import { WIDGET_REGISTRY } from './widgets';

interface DashboardProps {
  dashboard: DashboardType;
  allDashboards?: DashboardType[];
  onSelect?: (dashboardId: string) => void;
  onWidgetResize?: (dashboardId: string, widgetId: string, dimensions: any) => void;
  onWidgetDelete?: (dashboardId: string, widgetId: string) => void;
  widgetConversations: Record<string, string | null>;
  setWidgetConversations: (update: (prev: Record<string, string | null>) => Record<string, string | null>) => void;
  dashboardVersion?: number;
  onRefreshProjects?: () => void;
  onDashboardsChange?: () => void;
}

export const Dashboard: FC<DashboardProps> = ({
  dashboard,
  onWidgetResize,
  onWidgetDelete,
  widgetConversations,
  setWidgetConversations,
  dashboardVersion
}) => {
  const theme = dashboard.theme && typeof dashboard.theme === 'object'
    ? { ...DEFAULT_THEME, ...dashboard.theme }
    : DEFAULT_THEME;

  const layout = dashboard.layoutSettings || {};
  const reloadTrigger = dashboardVersion;
  const cellSize = layout?.gridSize || 16;
  const gap = layout?.gap !== undefined ? layout.gap : 8;

  return (
    <div style={{
      width: '100%',
      height: '100vh',
      background: theme.bgApp,
      color: theme.textBody,
      fontFamily: theme.textBody
    }}>
      <Grid
        cellSize={cellSize}
        gap={gap}
        mode="infinite-canvas"
        showGrid={false}
      >
        {dashboard.widgets?.map((widget: WidgetConfig) => (
          <Widget
            key={widget.id}
            theme={theme}
            config={widget}
            dashboardId={dashboard.id}
            dashboard={dashboard}
            layout={layout}
            widgetConversations={widgetConversations}
            setWidgetConversations={setWidgetConversations}
            reloadTrigger={reloadTrigger}
            onResize={onWidgetResize}
            onDelete={onWidgetDelete}
            widgetComponents={WIDGET_REGISTRY}
          />
        ))}
      </Grid>
    </div>
  );
};
