import { Suspense, useState, useEffect } from 'react';
import type { Dashboard as DashboardType } from './types';
import { Dashboard } from './Dashboard';
import { LoadingFallback } from './components';
import './styles.css';

function App() {
  const [dashboards, setDashboards] = useState<DashboardType[]>([]);
  const [selectedDashboardId, setSelectedDashboardId] = useState<string | null>(null);
  const [widgetConversations, setWidgetConversations] = useState<Record<string, string | null>>({});
  const [dashboardVersion, setDashboardVersion] = useState(0);

  // Load dashboards on mount
  useEffect(() => {
    if (window.dashboardAPI) {
      window.dashboardAPI.loadDashboards()
        .then((loadedDashboards) => {
          setDashboards(loadedDashboards);
          if (loadedDashboards.length > 0 && !selectedDashboardId) {
            setSelectedDashboardId(loadedDashboards[0].id);
          }
        })
        .catch((error) => {
          console.error('Failed to load dashboards:', error);
        });

      // Listen for dashboard updates
      window.dashboardAPI.onDashboardUpdate((updatedDashboards) => {
        setDashboards(updatedDashboards);
        setDashboardVersion(prev => prev + 1);
      });

      window.dashboardAPI.onError((error) => {
        console.error('Dashboard error:', error);
      });
    }
  }, []);

  const selectedDashboard = dashboards.find(d => d.id === selectedDashboardId);

  const handleWidgetResize = async (dashboardId: string, widgetId: string, dimensions: any) => {
    if (window.dashboardAPI) {
      try {
        await window.dashboardAPI.updateWidgetDimensions(dashboardId, widgetId, dimensions);
      } catch (error) {
        console.error('Failed to update widget dimensions:', error);
      }
    }
  };

  const handleWidgetDelete = async (dashboardId: string, widgetId: string) => {
    if (window.dashboardAPI) {
      try {
        await window.dashboardAPI.deleteWidget(dashboardId, widgetId);
      } catch (error) {
        console.error('Failed to delete widget:', error);
      }
    }
  };

  if (!selectedDashboard) {
    return <LoadingFallback />;
  }

  return (
    <Suspense fallback={<LoadingFallback />}>
      <Dashboard
        dashboard={selectedDashboard}
        allDashboards={dashboards}
        onSelect={setSelectedDashboardId}
        onWidgetResize={handleWidgetResize}
        onWidgetDelete={handleWidgetDelete}
        widgetConversations={widgetConversations}
        setWidgetConversations={setWidgetConversations}
        dashboardVersion={dashboardVersion}
      />
    </Suspense>
  );
}

export default App;
