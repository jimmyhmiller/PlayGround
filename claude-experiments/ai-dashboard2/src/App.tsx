import { Suspense, useState, useEffect } from 'react';
import type { Dashboard as DashboardType, WidgetDimensions } from './types';
import { Dashboard } from './Dashboard';
import { LoadingFallback } from './components';
import './styles.css';

function App() {
  const [dashboards, setDashboards] = useState<DashboardType[]>([]);
  const [selectedDashboardId, setSelectedDashboardId] = useState<string | null>(null);
  const [widgetConversations, setWidgetConversations] = useState<Record<string, string | null>>({});
  const [dashboardVersion, setDashboardVersion] = useState(0);
  const [isLoading, setIsLoading] = useState(true);

  // Load dashboards on mount
  useEffect(() => {
    if (window.dashboardAPI) {
      window.dashboardAPI.loadDashboards()
        .then((loadedDashboards) => {
          setDashboards(loadedDashboards);
          if (loadedDashboards.length > 0 && !selectedDashboardId) {
            setSelectedDashboardId(loadedDashboards[0].id);
          }
          setIsLoading(false);
        })
        .catch((error) => {
          console.error('Failed to load dashboards:', error);
          setIsLoading(false);
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

  const handleWidgetResize = async (dashboardId: string, widgetId: string, dimensions: Partial<WidgetDimensions>) => {
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

  if (isLoading) {
    return <LoadingFallback />;
  }

  if (dashboards.length === 0) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        flexDirection: 'column',
        gap: '16px',
        color: '#888'
      }}>
        <div style={{ fontSize: '48px' }}>📊</div>
        <div style={{ fontSize: '18px' }}>No dashboards configured</div>
        <div style={{ fontSize: '14px', opacity: 0.7 }}>
          Add a dashboard JSON file to get started
        </div>
      </div>
    );
  }

  if (!selectedDashboard) {
    return <LoadingFallback />;
  }

  const handleRefreshProjects = () => {
    if (window.dashboardAPI) {
      window.dashboardAPI.loadDashboards()
        .then((loadedDashboards) => {
          setDashboards(loadedDashboards);
          setDashboardVersion(prev => prev + 1);
        })
        .catch((error) => {
          console.error('Failed to refresh projects:', error);
        });
    }
  };

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
        onRefreshProjects={handleRefreshProjects}
        onDashboardsChange={handleRefreshProjects}
      />
    </Suspense>
  );
}

export default App;
