import { Suspense, useState, useEffect } from 'react';
import type { Dashboard as DashboardType, WidgetDimensions } from './types';
import { Dashboard } from './Dashboard';
import { LoadingFallback } from './components';
import './styles.css';

interface NavigationStackItem {
  dashboardId: string;
  widgetId: string;
  parentTheme: any;
}

function App() {
  const [dashboards, setDashboards] = useState<DashboardType[]>([]);
  const [selectedDashboardId, setSelectedDashboardId] = useState<string | null>(null);
  const [widgetConversations, setWidgetConversations] = useState<Record<string, string | null>>({});
  const [dashboardVersion, setDashboardVersion] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [navigationStack, setNavigationStack] = useState<NavigationStackItem[]>([]);
  const [viewingNestedDashboard, setViewingNestedDashboard] = useState<DashboardType | null>(null);

  // Load dashboards on mount
  useEffect(() => {
    if (window.dashboardAPI) {
      window.dashboardAPI.loadDashboards()
        .then((loadedDashboards) => {
          setDashboards(Array.isArray(loadedDashboards) ? loadedDashboards : []);
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
        setDashboards(Array.isArray(updatedDashboards) ? updatedDashboards : []);
        setDashboardVersion(prev => prev + 1);
      });

      window.dashboardAPI.onError((error) => {
        console.error('Dashboard error:', error);
      });
    }
  }, []);

  // Listen for nested dashboard navigation
  useEffect(() => {
    const handleNavigateToNested = (event: any) => {
      console.log('🔍 Navigate to nested dashboard event received:', event.detail);
      const { dashboard, parentDashboardId, widgetId, parentTheme } = event.detail;

      // Merge parent theme with nested dashboard theme
      const mergedDashboard = {
        ...dashboard,
        theme: parentTheme ? { ...parentTheme, ...dashboard.theme } : dashboard.theme,
      };

      console.log('📊 Setting viewing nested dashboard:', mergedDashboard);

      // Only push to navigation stack if we're not already viewing a nested dashboard
      // This prevents duplicates when double-clicking while already in nested view
      setNavigationStack(prev => {
        // If already viewing nested, don't add duplicate
        if (prev.length > 0) {
          return prev;
        }
        return [...prev, { dashboardId: parentDashboardId, widgetId, parentTheme }];
      });
      setViewingNestedDashboard(mergedDashboard);
    };

    window.addEventListener('navigate-to-nested-dashboard', handleNavigateToNested);
    return () => window.removeEventListener('navigate-to-nested-dashboard', handleNavigateToNested);
  }, []);

  // Exit nested dashboard view when switching to a different dashboard
  useEffect(() => {
    if (viewingNestedDashboard) {
      setViewingNestedDashboard(null);
      setNavigationStack([]);
    }
  }, [selectedDashboardId]);

  const selectedDashboard = dashboards.find(d => d.id === selectedDashboardId);

  const handleDashboardSelect = (dashboardId: string) => {
    // Exit nested dashboard view whenever any dashboard is selected (even if it's the same one)
    if (viewingNestedDashboard) {
      setViewingNestedDashboard(null);
      setNavigationStack([]);
    }
    setSelectedDashboardId(dashboardId);
  };

  const handleWidgetResize = async (dashboardId: string, widgetId: string, dimensions: Partial<WidgetDimensions>) => {
    if (window.dashboardAPI) {
      try {
        // If we're viewing a nested dashboard, we need to update the nested widget within the parent
        if (viewingNestedDashboard && navigationStack.length > 0) {
          const parentItem = navigationStack[navigationStack.length - 1];
          const parentDashboard = Array.isArray(dashboards)
            ? dashboards.find(d => d.id === parentItem.dashboardId)
            : undefined;

          if (parentDashboard) {
            const nestedWidget = parentDashboard.widgets.find((w: any) => w.id === parentItem.widgetId);
            if (nestedWidget && (nestedWidget as any).dashboard) {
              const widgetInNested = (nestedWidget as any).dashboard.widgets.find((w: any) => w.id === widgetId);
              if (widgetInNested) {
                Object.assign(widgetInNested, dimensions);
                await window.dashboardAPI.updateWidget(parentDashboard.id, parentItem.widgetId, nestedWidget);
                return;
              }
            }
          }
        }

        await window.dashboardAPI.updateWidgetDimensions(dashboardId, widgetId, dimensions);
      } catch (error) {
        console.error('Failed to update widget dimensions:', error);
      }
    }
  };

  const handleWidgetDelete = async (dashboardId: string, widgetId: string) => {
    if (window.dashboardAPI) {
      try {
        // If we're viewing a nested dashboard, we need to delete from the nested dashboard
        if (viewingNestedDashboard && navigationStack.length > 0) {
          const parentItem = navigationStack[navigationStack.length - 1];
          const parentDashboard = Array.isArray(dashboards)
            ? dashboards.find(d => d.id === parentItem.dashboardId)
            : undefined;

          if (parentDashboard) {
            const nestedWidget = parentDashboard.widgets.find((w: any) => w.id === parentItem.widgetId);
            if (nestedWidget && (nestedWidget as any).dashboard) {
              (nestedWidget as any).dashboard.widgets = (nestedWidget as any).dashboard.widgets.filter((w: any) => w.id !== widgetId);
              await window.dashboardAPI.updateWidget(parentDashboard.id, parentItem.widgetId, nestedWidget);
              return;
            }
          }
        }

        await window.dashboardAPI.deleteWidget(dashboardId, widgetId);
      } catch (error) {
        console.error('Failed to delete widget:', error);
      }
    }
  };

  const handleGoBack = () => {
    if (navigationStack.length > 0) {
      setNavigationStack(prev => prev.slice(0, -1));
      setViewingNestedDashboard(null);
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
          setDashboards(Array.isArray(loadedDashboards) ? loadedDashboards : []);
          setDashboardVersion(prev => prev + 1);
        })
        .catch((error) => {
          console.error('Failed to refresh projects:', error);
        });
    }
  };

  // Determine which dashboard to show
  const displayDashboard = viewingNestedDashboard || selectedDashboard;

  // Build breadcrumbs for nested dashboard navigation
  const breadcrumbs = viewingNestedDashboard && navigationStack.length > 0
    ? navigationStack.map((item) => {
        const parentDash = Array.isArray(dashboards)
          ? dashboards.find(d => d.id === item.dashboardId)
          : undefined;
        return {
          title: parentDash?.title || 'Dashboard',
          onClick: () => {
            // Navigate back to parent - clear nested view completely
            setNavigationStack([]);
            setViewingNestedDashboard(null);
          }
        };
      })
    : undefined;

  return (
    <Suspense fallback={<LoadingFallback />}>
      <Dashboard
        dashboard={displayDashboard}
        allDashboards={dashboards}
        onSelect={handleDashboardSelect}
        onWidgetResize={handleWidgetResize}
        onWidgetDelete={handleWidgetDelete}
        widgetConversations={widgetConversations}
        setWidgetConversations={setWidgetConversations}
        dashboardVersion={dashboardVersion}
        onRefreshProjects={handleRefreshProjects}
        onDashboardsChange={handleRefreshProjects}
        breadcrumbs={breadcrumbs}
      />
    </Suspense>
  );
}

export default App;
