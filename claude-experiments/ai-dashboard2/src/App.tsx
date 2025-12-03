import { Suspense, useState, useEffect } from 'react';
import type { Dashboard as DashboardType, WidgetDimensions, NestedDashboardConfig } from './types';
import { Dashboard } from './Dashboard';
import { LoadingFallback } from './components';
import { findFirstAvailableSpace } from './utils/autoLayout';
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
        console.log('📡 Dashboard update received:', updatedDashboards);
        if (Array.isArray(updatedDashboards) && updatedDashboards.length > 0) {
          setDashboards(updatedDashboards);
          setDashboardVersion(prev => prev + 1);
        } else {
          console.error('⚠️ Received invalid dashboard update, ignoring:', updatedDashboards);
        }
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
    if (!window.dashboardAPI) return;

    try {
      console.log(`📐 Resize widget: dashboard=${dashboardId}, widget=${widgetId}`, dimensions);
      console.log(`   Viewing nested: ${!!viewingNestedDashboard}, Stack depth: ${navigationStack.length}`);

      // If we're viewing a nested dashboard, we need to update the nested widget within the parent
      if (viewingNestedDashboard && navigationStack.length > 0) {
        const parentItem = navigationStack[navigationStack.length - 1];
        const parentDashboard = Array.isArray(dashboards)
          ? dashboards.find(d => d.id === parentItem.dashboardId)
          : undefined;

        console.log(`   Parent dashboard ID: ${parentItem.dashboardId}, Found: ${!!parentDashboard}`);

        if (parentDashboard) {
          const nestedWidget = parentDashboard.widgets.find((w: any) => w.id === parentItem.widgetId);
          console.log(`   Nested widget ID: ${parentItem.widgetId}, Found: ${!!nestedWidget}`);

          if (nestedWidget && (nestedWidget as any).dashboard) {
            const widgetInNested = (nestedWidget as any).dashboard.widgets.find((w: any) => w.id === widgetId);
            console.log(`   Widget in nested: ${widgetId}, Found: ${!!widgetInNested}`);

            if (widgetInNested) {
              // Create a deep copy to avoid mutation issues
              const updatedNestedWidget = JSON.parse(JSON.stringify(nestedWidget));
              const targetWidget = updatedNestedWidget.dashboard.widgets.find((w: any) => w.id === widgetId);
              Object.assign(targetWidget, dimensions);

              console.log(`   ✅ Updating nested widget through parent`);
              console.log('   📦 Updated widget:', JSON.stringify(updatedNestedWidget, null, 2).substring(0, 500));

              try {
                await window.dashboardAPI.updateWidget(parentDashboard.id, parentItem.widgetId, updatedNestedWidget);
                console.log('   ✅ Update successful');
              } catch (err) {
                console.error('   ❌ Update failed:', err);
                throw err;
              }
              return;
            }
          }
        }

        console.error(`   ❌ Failed to find nested widget path, falling back to direct update`);
      }

      console.log(`   ✅ Direct update`);
      await window.dashboardAPI.updateWidgetDimensions(dashboardId, widgetId, dimensions);
    } catch (error) {
      console.error('❌ Failed to update widget dimensions:', error);
      // Don't throw - let the UI continue working
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

  /**
   * Check if a widget can be dropped into a nested dashboard (prevent circular references)
   */
  const canDropIntoNested = (widgetId: string, targetNestedId: string): boolean => {
    // Prevent dropping a widget into itself
    if (widgetId === targetNestedId) return false;

    // Get the widget being dragged
    const sourceDashboard = viewingNestedDashboard || selectedDashboard;
    if (!sourceDashboard) return false;

    const draggedWidget = sourceDashboard.widgets.find((w: any) => w.id === widgetId);
    if (!draggedWidget) return false;

    // If dragged widget is not a nested dashboard, it's always safe
    if (draggedWidget.type !== 'nested-dashboard') return true;

    // Check for circular reference: prevent nesting A into B if B is inside A
    const nestedConfig = draggedWidget as NestedDashboardConfig;
    const containsWidget = (dashboard: DashboardType, searchId: string): boolean => {
      return dashboard.widgets.some((w: any) => {
        if (w.id === searchId) return true;
        if (w.type === 'nested-dashboard' && w.dashboard) {
          return containsWidget(w.dashboard, searchId);
        }
        return false;
      });
    };

    return !containsWidget(nestedConfig.dashboard, targetNestedId);
  };

  /**
   * Get the current nesting depth for a widget
   */
  const getNestedDepth = (dashboard: DashboardType, widgetId: string, currentDepth = 0): number => {
    const widget = dashboard.widgets.find((w: any) => w.id === widgetId);
    if (!widget || widget.type !== 'nested-dashboard') return currentDepth;

    const nestedConfig = widget as NestedDashboardConfig;
    const nestedDashboards = nestedConfig.dashboard.widgets.filter((w: any) => w.type === 'nested-dashboard');

    if (nestedDashboards.length === 0) return currentDepth + 1;

    return Math.max(...nestedDashboards.map((w: any) =>
      getNestedDepth(nestedConfig.dashboard, w.id, currentDepth + 1)
    ));
  };

  /**
   * Transfer a widget into a nested dashboard or back to parent
   */
  const handleWidgetTransfer = async (
    widgetId: string,
    targetNestedWidgetId: string | null
  ) => {
    if (!window.dashboardAPI) return;

    try {
      const sourceDashboard = viewingNestedDashboard || selectedDashboard;
      if (!sourceDashboard) return;

      console.log(`🚀 Transfer requested: widget=${widgetId} → target=${targetNestedWidgetId}`);
      console.log(`📍 Current context: viewing nested=${!!viewingNestedDashboard}, source dashboard ID=${sourceDashboard.id}`);

      // Find the widget to transfer
      const widgetToTransfer = sourceDashboard.widgets.find((w: any) => w.id === widgetId);
      if (!widgetToTransfer) {
        console.error(`❌ Could not find widget ${widgetId} in source dashboard`);
        return;
      }

      // Handle transfer INTO a nested dashboard
      if (targetNestedWidgetId) {
        // Find the target nested dashboard widget
        const targetWidget = sourceDashboard.widgets.find((w: any) => w.id === targetNestedWidgetId);
        console.log(`🎯 Target widget:`, targetWidget);

        if (!targetWidget || targetWidget.type !== 'nested-dashboard') {
          console.error(`❌ Target is not a nested dashboard. Type: ${targetWidget?.type}`);
          return;
        }

        // Check for circular references
        if (!canDropIntoNested(widgetId, targetNestedWidgetId)) {
          alert('⚠️ Cannot nest dashboard: This would create a circular reference');
          return;
        }

        // Check max nesting depth (5 levels)
        const currentDepth = viewingNestedDashboard ? 1 : 0; // If viewing nested, we're already at depth 1
        const targetDepth = getNestedDepth(sourceDashboard, targetNestedWidgetId, currentDepth);

        if (targetDepth >= 5) {
          alert('⚠️ Cannot nest dashboard: Maximum nesting depth (5) would be exceeded');
          return;
        }

        const nestedConfig = targetWidget as NestedDashboardConfig;

        // Calculate position using auto-layout
        const gridSize = nestedConfig.dashboard.layout?.gridSize || 16;
        const position = findFirstAvailableSpace(
          nestedConfig.dashboard.widgets,
          widgetToTransfer,
          gridSize
        );

        // Add widget to nested dashboard
        const updatedNestedDashboard = {
          ...nestedConfig.dashboard,
          widgets: [
            ...nestedConfig.dashboard.widgets,
            { ...widgetToTransfer, x: position.x, y: position.y }
          ]
        };

        // Remove widget from source dashboard
        const updatedSourceWidgets = sourceDashboard.widgets.filter((w: any) => w.id !== widgetId);

        // Update the nested dashboard widget
        const updatedNestedWidget = {
          ...nestedConfig,
          dashboard: updatedNestedDashboard
        };

        // If we're in nested view, update through parent
        if (viewingNestedDashboard && navigationStack.length > 0) {
          const parentItem = navigationStack[navigationStack.length - 1];
          const parentDashboard = dashboards.find(d => d.id === parentItem.dashboardId);

          if (parentDashboard) {
            const parentNestedWidget = parentDashboard.widgets.find((w: any) => w.id === parentItem.widgetId);
            if (parentNestedWidget && (parentNestedWidget as any).dashboard) {
              // Update the parent's nested dashboard
              (parentNestedWidget as any).dashboard.widgets = updatedSourceWidgets;
              // Also update the target nested widget within the parent's nested dashboard
              const targetInParent = updatedSourceWidgets.find((w: any) => w.id === targetNestedWidgetId);
              if (targetInParent) {
                Object.assign(targetInParent, updatedNestedWidget);
              }
              await window.dashboardAPI.updateWidget(parentDashboard.id, parentItem.widgetId, parentNestedWidget);
              return;
            }
          }
        }

        // Otherwise, update the main dashboard
        // First update the target nested widget
        await window.dashboardAPI.updateWidget(sourceDashboard.id, targetNestedWidgetId, updatedNestedWidget);

        // Then delete the widget from the source dashboard
        await window.dashboardAPI.deleteWidget(sourceDashboard.id, widgetId);

        console.log(`✅ Widget "${widgetId}" moved into nested dashboard "${targetNestedWidgetId}"`);
      }
    } catch (error) {
      console.error('Failed to transfer widget:', error);
      alert('Failed to transfer widget. See console for details.');
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
        onWidgetTransfer={handleWidgetTransfer}
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
