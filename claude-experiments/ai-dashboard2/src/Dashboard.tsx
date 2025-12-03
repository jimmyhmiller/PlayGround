import { FC, useState, useEffect } from 'react';
import type { Dashboard as DashboardType, WidgetConfig } from './types';
import { DEFAULT_THEME } from './types/theme';
import { Grid } from './components';
import { Widget } from './components/ui/Widget';
import { WIDGET_REGISTRY } from './widgets';
import { ProjectNode, AddProjectDialog } from './components';

interface DashboardProps {
  dashboard: DashboardType;
  allDashboards?: DashboardType[];
  onSelect?: (dashboardId: string) => void;
  onWidgetResize?: (dashboardId: string, widgetId: string, dimensions: any) => void;
  onWidgetDelete?: (dashboardId: string, widgetId: string) => void;
  onWidgetTransfer?: (widgetId: string, targetNestedWidgetId: string | null) => void;
  widgetConversations: Record<string, string | null>;
  setWidgetConversations: (update: (prev: Record<string, string | null>) => Record<string, string | null>) => void;
  dashboardVersion?: number;
  onRefreshProjects?: () => void;
  onDashboardsChange?: () => void;
  breadcrumbs?: Array<{ title: string; onClick: () => void }>;
}

export const Dashboard: FC<DashboardProps> = ({
  dashboard,
  allDashboards,
  onSelect,
  onWidgetResize,
  onWidgetDelete,
  onWidgetTransfer,
  widgetConversations,
  setWidgetConversations,
  dashboardVersion,
  onRefreshProjects,
  onDashboardsChange,
  breadcrumbs
}) => {
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [projectToDelete, setProjectToDelete] = useState<DashboardType | null>(null);
  const [isRegeneratingAll, setIsRegeneratingAll] = useState(false);
  const [regenerateMessage, setRegenerateMessage] = useState<{ type: string; text: string } | null>(null);

  const theme = dashboard.theme && typeof dashboard.theme === 'object'
    ? { ...DEFAULT_THEME, ...dashboard.theme }
    : DEFAULT_THEME;

  const layout = dashboard.layout || dashboard.layoutSettings || {};
  const reloadTrigger = dashboardVersion;

  const cellSize = layout?.gridSize || 16;
  const gapX = layout?.widgetGap || 10;
  const gapY = layout?.widgetGap || 10;
  const layoutMode = layout?.mode || 'single-pane';

  console.log('[Dashboard] Layout mode:', layoutMode);
  console.log('[Dashboard] Layout settings:', layout);

  const handleAddProject = (project: any) => {
    if (onRefreshProjects) {
      onRefreshProjects();
    }
  };

  const handleDeleteProject = (dashboardToDelete: DashboardType) => {
    setProjectToDelete(dashboardToDelete);
    setShowDeleteConfirm(true);
  };

  const confirmDelete = async () => {
    if (projectToDelete) {
      try {
        // If we're deleting the active dashboard, switch to another one first
        if (projectToDelete.id === dashboard.id && allDashboards && allDashboards.length > 1) {
          const nextDashboard = allDashboards.find(d => d.id !== projectToDelete.id);
          if (nextDashboard && onSelect) {
            onSelect(nextDashboard.id);
          }
        }

        // Check if this dashboard is associated with a project
        if ((projectToDelete as any)._projectId && (window as any).projectAPI) {
          console.log('Removing project:', (projectToDelete as any)._projectId);
          await (window as any).projectAPI.removeProject((projectToDelete as any)._projectId);
        } else if ((window as any).dashboardAPI && (projectToDelete as any)._sourcePath) {
          console.log('Removing standalone dashboard file:', (projectToDelete as any)._sourcePath);
          await (window as any).dashboardAPI.removeConfigPath((projectToDelete as any)._sourcePath);
        }

        if (onRefreshProjects) {
          onRefreshProjects();
        }

        if (onDashboardsChange) {
          onDashboardsChange();
        }
      } catch (error) {
        console.error('Failed to remove:', error);
      }
    }
    setShowDeleteConfirm(false);
    setProjectToDelete(null);
  };

  const handleRegenerateAll = async () => {
    setIsRegeneratingAll(true);
    setRegenerateMessage(null);

    try {
      const result = await (window as any).dashboardAPI.regenerateAllWidgets(dashboard.id);
      if (result.success) {
        setRegenerateMessage({ type: 'success', text: result.message });
      } else {
        setRegenerateMessage({ type: 'error', text: result.error });
      }
      setTimeout(() => setRegenerateMessage(null), 5000);
    } catch (error: any) {
      setRegenerateMessage({ type: 'error', text: error.message });
      setTimeout(() => setRegenerateMessage(null), 5000);
    } finally {
      setIsRegeneratingAll(false);
    }
  };

  const hasRegeneratableWidgets = dashboard.widgets?.some(w =>
    (w as any).regenerateCommand || (w as any).regenerateScript
  );

  const getWindowFrameOverflow = () => {
    switch (layoutMode) {
      case 'vertical-scroll':
      case 'horizontal-scroll':
        return 'visible';
      case 'infinite-canvas':
      case 'single-pane':
      default:
        return 'hidden';
    }
  };

  const getWindowFrameHeight = () => {
    switch (layoutMode) {
      case 'vertical-scroll':
      case 'horizontal-scroll':
        return 'auto';
      case 'infinite-canvas':
      case 'single-pane':
      default:
        return '100vh';
    }
  };

  return (
    <div className="window-frame" style={{ '--accent': theme.accent, overflow: getWindowFrameOverflow(), height: getWindowFrameHeight(), minHeight: layoutMode === 'vertical-scroll' || layoutMode === 'horizontal-scroll' ? '100vh' : undefined } as any}>
      <div className="titlebar" />
      <div className="bg-layer" style={theme.bgLayer as any} />
      <div className="sidebar">
        {allDashboards?.map((d) => {
          const dTheme = d.theme && typeof d.theme === 'object'
            ? { ...DEFAULT_THEME, ...d.theme }
            : DEFAULT_THEME;
          return (
            <ProjectNode
              key={d.id}
              icon={d.icon}
              active={d.id === dashboard.id}
              accent={dTheme.accent}
              hoverAccent={dTheme.accent}
              onClick={() => onSelect?.(d.id)}
              onDelete={() => handleDeleteProject(d)}
            />
          );
        })}
        <div
          className="project-node add-button"
          style={{ '--accent': theme.accent, '--hover-accent': theme.accent } as any}
          onClick={() => setShowAddDialog(true)}
          title="Add Project"
        >
          <svg viewBox="0 0 60 60">
            <line x1="30" y1="15" x2="30" y2="45" strokeWidth="4" />
            <line x1="15" y1="30" x2="45" y2="30" strokeWidth="4" />
          </svg>
        </div>
      </div>
      <div className="dashboard" style={{ backgroundColor: theme.bgApp }}>
        <div className="header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            {breadcrumbs && breadcrumbs.length > 0 ? (
              <h1 style={{ fontFamily: theme.textHead, color: theme.textColor, display: 'flex', alignItems: 'center', gap: '12px' }}>
                {breadcrumbs.map((crumb, index) => (
                  <span key={index} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <span
                      onClick={crumb.onClick}
                      style={{
                        cursor: 'pointer',
                        opacity: 0.6,
                        transition: 'opacity 0.2s ease',
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.opacity = '1'}
                      onMouseLeave={(e) => e.currentTarget.style.opacity = '0.6'}
                    >
                      {crumb.title}
                    </span>
                    <span style={{ opacity: 0.4, fontWeight: 300 }}>›</span>
                  </span>
                ))}
                <span>{dashboard.title}</span>
              </h1>
            ) : (
              <h1 style={{ fontFamily: theme.textHead, color: theme.textColor }}>{dashboard.title}</h1>
            )}
            <p style={{ fontFamily: theme.textBody, color: theme.accent }}>{dashboard.subtitle}</p>
          </div>
          {hasRegeneratableWidgets && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              {regenerateMessage && (
                <span style={{
                  fontSize: '0.75rem',
                  color: regenerateMessage.type === 'success' ? theme.positive : theme.negative,
                  fontFamily: theme.textBody
                }}>
                  {regenerateMessage.text}
                </span>
              )}
              <button
                onClick={handleRegenerateAll}
                disabled={isRegeneratingAll}
                title="Regenerate all widgets with commands"
                style={{
                  background: theme.accent,
                  border: 'none',
                  borderRadius: 6,
                  padding: '8px 16px',
                  color: '#000',
                  cursor: isRegeneratingAll ? 'wait' : 'pointer',
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  opacity: isRegeneratingAll ? 0.6 : 1,
                  fontFamily: theme.textBody,
                  transition: 'opacity 0.2s'
                }}
              >
                {isRegeneratingAll ? '⟳ Refreshing...' : '⟳ Refresh All Data'}
              </button>
            </div>
          )}
        </div>
        <Grid cellSize={cellSize} gapX={gapX} gapY={gapY} mode={layoutMode} width="100%" height="calc(100% - 80px)">
          {dashboard.widgets?.map((widget: WidgetConfig) => (
            <Widget
              key={`${dashboard.id}-${widget.id}`}
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
              onTransfer={onWidgetTransfer}
              widgetComponents={WIDGET_REGISTRY}
            />
          ))}
        </Grid>
      </div>
      {showAddDialog && (
        <AddProjectDialog
          theme={theme}
          onClose={() => setShowAddDialog(false)}
          onAdd={handleAddProject}
        />
      )}
      {showDeleteConfirm && projectToDelete && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: theme.widgetBg || '#1a1a1a',
            border: theme.widgetBorder || '1px solid #333',
            borderRadius: theme.widgetRadius || '6px',
            padding: 24,
            width: 400,
            maxWidth: '90%',
            fontFamily: theme.textBody || 'system-ui'
          }}>
            <h2 style={{
              margin: '0 0 16px 0',
              color: theme.textColor || '#fff',
              fontSize: '1.2rem'
            }}>
              Remove Project
            </h2>
            <p style={{
              margin: '0 0 20px 0',
              color: theme.textColor || '#fff',
              opacity: 0.8
            }}>
              Are you sure you want to remove "{projectToDelete.title}"?
            </p>
            <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end' }}>
              <button
                onClick={() => {
                  setShowDeleteConfirm(false);
                  setProjectToDelete(null);
                }}
                style={{
                  padding: '8px 16px',
                  backgroundColor: 'transparent',
                  border: '1px solid #555',
                  borderRadius: 4,
                  color: theme.textColor || '#fff',
                  cursor: 'pointer',
                  fontFamily: theme.textBody || 'system-ui'
                }}
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                style={{
                  padding: '8px 16px',
                  backgroundColor: theme.negative || '#ff4757',
                  border: 'none',
                  borderRadius: 4,
                  color: '#fff',
                  cursor: 'pointer',
                  fontFamily: theme.textBody || 'system-ui',
                  fontWeight: 600
                }}
              >
                Remove
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
