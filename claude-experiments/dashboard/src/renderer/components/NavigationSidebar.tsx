import { memo, useState, useCallback, useMemo, useEffect } from 'react';
import { useProjectDashboardTree, useDashboardCommands, useProjectCommands, useBackendState } from '../hooks/useBackendState';
import { usePersistentState } from '../hooks/useWidgetState';
import { useEmit } from '../hooks/useEvents';
import type { ProjectState, DashboardState } from '../../types/state';

/**
 * Navigation Sidebar - Miller columns layout for project and dashboard switching
 *
 * Uses a two-column layout:
 * - Left column: Projects list
 * - Right column: Dashboards for the selected project
 */

interface NavigationSidebarProps {
  showSearch?: boolean;
}

const NavigationSidebar = memo(function NavigationSidebar({ 
  showSearch = false 
}: NavigationSidebarProps) {
  const [query, setQuery] = useState('');
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [isPanelPinned, setIsPanelPinned] = usePersistentState('navSidebarPinned', true);
  const [currentSlot, setCurrentSlot] = useState<string>('left-panel');

  const { tree, activeProjectId, activeDashboardId, loading } = useProjectDashboardTree();
  const { switchDashboard } = useDashboardCommands();
  const { switchProject } = useProjectCommands();
  const emit = useEmit();

  // Detect which panel we're in using the backend state hook
  const [globalUIState] = useBackendState('globalUI');
  const globalUI = globalUIState as { widgets?: Array<{ id: string; type: string; slot: string }> } | undefined;

  useEffect(() => {
    if (globalUI?.widgets) {
      const widget = globalUI.widgets.find((w) => w.type === 'navigation-sidebar');
      if (widget?.slot) {
        setCurrentSlot(widget.slot);
      }
    }
  }, [globalUI]);

  // Listen for move events to switch sides
  useEffect(() => {
    const unsubscribe = window.eventAPI.subscribe('globalUI.panel.moveTo', async (event) => {
      const newSide = event.payload as string;
      const newSlot = `${newSide}-panel`;
      
      console.log('[NavigationSidebar] Move event received:', newSide, newSlot);
      
      // Find this widget's ID from globalUIState
      if (globalUI?.widgets) {
        const widget = globalUI.widgets.find((w) => w.type === 'navigation-sidebar');
        
        console.log('[NavigationSidebar] Found widget:', widget);
        
        if (widget) {
          // Move widget to the new panel
          const result = await window.stateAPI.command('globalUI.updateWidget', {
            id: widget.id,
            slot: newSlot
          });
          
          console.log('[NavigationSidebar] Update result:', result);
          
          setCurrentSlot(newSlot);
          
          // Emit toggle event to new side with current pin state
          emit(`globalUI.${newSide}Panel.toggle`, isPanelPinned);
        }
      }
    });
    return unsubscribe;
  }, [isPanelPinned, emit, globalUI]);

  const panelSide = currentSlot === 'left-panel' ? 'left' : 'right';

  // Emit initial pin state on mount
  useEffect(() => {
    emit(`globalUI.${panelSide}Panel.toggle`, isPanelPinned);
  }, [panelSide, emit, isPanelPinned]);

  const togglePin = useCallback(() => {
    const newValue = !isPanelPinned;
    setIsPanelPinned(newValue);
    emit(`globalUI.${panelSide}Panel.toggle`, newValue);
  }, [isPanelPinned, panelSide, emit]);

  // Filter tree based on search query
  const filteredTree = useMemo(() => {
    if (!query.trim()) return tree;

    const lowerQuery = query.toLowerCase();
    return tree
      .map(({ project, dashboards }) => {
        const projectMatches = project.name.toLowerCase().includes(lowerQuery);
        const matchingDashboards = dashboards.filter((d) =>
          d.name.toLowerCase().includes(lowerQuery)
        );

        if (projectMatches || matchingDashboards.length > 0) {
          return {
            project,
            dashboards: projectMatches ? dashboards : matchingDashboards,
          };
        }
        return null;
      })
      .filter((item): item is NonNullable<typeof item> => item !== null);
  }, [tree, query]);

  // Get the currently selected or active project
  const effectiveProjectId = selectedProjectId ?? activeProjectId;
  
  // Get dashboards for the selected/active project
  const currentProjectDashboards = useMemo(() => {
    if (!effectiveProjectId) return [];
    const projectEntry = filteredTree.find(({ project }) => project.id === effectiveProjectId);
    return projectEntry?.dashboards ?? [];
  }, [filteredTree, effectiveProjectId]);

  const handleProjectClick = useCallback(
    (project: ProjectState) => {
      setSelectedProjectId(project.id);
    },
    []
  );

  const handleProjectDoubleClick = useCallback(
    async (project: ProjectState) => {
      await switchProject(project.id);
    },
    [switchProject]
  );

  const handleDashboardClick = useCallback(
    async (dashboard: DashboardState) => {
      await switchDashboard(dashboard.id);
    },
    [switchDashboard]
  );

  if (loading) {
    return (
      <div
        style={{
          padding: 'var(--theme-spacing-lg)',
          color: 'var(--theme-text-muted)',
          fontSize: 'var(--theme-font-size-md)',
        }}
      >
        Loading...
      </div>
    );
  }

  const isEmpty = tree.length === 0;

  return (
    <div
      className="navigation-sidebar"
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '11px',
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
          color: 'var(--theme-text-muted)',
          padding: '12px 16px 8px 16px',
          borderBottom: '1px solid var(--theme-border-primary)',
        }}
      >
        <span>Navigation</span>
        <button
          onClick={togglePin}
          title={isPanelPinned ? 'Auto-hide sidebar' : 'Keep sidebar open'}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            padding: '4px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--theme-text-muted)',
            opacity: isPanelPinned ? 1 : 0.5,
            transition: 'opacity 0.2s',
          }}
        >
          <svg
            width={16}
            height={16}
            viewBox="0 0 16 16"
            fill="currentColor"
            style={{ display: 'block' }}
          >
            <rect x={1} y={1} width={14} height={14} rx={1} fill="none" stroke="currentColor" strokeWidth={1.5} />
            {panelSide === 'right' ? (
              <>
                <line x1={10} y1={1} x2={10} y2={15} stroke="currentColor" strokeWidth={1.5} />
                <line x1={11.5} y1={4} x2={13.5} y2={4} stroke="currentColor" strokeWidth={1.2} />
                <line x1={11.5} y1={7} x2={13.5} y2={7} stroke="currentColor" strokeWidth={1.2} />
                <line x1={11.5} y1={10} x2={13.5} y2={10} stroke="currentColor" strokeWidth={1.2} />
              </>
            ) : (
              <>
                <line x1={6} y1={1} x2={6} y2={15} stroke="currentColor" strokeWidth={1.5} />
                <line x1={2.5} y1={4} x2={4.5} y2={4} stroke="currentColor" strokeWidth={1.2} />
                <line x1={2.5} y1={7} x2={4.5} y2={7} stroke="currentColor" strokeWidth={1.2} />
                <line x1={2.5} y1={10} x2={4.5} y2={10} stroke="currentColor" strokeWidth={1.2} />
              </>
            )}
          </svg>
        </button>
      </div>

      {/* Optional Search */}
      {showSearch && (
        <div style={{ padding: '12px 16px' }}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search..."
            style={{
              width: '100%',
              padding: '8px 12px',
              background: 'var(--theme-bg-tertiary)',
              border: '1px solid var(--theme-border-primary)',
              borderRadius: '6px',
              color: 'var(--theme-text-primary)',
              fontSize: '13px',
              fontFamily: 'var(--theme-font-family)',
              outline: 'none',
            }}
          />
        </div>
      )}

      {/* Miller Columns Layout */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          overflow: 'hidden',
        }}
      >
        {/* Projects Column */}
        <div
          style={{
            flex: 1,
            background: 'var(--theme-bg-secondary)',
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            borderRight: '1px solid var(--theme-border-primary)',
          }}
        >
          {isEmpty ? (
            <div
              style={{
                padding: '24px 16px',
                color: 'var(--theme-text-muted)',
                fontSize: '13px',
                textAlign: 'center',
              }}
            >
              No projects yet
            </div>
          ) : filteredTree.length === 0 ? (
            <div
              style={{
                padding: '24px 16px',
                color: 'var(--theme-text-muted)',
                fontSize: '13px',
                textAlign: 'center',
              }}
            >
              No matches found
            </div>
          ) : (
            filteredTree.map(({ project, dashboards }) => {
              const isActive = project.id === activeProjectId;
              const isSelected = project.id === effectiveProjectId;

              return (
                <div
                  key={project.id}
                  onClick={() => handleProjectClick(project)}
                  onDoubleClick={() => handleProjectDoubleClick(project)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '10px 16px',
                    cursor: 'pointer',
                    background: isSelected
                      ? 'var(--theme-accent-primary)'
                      : 'transparent',
                    borderLeft: isActive && !isSelected ? '3px solid var(--theme-accent-primary)' : '3px solid transparent',
                    fontSize: '14px',
                    fontWeight: 500,
                    color: isSelected
                      ? 'white'
                      : isActive
                      ? 'var(--theme-accent-primary)'
                      : 'var(--theme-text-primary)',
                  }}
                  onMouseEnter={(e) => {
                    if (!isSelected) {
                      e.currentTarget.style.background = 'var(--theme-bg-tertiary)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!isSelected) {
                      e.currentTarget.style.background = 'transparent';
                    }
                  }}
                >
                  {/* Project Name */}
                  <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {project.name}
                  </span>

                  {/* Dashboard Count */}
                  <span
                    style={{
                      fontSize: '12px',
                      color: isSelected ? 'rgba(255, 255, 255, 0.8)' : 'var(--theme-text-muted)',
                      padding: '2px 8px',
                      borderRadius: '10px',
                      background: isSelected ? 'rgba(255, 255, 255, 0.2)' : 'var(--theme-bg-tertiary)',
                      fontWeight: 600,
                      minWidth: '22px',
                      textAlign: 'center',
                    }}
                  >
                    {dashboards.length}
                  </span>
                </div>
              );
            })
          )}
        </div>

        {/* Dashboards Column */}
        <div
          style={{
            flex: 1,
            background: 'var(--theme-bg-primary)',
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {!effectiveProjectId ? (
            <div
              style={{
                padding: '24px 16px',
                color: 'var(--theme-text-muted)',
                fontSize: '13px',
                textAlign: 'center',
              }}
            >
              Select a project
            </div>
          ) : currentProjectDashboards.length === 0 ? (
            <div
              style={{
                padding: '24px 16px',
                color: 'var(--theme-text-muted)',
                fontSize: '13px',
                textAlign: 'center',
              }}
            >
              No dashboards
            </div>
          ) : (
            currentProjectDashboards.map((dashboard) => {
              const isActive = dashboard.id === activeDashboardId;

              return (
                <div
                  key={dashboard.id}
                  onClick={() => handleDashboardClick(dashboard)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '10px 16px',
                    cursor: 'pointer',
                    background: isActive
                      ? 'var(--theme-accent-primary)'
                      : 'transparent',
                    fontSize: '14px',
                    fontWeight: 500,
                    color: isActive
                      ? 'white'
                      : 'var(--theme-text-primary)',
                  }}
                  onMouseEnter={(e) => {
                    if (!isActive) {
                      e.currentTarget.style.background = 'var(--theme-bg-tertiary)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive) {
                      e.currentTarget.style.background = 'transparent';
                    }
                  }}
                >
                  {/* Dashboard Name */}
                  <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {dashboard.name}
                  </span>

                  {/* Active Indicator */}
                  {isActive && (
                    <span
                      style={{
                        width: '6px',
                        height: '6px',
                        borderRadius: '50%',
                        background: 'white',
                      }}
                    />
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
});

export default NavigationSidebar;
