/**
 * StateStore
 *
 * Central state management for the application.
 * - Maintains state tree in memory
 * - Handles commands to update state
 * - Emits events on state changes
 * - Persists state to disk (debounced)
 */

import { loadState, saveState, DEFAULT_STATE } from './persistence';
import type {
  AppState,
  WindowsState,
  WindowState,
  ThemeState,
  SettingsState,
  ComponentsState,
  ComponentInstance,
  CommandResult,
  WindowCreatePayload,
  WindowUpdatePayload,
  ProjectsState,
  ProjectState,
  DashboardsState,
  DashboardState,
  ProjectCreatePayload,
  ProjectRenamePayload,
  ProjectSetThemePayload,
  DashboardCreatePayload,
  DashboardRenamePayload,
  DashboardSetThemeOverridePayload,
  GlobalUIState,
  SlotState,
  WidgetState,
  SlotAddPayload,
  WidgetAddPayload,
  WidgetUpdatePayload,
  WidgetStateSetPayload,
  WidgetStateGetPayload,
  WidgetStateClearPayload,
  CustomWidgetsState,
  CustomWidgetDefinition,
  CustomWidgetRegisterPayload,
  CustomWidgetUnregisterPayload,
  CustomWidgetUpdatePayload,
} from '../../types/state';

// Type for the events module
interface EventEmitter {
  emit(type: string, payload: unknown): void;
}

export class StateStore {
  private events: EventEmitter;
  private state: AppState;
  private saveTimeout: ReturnType<typeof setTimeout> | null = null;
  private saveDebounceMs = 100;

  constructor(events: EventEmitter) {
    this.events = events;
    this.state = loadState();
  }

  /**
   * Get state at a path
   */
  getState(path?: string): unknown {
    if (!path) {
      return this.state;
    }

    const parts = path.split('.');
    let current: unknown = this.state;

    for (const part of parts) {
      if (current === undefined || current === null) {
        return undefined;
      }
      current = (current as Record<string, unknown>)[part];
    }

    return current;
  }

  /**
   * Set state at a path
   */
  setState(path: string, value: unknown): void {
    if (!path) {
      this.state = value as AppState;
    } else {
      const parts = path.split('.');
      const lastPart = parts.pop()!;
      let current: Record<string, unknown> = this.state as unknown as Record<string, unknown>;

      for (const part of parts) {
        if (current[part] === undefined) {
          current[part] = {};
        }
        current = current[part] as Record<string, unknown>;
      }

      current[lastPart] = value;
    }

    // Emit state change event
    this.emitChange(path);

    // Schedule debounced save
    this.scheduleSave();
  }

  /**
   * Emit a state change event
   */
  private emitChange(path: string): void {
    const eventType = path ? `state.changed.${path}` : 'state.changed';
    this.events.emit(eventType, { path, value: this.getState(path) });
  }

  /**
   * Schedule a debounced save
   */
  private scheduleSave(): void {
    if (this.saveTimeout) {
      clearTimeout(this.saveTimeout);
    }
    this.saveTimeout = setTimeout(() => {
      this.save();
    }, this.saveDebounceMs);
  }

  /**
   * Save state to disk immediately
   */
  save(): void {
    if (this.saveTimeout) {
      clearTimeout(this.saveTimeout);
      this.saveTimeout = null;
    }
    try {
      const widgetStateCount = Object.keys((this.state.widgetState?.data as Record<string, unknown>) || {}).length;
      console.log(`[StateStore] Saving state to disk (${widgetStateCount} widget states)`);
      saveState(this.state);
    } catch (err) {
      console.error('StateStore: Failed to save:', err);
    }
  }

  /**
   * Handle a command to update state
   */
  handleCommand(type: string, payload: unknown): CommandResult {
    const [domain, action] = type.split('.');

    switch (domain) {
      case 'windows':
        return this.handleWindowsCommand(action!, payload);
      case 'theme':
        return this.handleThemeCommand(action!, payload);
      case 'settings':
        return this.handleSettingsCommand(action!, payload);
      case 'components':
        return this.handleComponentsCommand(action!, payload);
      case 'projects':
        return this.handleProjectsCommand(action!, payload);
      case 'dashboards':
        return this.handleDashboardsCommand(action!, payload);
      case 'globalUI':
        return this.handleGlobalUICommand(action!, payload);
      case 'widgetState':
        return this.handleWidgetStateCommand(action!, payload);
      case 'customWidgets':
        return this.handleCustomWidgetsCommand(action!, payload);
      default:
        throw new Error(`Unknown command domain: ${domain}`);
    }
  }

  // ========== Windows Commands ==========

  private handleWindowsCommand(action: string, payload: unknown): CommandResult {
    const windows = this.getState('windows') as WindowsState;

    switch (action) {
      case 'create': {
        const p = payload as WindowCreatePayload;
        const id = `win_${windows.nextId}`;
        const newWindow: WindowState = {
          id,
          title: p.title ?? 'Window',
          component: p.component,
          props: p.props ?? {},
          x: p.x ?? 50 + ((windows.nextId - 1) % 10) * 30,
          y: p.y ?? 50 + ((windows.nextId - 1) % 10) * 30,
          width: p.width ?? 500,
          height: p.height ?? 350,
          zIndex: windows.list.length + 1,
        };

        this.setState('windows', {
          ...windows,
          list: [...windows.list, newWindow],
          focusedId: id,
          nextId: windows.nextId + 1,
        });

        return { id };
      }

      case 'close': {
        const { id } = payload as { id: string };
        const newList = windows.list.filter((w) => w.id !== id);
        this.setState('windows', {
          ...windows,
          list: newList,
          focusedId: windows.focusedId === id ? null : windows.focusedId,
        });
        return { success: true };
      }

      case 'focus': {
        const { id } = payload as { id: string };
        // Skip if already focused
        if (windows.focusedId === id) {
          return { success: true, noChange: true };
        }
        const maxZ = Math.max(0, ...windows.list.map((w) => w.zIndex));
        const newList = windows.list.map((w) =>
          w.id === id ? { ...w, zIndex: maxZ + 1 } : w
        );
        this.setState('windows', {
          ...windows,
          list: newList,
          focusedId: id,
        });
        return { success: true };
      }

      case 'update': {
        const { id, ...updates } = payload as WindowUpdatePayload;
        const window = windows.list.find((w) => w.id === id);
        if (!window) {
          return { success: false, error: 'Window not found' };
        }
        // Check if anything actually changed
        const hasChanges = Object.keys(updates).some((key) => {
          if (key === 'props') {
            // Deep compare props
            return JSON.stringify(window.props) !== JSON.stringify(updates.props);
          }
          return (window as unknown as Record<string, unknown>)[key] !== (updates as Record<string, unknown>)[key];
        });
        if (!hasChanges) {
          return { success: true, noChange: true };
        }
        const newList = windows.list.map((w) =>
          w.id === id ? { ...w, ...updates } : w
        );
        this.setState('windows', {
          ...windows,
          list: newList,
        });
        return { success: true };
      }

      default:
        throw new Error(`Unknown windows action: ${action}`);
    }
  }

  // ========== Theme Commands ==========

  private handleThemeCommand(action: string, payload: unknown): CommandResult {
    const theme = this.getState('theme') as ThemeState;

    switch (action) {
      case 'set': {
        const { theme: newTheme } = payload as { theme: string };
        this.setState('theme', {
          ...theme,
          current: newTheme,
        });
        return { success: true };
      }

      case 'setVariable': {
        const { variable, value } = payload as { variable: string; value: string };
        this.setState('theme', {
          ...theme,
          overrides: {
            ...theme.overrides,
            [variable]: value,
          },
        });
        return { success: true };
      }

      case 'resetVariable': {
        const { variable } = payload as { variable: string };
        const newOverrides = { ...theme.overrides };
        delete newOverrides[variable];
        this.setState('theme', {
          ...theme,
          overrides: newOverrides,
        });
        return { success: true };
      }

      case 'resetOverrides': {
        this.setState('theme', {
          ...theme,
          overrides: {},
        });
        return { success: true };
      }

      default:
        throw new Error(`Unknown theme action: ${action}`);
    }
  }

  // ========== Settings Commands ==========

  private handleSettingsCommand(action: string, payload: unknown): CommandResult {
    const settings = this.getState('settings') as SettingsState;

    switch (action) {
      case 'update': {
        const { key, value } = payload as { key: string; value: unknown };
        this.setState('settings', {
          ...settings,
          [key]: value,
        });
        return { success: true };
      }

      case 'reset': {
        this.setState('settings', DEFAULT_STATE.settings);
        return { success: true };
      }

      default:
        throw new Error(`Unknown settings action: ${action}`);
    }
  }

  // ========== Components Commands ==========

  private handleComponentsCommand(action: string, payload: unknown): CommandResult {
    const components = this.getState('components') as ComponentsState;

    switch (action) {
      case 'add': {
        const p = payload as { id?: string; type: string; props?: Record<string, unknown> };
        const instance: ComponentInstance = {
          id: p.id ?? `inst_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
          type: p.type,
          props: p.props ?? {},
          createdAt: Date.now(),
        };
        this.setState('components', {
          ...components,
          instances: [...components.instances, instance],
        });
        return { id: instance.id };
      }

      case 'remove': {
        const { id } = payload as { id: string };
        this.setState('components', {
          ...components,
          instances: components.instances.filter((i) => i.id !== id),
        });
        return { success: true };
      }

      case 'updateProps': {
        const { id, props } = payload as { id: string; props: Record<string, unknown> };
        this.setState('components', {
          ...components,
          instances: components.instances.map((i) =>
            i.id === id ? { ...i, props: { ...i.props, ...props } } : i
          ),
        });
        return { success: true };
      }

      default:
        throw new Error(`Unknown components action: ${action}`);
    }
  }

  // ========== Projects Commands ==========

  private handleProjectsCommand(action: string, payload: unknown): CommandResult {
    const projects = this.getState('projects') as ProjectsState;
    const dashboards = this.getState('dashboards') as DashboardsState;

    switch (action) {
      case 'create': {
        const p = payload as ProjectCreatePayload;
        const now = Date.now();
        const projectId = `proj_${projects.nextProjectId}`;
        const dashboardId = `dash_${projects.nextDashboardId}`;

        // Create the default dashboard for this project
        const defaultDashboard: DashboardState = {
          id: dashboardId,
          name: 'Default',
          projectId,
          windows: [],
          widgetState: {},
          createdAt: now,
          updatedAt: now,
        };

        // Create the project
        const newProject: ProjectState = {
          id: projectId,
          name: p.name,
          rootDir: p.rootDir,
          defaultTheme: { current: 'dark', overrides: {} },
          dashboardIds: [dashboardId],
          activeDashboardId: dashboardId,
          createdAt: now,
          updatedAt: now,
        };

        // Update dashboards
        this.setState('dashboards', {
          ...dashboards,
          list: [...dashboards.list, defaultDashboard],
        });

        // Update projects
        this.setState('projects', {
          ...projects,
          list: [...projects.list, newProject],
          activeProjectId: projectId,
          nextProjectId: projects.nextProjectId + 1,
          nextDashboardId: projects.nextDashboardId + 1,
        });

        // Clear current windows for the new project
        this.setState('windows', {
          list: [],
          focusedId: null,
          nextId: 1,
        });

        // Apply project theme
        this.setState('theme', newProject.defaultTheme);

        return { id: projectId };
      }

      case 'delete': {
        const { id } = payload as { id: string };
        const project = projects.list.find((p) => p.id === id);
        if (!project) {
          return { success: false, error: 'Project not found' };
        }

        // Remove all dashboards belonging to this project
        const newDashboardsList = dashboards.list.filter((d) => d.projectId !== id);
        this.setState('dashboards', {
          ...dashboards,
          list: newDashboardsList,
        });

        // Remove the project
        const newProjectsList = projects.list.filter((p) => p.id !== id);
        const newActiveProjectId =
          projects.activeProjectId === id
            ? newProjectsList.length > 0
              ? newProjectsList[0]!.id
              : null
            : projects.activeProjectId;

        this.setState('projects', {
          ...projects,
          list: newProjectsList,
          activeProjectId: newActiveProjectId,
        });

        // If we switched to a different project, load its active dashboard
        if (newActiveProjectId && newActiveProjectId !== id) {
          const newActiveProject = newProjectsList.find((p) => p.id === newActiveProjectId);
          if (newActiveProject?.activeDashboardId) {
            const activeDashboard = newDashboardsList.find(
              (d) => d.id === newActiveProject.activeDashboardId
            );
            if (activeDashboard) {
              this.setState('windows', {
                list: activeDashboard.windows,
                focusedId: null,
                nextId: Math.max(1, ...activeDashboard.windows.map((w) => parseInt(w.id.split('_')[1] || '0', 10) + 1)),
              });
              this.setState('theme', activeDashboard.themeOverride ?? newActiveProject.defaultTheme);
            }
          }
        } else if (!newActiveProjectId) {
          // No projects left, clear windows
          this.setState('windows', {
            list: [],
            focusedId: null,
            nextId: 1,
          });
          this.setState('theme', { current: 'dark', overrides: {} });
        }

        return { success: true };
      }

      case 'rename': {
        const p = payload as ProjectRenamePayload;
        const projectIndex = projects.list.findIndex((proj) => proj.id === p.id);
        if (projectIndex === -1) {
          return { success: false, error: 'Project not found' };
        }

        const updatedProjects = [...projects.list];
        updatedProjects[projectIndex] = {
          ...updatedProjects[projectIndex]!,
          name: p.name,
          updatedAt: Date.now(),
        };

        this.setState('projects', {
          ...projects,
          list: updatedProjects,
        });

        return { success: true };
      }

      case 'setTheme': {
        const p = payload as ProjectSetThemePayload;
        const projectIndex = projects.list.findIndex((proj) => proj.id === p.id);
        if (projectIndex === -1) {
          return { success: false, error: 'Project not found' };
        }

        const updatedProjects = [...projects.list];
        updatedProjects[projectIndex] = {
          ...updatedProjects[projectIndex]!,
          defaultTheme: p.theme,
          updatedAt: Date.now(),
        };

        this.setState('projects', {
          ...projects,
          list: updatedProjects,
        });

        // If this is the active project and the active dashboard has no override, apply theme
        if (projects.activeProjectId === p.id) {
          const project = updatedProjects[projectIndex]!;
          const activeDashboard = dashboards.list.find((d) => d.id === project.activeDashboardId);
          if (!activeDashboard?.themeOverride) {
            this.setState('theme', p.theme);
          }
        }

        return { success: true };
      }

      case 'switch': {
        const { id } = payload as { id: string };
        if (projects.activeProjectId === id) {
          return { success: true, noChange: true };
        }

        const project = projects.list.find((p) => p.id === id);
        if (!project) {
          return { success: false, error: 'Project not found' };
        }

        // Save current windows to current active dashboard
        const currentProject = projects.list.find((p) => p.id === projects.activeProjectId);
        if (currentProject?.activeDashboardId) {
          const windows = this.getState('windows') as WindowsState;
          const dashboardIndex = dashboards.list.findIndex(
            (d) => d.id === currentProject.activeDashboardId
          );
          if (dashboardIndex !== -1) {
            const updatedDashboards = [...dashboards.list];
            updatedDashboards[dashboardIndex] = {
              ...updatedDashboards[dashboardIndex]!,
              windows: windows.list,
              updatedAt: Date.now(),
            };
            this.setState('dashboards', {
              ...dashboards,
              list: updatedDashboards,
            });
          }
        }

        // Switch to new project
        this.setState('projects', {
          ...projects,
          activeProjectId: id,
        });

        // Load the new project's active dashboard
        if (project.activeDashboardId) {
          const updatedDashboards = this.getState('dashboards') as DashboardsState;
          const activeDashboard = updatedDashboards.list.find(
            (d) => d.id === project.activeDashboardId
          );
          if (activeDashboard) {
            this.setState('windows', {
              list: activeDashboard.windows,
              focusedId: null,
              nextId: Math.max(1, ...activeDashboard.windows.map((w) => parseInt(w.id.split('_')[1] || '0', 10) + 1)),
            });
            this.setState('theme', activeDashboard.themeOverride ?? project.defaultTheme);
          }
        } else {
          this.setState('windows', {
            list: [],
            focusedId: null,
            nextId: 1,
          });
          this.setState('theme', project.defaultTheme);
        }

        return { success: true };
      }

      default:
        throw new Error(`Unknown projects action: ${action}`);
    }
  }

  // ========== Dashboards Commands ==========

  private handleDashboardsCommand(action: string, payload: unknown): CommandResult {
    const projects = this.getState('projects') as ProjectsState;
    const dashboards = this.getState('dashboards') as DashboardsState;

    switch (action) {
      case 'create': {
        const p = payload as DashboardCreatePayload;
        const project = projects.list.find((proj) => proj.id === p.projectId);
        if (!project) {
          return { success: false, error: 'Project not found' };
        }

        const now = Date.now();
        const dashboardId = `dash_${projects.nextDashboardId}`;

        const newDashboard: DashboardState = {
          id: dashboardId,
          name: p.name,
          projectId: p.projectId,
          windows: [],
          widgetState: {},
          createdAt: now,
          updatedAt: now,
        };

        // Add dashboard
        this.setState('dashboards', {
          ...dashboards,
          list: [...dashboards.list, newDashboard],
        });

        // Update project's dashboardIds
        const projectIndex = projects.list.findIndex((proj) => proj.id === p.projectId);
        const updatedProjects = [...projects.list];
        updatedProjects[projectIndex] = {
          ...updatedProjects[projectIndex]!,
          dashboardIds: [...project.dashboardIds, dashboardId],
          updatedAt: now,
        };

        this.setState('projects', {
          ...projects,
          list: updatedProjects,
          nextDashboardId: projects.nextDashboardId + 1,
        });

        return { id: dashboardId };
      }

      case 'delete': {
        const { id } = payload as { id: string };
        const dashboard = dashboards.list.find((d) => d.id === id);
        if (!dashboard) {
          return { success: false, error: 'Dashboard not found' };
        }

        const project = projects.list.find((p) => p.id === dashboard.projectId);
        if (!project) {
          return { success: false, error: 'Parent project not found' };
        }

        // Don't allow deleting the last dashboard
        if (project.dashboardIds.length <= 1) {
          return { success: false, error: 'Cannot delete the last dashboard in a project' };
        }

        // Remove dashboard
        this.setState('dashboards', {
          ...dashboards,
          list: dashboards.list.filter((d) => d.id !== id),
        });

        // Update project's dashboardIds
        const projectIndex = projects.list.findIndex((p) => p.id === dashboard.projectId);
        const newDashboardIds = project.dashboardIds.filter((dId) => dId !== id);
        const newActiveDashboardId =
          project.activeDashboardId === id ? newDashboardIds[0]! : project.activeDashboardId;

        const updatedProjects = [...projects.list];
        updatedProjects[projectIndex] = {
          ...updatedProjects[projectIndex]!,
          dashboardIds: newDashboardIds,
          activeDashboardId: newActiveDashboardId,
          updatedAt: Date.now(),
        };

        this.setState('projects', {
          ...projects,
          list: updatedProjects,
        });

        // If we deleted the active dashboard and this is the active project, load the new active dashboard
        if (project.activeDashboardId === id && projects.activeProjectId === dashboard.projectId) {
          const updatedDashboards = this.getState('dashboards') as DashboardsState;
          const newActiveDashboard = updatedDashboards.list.find((d) => d.id === newActiveDashboardId);
          if (newActiveDashboard) {
            this.setState('windows', {
              list: newActiveDashboard.windows,
              focusedId: null,
              nextId: Math.max(1, ...newActiveDashboard.windows.map((w) => parseInt(w.id.split('_')[1] || '0', 10) + 1)),
            });
            const updatedProject = updatedProjects[projectIndex]!;
            this.setState('theme', newActiveDashboard.themeOverride ?? updatedProject.defaultTheme);
          }
        }

        return { success: true };
      }

      case 'rename': {
        const p = payload as DashboardRenamePayload;
        const dashboardIndex = dashboards.list.findIndex((d) => d.id === p.id);
        if (dashboardIndex === -1) {
          return { success: false, error: 'Dashboard not found' };
        }

        const updatedDashboards = [...dashboards.list];
        updatedDashboards[dashboardIndex] = {
          ...updatedDashboards[dashboardIndex]!,
          name: p.name,
          updatedAt: Date.now(),
        };

        this.setState('dashboards', {
          ...dashboards,
          list: updatedDashboards,
        });

        return { success: true };
      }

      case 'switch': {
        const { id } = payload as { id: string };
        const dashboard = dashboards.list.find((d) => d.id === id);
        if (!dashboard) {
          return { success: false, error: 'Dashboard not found' };
        }

        const project = projects.list.find((p) => p.id === dashboard.projectId);
        if (!project) {
          return { success: false, error: 'Parent project not found' };
        }

        // Check if already on this dashboard
        if (project.activeDashboardId === id && projects.activeProjectId === dashboard.projectId) {
          return { success: true, noChange: true };
        }

        // Save current windows to current active dashboard if in same project
        const windows = this.getState('windows') as WindowsState;
        const currentProject = projects.list.find((p) => p.id === projects.activeProjectId);
        if (currentProject?.activeDashboardId) {
          const currentDashboardIndex = dashboards.list.findIndex(
            (d) => d.id === currentProject.activeDashboardId
          );
          if (currentDashboardIndex !== -1) {
            const updatedDashboards = [...dashboards.list];
            updatedDashboards[currentDashboardIndex] = {
              ...updatedDashboards[currentDashboardIndex]!,
              windows: windows.list,
              updatedAt: Date.now(),
            };
            this.setState('dashboards', {
              ...dashboards,
              list: updatedDashboards,
            });
          }
        }

        // Update project's activeDashboardId and possibly switch active project
        const projectIndex = projects.list.findIndex((p) => p.id === dashboard.projectId);
        const updatedProjects = [...projects.list];
        updatedProjects[projectIndex] = {
          ...updatedProjects[projectIndex]!,
          activeDashboardId: id,
          updatedAt: Date.now(),
        };

        this.setState('projects', {
          ...projects,
          list: updatedProjects,
          activeProjectId: dashboard.projectId,
        });

        // Load the dashboard's windows
        const updatedDashboards = this.getState('dashboards') as DashboardsState;
        const targetDashboard = updatedDashboards.list.find((d) => d.id === id);
        if (targetDashboard) {
          this.setState('windows', {
            list: targetDashboard.windows,
            focusedId: null,
            nextId: Math.max(1, ...targetDashboard.windows.map((w) => parseInt(w.id.split('_')[1] || '0', 10) + 1)),
          });
          const targetProject = updatedProjects[projectIndex]!;
          this.setState('theme', targetDashboard.themeOverride ?? targetProject.defaultTheme);
        }

        return { success: true };
      }

      case 'setThemeOverride': {
        const p = payload as DashboardSetThemeOverridePayload;
        const dashboardIndex = dashboards.list.findIndex((d) => d.id === p.id);
        if (dashboardIndex === -1) {
          return { success: false, error: 'Dashboard not found' };
        }

        const dashboard = dashboards.list[dashboardIndex]!;
        const updatedDashboards = [...dashboards.list];
        updatedDashboards[dashboardIndex] = {
          ...dashboard,
          themeOverride: p.themeOverride,
          updatedAt: Date.now(),
        };

        this.setState('dashboards', {
          ...dashboards,
          list: updatedDashboards,
        });

        // If this is the active dashboard, apply the theme
        const project = projects.list.find((proj) => proj.id === dashboard.projectId);
        if (
          project &&
          projects.activeProjectId === dashboard.projectId &&
          project.activeDashboardId === p.id
        ) {
          this.setState('theme', p.themeOverride ?? project.defaultTheme);
        }

        return { success: true };
      }

      case 'saveLayout': {
        // Save current windows to the active dashboard
        const currentProject = projects.list.find((p) => p.id === projects.activeProjectId);
        if (!currentProject?.activeDashboardId) {
          return { success: false, error: 'No active dashboard' };
        }

        const windows = this.getState('windows') as WindowsState;
        const dashboardIndex = dashboards.list.findIndex(
          (d) => d.id === currentProject.activeDashboardId
        );
        if (dashboardIndex === -1) {
          return { success: false, error: 'Active dashboard not found' };
        }

        const updatedDashboards = [...dashboards.list];
        updatedDashboards[dashboardIndex] = {
          ...updatedDashboards[dashboardIndex]!,
          windows: windows.list,
          updatedAt: Date.now(),
        };

        this.setState('dashboards', {
          ...dashboards,
          list: updatedDashboards,
        });

        return { success: true };
      }

      default:
        throw new Error(`Unknown dashboards action: ${action}`);
    }
  }

  // ========== Global UI Commands ==========

  private handleGlobalUICommand(action: string, payload: unknown): CommandResult {
    const globalUI = this.getState('globalUI') as GlobalUIState;

    switch (action) {
      case 'addSlot': {
        const p = payload as SlotAddPayload;
        // Check if slot already exists
        if (globalUI.slots.some((s) => s.id === p.id)) {
          return { success: false, error: `Slot "${p.id}" already exists` };
        }

        const newSlot: SlotState = {
          id: p.id,
          position: p.position,
          zIndex: p.zIndex,
        };

        this.setState('globalUI', {
          ...globalUI,
          slots: [...globalUI.slots, newSlot],
        });

        return { id: p.id };
      }

      case 'removeSlot': {
        const { id } = payload as { id: string };
        // Remove slot and any widgets in it
        this.setState('globalUI', {
          ...globalUI,
          slots: globalUI.slots.filter((s) => s.id !== id),
          widgets: globalUI.widgets.filter((w) => w.slot !== id),
        });
        return { success: true };
      }

      case 'addWidget': {
        const p = payload as WidgetAddPayload;
        // Generate ID if not provided
        const id = p.id ?? `widget_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;

        // Check if slot exists
        if (!globalUI.slots.some((s) => s.id === p.slot)) {
          return { success: false, error: `Slot "${p.slot}" does not exist` };
        }

        const newWidget: WidgetState = {
          id,
          type: p.type,
          slot: p.slot,
          props: p.props ?? {},
          priority: p.priority ?? 10,
          visible: true,
        };

        this.setState('globalUI', {
          ...globalUI,
          widgets: [...globalUI.widgets, newWidget],
        });

        return { id };
      }

      case 'removeWidget': {
        const { id } = payload as { id: string };
        this.setState('globalUI', {
          ...globalUI,
          widgets: globalUI.widgets.filter((w) => w.id !== id),
        });
        return { success: true };
      }

      case 'updateWidget': {
        const p = payload as WidgetUpdatePayload;
        const widgetIndex = globalUI.widgets.findIndex((w) => w.id === p.id);
        if (widgetIndex === -1) {
          return { success: false, error: 'Widget not found' };
        }

        const widget = globalUI.widgets[widgetIndex]!;
        const updatedWidgets = [...globalUI.widgets];
        updatedWidgets[widgetIndex] = {
          ...widget,
          ...(p.props !== undefined && { props: { ...widget.props, ...p.props } }),
          ...(p.slot !== undefined && { slot: p.slot }),
          ...(p.priority !== undefined && { priority: p.priority }),
          ...(p.visible !== undefined && { visible: p.visible }),
        };

        this.setState('globalUI', {
          ...globalUI,
          widgets: updatedWidgets,
        });

        return { success: true };
      }

      case 'setWidgetVisible': {
        const { id, visible } = payload as { id: string; visible: boolean };
        const widgetIndex = globalUI.widgets.findIndex((w) => w.id === id);
        if (widgetIndex === -1) {
          return { success: false, error: 'Widget not found' };
        }

        const updatedWidgets = [...globalUI.widgets];
        updatedWidgets[widgetIndex] = {
          ...updatedWidgets[widgetIndex]!,
          visible,
        };

        this.setState('globalUI', {
          ...globalUI,
          widgets: updatedWidgets,
        });

        return { success: true };
      }

      default:
        throw new Error(`Unknown globalUI action: ${action}`);
    }
  }

  // ========== Widget State Commands ==========

  private handleWidgetStateCommand(action: string, payload: unknown): CommandResult {
    const widgetStateStorage = (this.getState('widgetState') as { data: Record<string, unknown> }) || { data: {} };

    switch (action) {
      case 'set': {
        const p = payload as WidgetStateSetPayload;
        console.log(`[StateStore] widgetState.set "${p.widgetId}":`, JSON.stringify(p.state).slice(0, 100));
        this.setState('widgetState', {
          data: {
            ...widgetStateStorage.data,
            [p.widgetId]: p.state,
          },
        });
        console.log(`[StateStore] widgetState now has ${Object.keys(widgetStateStorage.data).length + 1} entries`);
        return { success: true };
      }

      case 'get': {
        const p = payload as WidgetStateGetPayload;
        const value = widgetStateStorage.data[p.widgetId];
        console.log(`[StateStore] widgetState.get "${p.widgetId}":`, value !== undefined ? 'found' : 'not found');
        return { success: true, state: value } as CommandResult & { state: unknown };
      }

      case 'clear': {
        const p = payload as WidgetStateClearPayload;
        const newData = { ...widgetStateStorage.data };
        delete newData[p.widgetId];
        this.setState('widgetState', { data: newData });
        return { success: true };
      }

      default:
        throw new Error(`Unknown widgetState action: ${action}`);
    }
  }

  // ========== Custom Widgets Commands ==========

  private handleCustomWidgetsCommand(action: string, payload: unknown): CommandResult {
    const customWidgets = (this.getState('customWidgets') as CustomWidgetsState) || { list: [] };

    switch (action) {
      case 'register': {
        const p = payload as CustomWidgetRegisterPayload;

        // Validate name
        if (!p.name || !/^[a-z][a-z0-9-]*$/.test(p.name)) {
          return {
            success: false,
            error: 'Widget name must start with lowercase letter and contain only lowercase letters, numbers, and hyphens',
          };
        }

        // Check if widget already exists
        const existingIndex = customWidgets.list.findIndex((w) => w.name === p.name);
        const now = Date.now();

        const widget: CustomWidgetDefinition = {
          name: p.name,
          description: p.description,
          category: p.category ?? 'custom',
          code: p.code,
          defaultProps: p.defaultProps ?? {},
          propsSchema: p.propsSchema ?? {},
          createdAt: existingIndex !== -1 ? customWidgets.list[existingIndex]!.createdAt : now,
          updatedAt: now,
        };

        let newList: CustomWidgetDefinition[];
        if (existingIndex !== -1) {
          // Update existing
          newList = [...customWidgets.list];
          newList[existingIndex] = widget;
        } else {
          // Add new
          newList = [...customWidgets.list, widget];
        }

        this.setState('customWidgets', { list: newList });

        // Emit event so renderer can update its registry
        this.events.emit('customWidgets.registered', { widget });

        return { success: true, id: p.name };
      }

      case 'unregister': {
        const p = payload as CustomWidgetUnregisterPayload;
        const existingIndex = customWidgets.list.findIndex((w) => w.name === p.name);

        if (existingIndex === -1) {
          return { success: false, error: `Widget "${p.name}" not found` };
        }

        const newList = customWidgets.list.filter((w) => w.name !== p.name);
        this.setState('customWidgets', { list: newList });

        // Emit event so renderer can update its registry
        this.events.emit('customWidgets.unregistered', { name: p.name });

        return { success: true };
      }

      case 'update': {
        const p = payload as CustomWidgetUpdatePayload;
        const existingIndex = customWidgets.list.findIndex((w) => w.name === p.name);

        if (existingIndex === -1) {
          return { success: false, error: `Widget "${p.name}" not found` };
        }

        const existing = customWidgets.list[existingIndex]!;
        const updated: CustomWidgetDefinition = {
          ...existing,
          ...(p.code !== undefined && { code: p.code }),
          ...(p.description !== undefined && { description: p.description }),
          ...(p.defaultProps !== undefined && { defaultProps: p.defaultProps }),
          ...(p.propsSchema !== undefined && { propsSchema: p.propsSchema }),
          updatedAt: Date.now(),
        };

        const newList = [...customWidgets.list];
        newList[existingIndex] = updated;
        this.setState('customWidgets', { list: newList });

        // Emit event so renderer can hot-reload the widget
        this.events.emit('customWidgets.updated', { widget: updated });

        return { success: true };
      }

      case 'list': {
        return {
          success: true,
          widgets: customWidgets.list,
        } as CommandResult & { widgets: CustomWidgetDefinition[] };
      }

      case 'get': {
        const { name } = payload as { name: string };
        const widget = customWidgets.list.find((w) => w.name === name);
        if (!widget) {
          return { success: false, error: `Widget "${name}" not found` };
        }
        return { success: true, widget } as CommandResult & { widget: CustomWidgetDefinition };
      }

      default:
        throw new Error(`Unknown customWidgets action: ${action}`);
    }
  }
}
