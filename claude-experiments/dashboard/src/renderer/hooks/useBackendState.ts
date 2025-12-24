import { useState, useEffect, useCallback, useRef } from 'react';
import type { DashboardEvent } from '../../types/events';
import type {
  CommandResult,
  SettingsState,
  ThemeState,
  WindowState,
  WindowsState,
  ProjectsState,
  ProjectState,
  DashboardsState,
  DashboardState,
} from '../../types/state';

/**
 * Shallow equality comparison for objects
 */
function shallowEqual<T>(a: T, b: T): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  if (typeof a !== 'object' || typeof b !== 'object') return a === b;
  if (Array.isArray(a) !== Array.isArray(b)) return false;

  const keysA = Object.keys(a as object);
  const keysB = Object.keys(b as object);
  if (keysA.length !== keysB.length) return false;
  return keysA.every(key => (a as Record<string, unknown>)[key] === (b as Record<string, unknown>)[key]);
}

/**
 * Selector-based state subscription (like Redux useSelector)
 * Only re-renders when the selected value changes.
 */
export function useBackendStateSelector<T, S>(
  path: string,
  selector: (state: T) => S,
  equalityFn: (a: S, b: S) => boolean = shallowEqual
): [S | null, boolean] {
  const [selectedState, setSelectedState] = useState<S | null>(null);
  const [loading, setLoading] = useState(true);

  // Keep selector in ref so we always use the latest
  const selectorRef = useRef(selector);
  selectorRef.current = selector;

  const equalityRef = useRef(equalityFn);
  equalityRef.current = equalityFn;

  // Load initial state
  useEffect(() => {
    let mounted = true;

    window.stateAPI.get(path).then(state => {
      if (mounted) {
        setSelectedState(selectorRef.current(state as T));
        setLoading(false);
      }
    }).catch(err => {
      console.error(`Failed to load state at path "${path}":`, err);
      if (mounted) {
        setLoading(false);
      }
    });

    return () => { mounted = false; };
  }, [path]);

  // Subscribe to changes, only update if selected value changed
  useEffect(() => {
    const unsubscribe = window.stateAPI.subscribe(path, async (event: DashboardEvent) => {
      // Get full state - either from event or refetch
      let fullState: T;
      const payload = event.payload as { path?: string; value?: T };
      if (payload?.path === path) {
        fullState = payload.value as T;
      } else {
        // Child path changed, refetch
        fullState = await window.stateAPI.get(path) as T;
      }

      if (fullState !== undefined) {
        const newSelected = selectorRef.current(fullState);
        setSelectedState(prev => {
          if (prev !== null && equalityRef.current(prev, newSelected)) {
            return prev; // No change, no re-render
          }
          return newSelected;
        });
      }
    });

    return unsubscribe;
  }, [path]);

  return [selectedState, loading];
}

type DispatchFn = (type: string, payload?: unknown) => Promise<CommandResult>;

/**
 * Hook for backend-driven state management
 */
export function useBackendState<T>(path: string): [T | null, DispatchFn, boolean] {
  const [state, setState] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const stateRef = useRef(state);

  // Keep ref in sync
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  // Load initial state
  useEffect(() => {
    let mounted = true;

    async function loadState(): Promise<void> {
      try {
        const initialState = await window.stateAPI.get(path) as T;
        if (mounted) {
          setState(initialState);
          setLoading(false);
        }
      } catch (err) {
        console.error(`Failed to load state at path "${path}":`, err);
        if (mounted) {
          setLoading(false);
        }
      }
    }

    loadState();

    return () => {
      mounted = false;
    };
  }, [path]);

  // Subscribe to state changes
  useEffect(() => {
    const unsubscribe = window.stateAPI.subscribe(path, (event: DashboardEvent) => {
      // Event payload contains { path, value }
      const payload = event.payload as { path?: string; value?: T };
      if (payload && payload.path) {
        const eventPath = payload.path;
        const eventValue = payload.value;

        // If the event path matches our path exactly, update the whole state
        if (eventPath === path) {
          setState(eventValue as T);
        } else if (eventPath.startsWith(path + '.')) {
          // If the event path is a child of our path, we need to refetch
          window.stateAPI.get(path).then((newState) => {
            setState(newState as T);
          });
        }
      }
    });

    return unsubscribe;
  }, [path]);

  // Dispatch function to send commands
  const dispatch = useCallback(async (type: string, payload?: unknown): Promise<CommandResult> => {
    try {
      const result = await window.stateAPI.command(type, payload);
      return result;
    } catch (err) {
      console.error(`State command "${type}" failed:`, err);
      throw err;
    }
  }, []);

  return [state, dispatch, loading];
}

/**
 * Get dispatch function for sending commands (without subscribing to state)
 */
export function useDispatch(): DispatchFn {
  return useCallback(async (type: string, payload?: unknown): Promise<CommandResult> => {
    try {
      const result = await window.stateAPI.command(type, payload);
      return result;
    } catch (err) {
      console.error(`State command "${type}" failed:`, err);
      throw err;
    }
  }, []);
}

interface WindowStateResult {
  windows: WindowState[];
  focusedId: string | null;
  createWindow: (options: {
    title?: string;
    component: string;
    props?: Record<string, unknown>;
    x?: number;
    y?: number;
    width?: number;
    height?: number;
  }) => Promise<CommandResult>;
  closeWindow: (id: string) => Promise<CommandResult>;
  focusWindow: (id: string) => Promise<CommandResult>;
  updateWindow: (id: string, updates: Partial<WindowState>) => Promise<CommandResult>;
  loading: boolean;
}

/**
 * Hook for window state (full subscription - use sparingly)
 */
export function useWindowState(): WindowStateResult {
  const [state, dispatch, loading] = useBackendState<WindowsState>('windows');

  const createWindow = useCallback(
    (options: {
      title?: string;
      component: string;
      props?: Record<string, unknown>;
      x?: number;
      y?: number;
      width?: number;
      height?: number;
    }) => dispatch('windows.create', options),
    [dispatch]
  );

  const closeWindow = useCallback(
    (id: string) => dispatch('windows.close', { id }),
    [dispatch]
  );

  const focusWindow = useCallback(
    (id: string) => dispatch('windows.focus', { id }),
    [dispatch]
  );

  const updateWindow = useCallback(
    (id: string, updates: Partial<WindowState>) => dispatch('windows.update', { id, ...updates }),
    [dispatch]
  );

  return {
    windows: state?.list ?? [],
    focusedId: state?.focusedId ?? null,
    createWindow,
    closeWindow,
    focusWindow,
    updateWindow,
    loading,
  };
}

/**
 * Hook for window list only (doesn't re-render on focus changes)
 */
export function useWindowList(): [WindowState[], boolean] {
  const [list, loading] = useBackendStateSelector<WindowsState, WindowState[]>(
    'windows',
    state => state?.list ?? []
  );
  return [list ?? [], loading];
}

/**
 * Hook for focused window ID only
 */
export function useWindowFocus(): [string | null, boolean] {
  const [focusedId, loading] = useBackendStateSelector<WindowsState, string | null>(
    'windows',
    state => state?.focusedId ?? null
  );
  return [focusedId, loading];
}

/**
 * Hook for a single window by ID - only re-renders when THAT window changes
 */
export function useWindow(windowId: string): [WindowState | null, boolean] {
  const [window, loading] = useBackendStateSelector<WindowsState, WindowState | null>(
    'windows',
    state => state?.list?.find(w => w.id === windowId) ?? null
  );
  return [window, loading];
}

interface WindowCommandsResult {
  createWindow: (options: {
    title?: string;
    component: string;
    props?: Record<string, unknown>;
    x?: number;
    y?: number;
    width?: number;
    height?: number;
  }) => Promise<CommandResult>;
  closeWindow: (id: string) => Promise<CommandResult>;
  focusWindow: (id: string) => Promise<CommandResult>;
  updateWindow: (id: string, updates: Partial<WindowState>) => Promise<CommandResult>;
}

/**
 * Hook for window commands without subscribing to state
 */
export function useWindowCommands(): WindowCommandsResult {
  const dispatch = useDispatch();

  const createWindow = useCallback(
    (options: {
      title?: string;
      component: string;
      props?: Record<string, unknown>;
      x?: number;
      y?: number;
      width?: number;
      height?: number;
    }) => dispatch('windows.create', options),
    [dispatch]
  );

  const closeWindow = useCallback(
    (id: string) => dispatch('windows.close', { id }),
    [dispatch]
  );

  const focusWindow = useCallback(
    (id: string) => dispatch('windows.focus', { id }),
    [dispatch]
  );

  const updateWindow = useCallback(
    (id: string, updates: Partial<WindowState>) => dispatch('windows.update', { id, ...updates }),
    [dispatch]
  );

  return { createWindow, closeWindow, focusWindow, updateWindow };
}

interface ThemeStateResult {
  currentTheme: string;
  overrides: Record<string, string>;
  setTheme: (theme: string) => Promise<CommandResult>;
  setVariable: (variable: string, value: string) => Promise<CommandResult>;
  resetVariable: (variable: string) => Promise<CommandResult>;
  resetOverrides: () => Promise<CommandResult>;
  loading: boolean;
}

/**
 * Hook for theme state
 */
export function useThemeState(): ThemeStateResult {
  const [state, dispatch, loading] = useBackendState<ThemeState>('theme');

  const setTheme = useCallback(
    (theme: string) => dispatch('theme.set', { theme }),
    [dispatch]
  );

  const setVariable = useCallback(
    (variable: string, value: string) => dispatch('theme.setVariable', { variable, value }),
    [dispatch]
  );

  const resetVariable = useCallback(
    (variable: string) => dispatch('theme.resetVariable', { variable }),
    [dispatch]
  );

  const resetOverrides = useCallback(
    () => dispatch('theme.resetOverrides', {}),
    [dispatch]
  );

  return {
    currentTheme: state?.current ?? 'dark',
    overrides: state?.overrides ?? {},
    setTheme,
    setVariable,
    resetVariable,
    resetOverrides,
    loading,
  };
}

interface SettingsStateResult {
  settings: SettingsState;
  updateSetting: (key: string, value: unknown) => Promise<CommandResult>;
  resetSettings: () => Promise<CommandResult>;
  loading: boolean;
}

/**
 * Hook for settings state
 */
export function useSettingsState(): SettingsStateResult {
  const [state, dispatch, loading] = useBackendState<SettingsState>('settings');

  const updateSetting = useCallback(
    (key: string, value: unknown) => dispatch('settings.update', { key, value }),
    [dispatch]
  );

  const resetSettings = useCallback(
    () => dispatch('settings.reset', {}),
    [dispatch]
  );

  return {
    settings: state ?? { fontSize: 'medium', fontScale: 1.0, spacing: 'normal' },
    updateSetting,
    resetSettings,
    loading,
  };
}

interface ComponentInstance {
  id: string;
  type: string;
  props: Record<string, unknown>;
}

interface ComponentsStateResult {
  instances: ComponentInstance[];
  addInstance: (type: string, props?: Record<string, unknown>) => Promise<CommandResult>;
  removeInstance: (id: string) => Promise<CommandResult>;
  updateInstanceProps: (id: string, props: Record<string, unknown>) => Promise<CommandResult>;
  loading: boolean;
}

/**
 * Hook for component registry state
 */
export function useComponentsState(): ComponentsStateResult {
  const [state, dispatch, loading] = useBackendState<{ instances: ComponentInstance[] }>('components');

  const addInstance = useCallback(
    (type: string, props?: Record<string, unknown>) => dispatch('components.add', { type, props }),
    [dispatch]
  );

  const removeInstance = useCallback(
    (id: string) => dispatch('components.remove', { id }),
    [dispatch]
  );

  const updateInstanceProps = useCallback(
    (id: string, props: Record<string, unknown>) => dispatch('components.updateProps', { id, props }),
    [dispatch]
  );

  return {
    instances: state?.instances ?? [],
    addInstance,
    removeInstance,
    updateInstanceProps,
    loading,
  };
}

// ========== Projects Hooks ==========

interface ProjectsStateResult {
  projects: ProjectState[];
  activeProjectId: string | null;
  activeProject: ProjectState | null;
  createProject: (name: string) => Promise<CommandResult>;
  deleteProject: (id: string) => Promise<CommandResult>;
  renameProject: (id: string, name: string) => Promise<CommandResult>;
  setProjectTheme: (id: string, theme: ThemeState) => Promise<CommandResult>;
  switchProject: (id: string) => Promise<CommandResult>;
  loading: boolean;
}

/**
 * Hook for projects state
 */
export function useProjectsState(): ProjectsStateResult {
  const [state, dispatch, loading] = useBackendState<ProjectsState>('projects');

  const createProject = useCallback(
    (name: string) => dispatch('projects.create', { name }),
    [dispatch]
  );

  const deleteProject = useCallback(
    (id: string) => dispatch('projects.delete', { id }),
    [dispatch]
  );

  const renameProject = useCallback(
    (id: string, name: string) => dispatch('projects.rename', { id, name }),
    [dispatch]
  );

  const setProjectTheme = useCallback(
    (id: string, theme: ThemeState) => dispatch('projects.setTheme', { id, theme }),
    [dispatch]
  );

  const switchProject = useCallback(
    (id: string) => dispatch('projects.switch', { id }),
    [dispatch]
  );

  const activeProject = state?.list.find((p) => p.id === state.activeProjectId) ?? null;

  return {
    projects: state?.list ?? [],
    activeProjectId: state?.activeProjectId ?? null,
    activeProject,
    createProject,
    deleteProject,
    renameProject,
    setProjectTheme,
    switchProject,
    loading,
  };
}

/**
 * Hook for active project only
 */
export function useActiveProject(): [ProjectState | null, boolean] {
  const [project, loading] = useBackendStateSelector<ProjectsState, ProjectState | null>(
    'projects',
    (state) => state?.list.find((p) => p.id === state.activeProjectId) ?? null
  );
  return [project, loading];
}

interface ProjectCommandsResult {
  createProject: (name: string) => Promise<CommandResult>;
  deleteProject: (id: string) => Promise<CommandResult>;
  renameProject: (id: string, name: string) => Promise<CommandResult>;
  setProjectTheme: (id: string, theme: ThemeState) => Promise<CommandResult>;
  switchProject: (id: string) => Promise<CommandResult>;
}

/**
 * Hook for project commands without subscribing to state
 */
export function useProjectCommands(): ProjectCommandsResult {
  const dispatch = useDispatch();

  const createProject = useCallback(
    (name: string) => dispatch('projects.create', { name }),
    [dispatch]
  );

  const deleteProject = useCallback(
    (id: string) => dispatch('projects.delete', { id }),
    [dispatch]
  );

  const renameProject = useCallback(
    (id: string, name: string) => dispatch('projects.rename', { id, name }),
    [dispatch]
  );

  const setProjectTheme = useCallback(
    (id: string, theme: ThemeState) => dispatch('projects.setTheme', { id, theme }),
    [dispatch]
  );

  const switchProject = useCallback(
    (id: string) => dispatch('projects.switch', { id }),
    [dispatch]
  );

  return { createProject, deleteProject, renameProject, setProjectTheme, switchProject };
}

// ========== Dashboards Hooks ==========

interface DashboardsStateResult {
  dashboards: DashboardState[];
  activeDashboard: DashboardState | null;
  createDashboard: (projectId: string, name: string) => Promise<CommandResult>;
  deleteDashboard: (id: string) => Promise<CommandResult>;
  renameDashboard: (id: string, name: string) => Promise<CommandResult>;
  switchDashboard: (id: string) => Promise<CommandResult>;
  setDashboardThemeOverride: (id: string, themeOverride?: ThemeState) => Promise<CommandResult>;
  saveLayout: () => Promise<CommandResult>;
  loading: boolean;
}

/**
 * Hook for dashboards state
 */
export function useDashboardsState(): DashboardsStateResult {
  const [dashboardsState, dispatch, dashboardsLoading] = useBackendState<DashboardsState>('dashboards');
  const [projectsState, , projectsLoading] = useBackendState<ProjectsState>('projects');

  const createDashboard = useCallback(
    (projectId: string, name: string) => dispatch('dashboards.create', { projectId, name }),
    [dispatch]
  );

  const deleteDashboard = useCallback(
    (id: string) => dispatch('dashboards.delete', { id }),
    [dispatch]
  );

  const renameDashboard = useCallback(
    (id: string, name: string) => dispatch('dashboards.rename', { id, name }),
    [dispatch]
  );

  const switchDashboard = useCallback(
    (id: string) => dispatch('dashboards.switch', { id }),
    [dispatch]
  );

  const setDashboardThemeOverride = useCallback(
    (id: string, themeOverride?: ThemeState) =>
      dispatch('dashboards.setThemeOverride', { id, themeOverride }),
    [dispatch]
  );

  const saveLayout = useCallback(() => dispatch('dashboards.saveLayout', {}), [dispatch]);

  // Find active dashboard
  const activeProject = projectsState?.list.find((p) => p.id === projectsState.activeProjectId);
  const activeDashboard =
    dashboardsState?.list.find((d) => d.id === activeProject?.activeDashboardId) ?? null;

  return {
    dashboards: dashboardsState?.list ?? [],
    activeDashboard,
    createDashboard,
    deleteDashboard,
    renameDashboard,
    switchDashboard,
    setDashboardThemeOverride,
    saveLayout,
    loading: dashboardsLoading || projectsLoading,
  };
}

/**
 * Hook for active dashboard only
 */
export function useActiveDashboard(): [DashboardState | null, boolean] {
  const [projectsState, projectsLoading] = useBackendStateSelector<ProjectsState, ProjectsState | null>(
    'projects',
    (state) => state
  );
  const [dashboardsState, dashboardsLoading] = useBackendStateSelector<DashboardsState, DashboardsState | null>(
    'dashboards',
    (state) => state
  );

  const activeProject = projectsState?.list.find((p) => p.id === projectsState?.activeProjectId);
  const activeDashboard =
    dashboardsState?.list.find((d) => d.id === activeProject?.activeDashboardId) ?? null;

  return [activeDashboard, projectsLoading || dashboardsLoading];
}

interface DashboardCommandsResult {
  createDashboard: (projectId: string, name: string) => Promise<CommandResult>;
  deleteDashboard: (id: string) => Promise<CommandResult>;
  renameDashboard: (id: string, name: string) => Promise<CommandResult>;
  switchDashboard: (id: string) => Promise<CommandResult>;
  setDashboardThemeOverride: (id: string, themeOverride?: ThemeState) => Promise<CommandResult>;
  saveLayout: () => Promise<CommandResult>;
}

/**
 * Hook for dashboard commands without subscribing to state
 */
export function useDashboardCommands(): DashboardCommandsResult {
  const dispatch = useDispatch();

  const createDashboard = useCallback(
    (projectId: string, name: string) => dispatch('dashboards.create', { projectId, name }),
    [dispatch]
  );

  const deleteDashboard = useCallback(
    (id: string) => dispatch('dashboards.delete', { id }),
    [dispatch]
  );

  const renameDashboard = useCallback(
    (id: string, name: string) => dispatch('dashboards.rename', { id, name }),
    [dispatch]
  );

  const switchDashboard = useCallback(
    (id: string) => dispatch('dashboards.switch', { id }),
    [dispatch]
  );

  const setDashboardThemeOverride = useCallback(
    (id: string, themeOverride?: ThemeState) =>
      dispatch('dashboards.setThemeOverride', { id, themeOverride }),
    [dispatch]
  );

  const saveLayout = useCallback(() => dispatch('dashboards.saveLayout', {}), [dispatch]);

  return {
    createDashboard,
    deleteDashboard,
    renameDashboard,
    switchDashboard,
    setDashboardThemeOverride,
    saveLayout,
  };
}

// ========== Combined Project/Dashboard Tree Hook ==========

interface ProjectWithDashboards {
  project: ProjectState;
  dashboards: DashboardState[];
}

interface ProjectDashboardTreeResult {
  tree: ProjectWithDashboards[];
  activeProjectId: string | null;
  activeDashboardId: string | null;
  loading: boolean;
}

/**
 * Hook for getting projects with their dashboards (for quick switcher)
 */
export function useProjectDashboardTree(): ProjectDashboardTreeResult {
  const [projectsState, projectsLoading] = useBackendStateSelector<ProjectsState, ProjectsState | null>(
    'projects',
    (state) => state
  );
  const [dashboardsState, dashboardsLoading] = useBackendStateSelector<DashboardsState, DashboardsState | null>(
    'dashboards',
    (state) => state
  );

  const tree: ProjectWithDashboards[] = (projectsState?.list ?? []).map((project) => ({
    project,
    dashboards: (dashboardsState?.list ?? []).filter((d) => d.projectId === project.id),
  }));

  const activeProject = projectsState?.list.find((p) => p.id === projectsState?.activeProjectId);

  return {
    tree,
    activeProjectId: projectsState?.activeProjectId ?? null,
    activeDashboardId: activeProject?.activeDashboardId ?? null,
    loading: projectsLoading || dashboardsLoading,
  };
}
