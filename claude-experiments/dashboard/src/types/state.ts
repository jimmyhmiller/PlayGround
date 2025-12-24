/**
 * State Management Types
 *
 * Type definitions for the application state
 */

import type { IEventStore } from './events';

/**
 * Window state
 */
export interface WindowState {
  id: string;
  title: string;
  component: string;
  props: Record<string, unknown>;
  x: number;
  y: number;
  width: number;
  height: number;
  zIndex: number;
}

/**
 * Windows collection state
 */
export interface WindowsState {
  list: WindowState[];
  focusedId: string | null;
  nextId: number;
}

/**
 * Theme state
 */
export interface ThemeState {
  current: 'dark' | 'light' | string;
  overrides: Record<string, string>;
}

/**
 * Settings state
 */
export interface SettingsState {
  fontSize: 'small' | 'medium' | 'large' | 'xlarge';
  fontScale: number;
  spacing: 'compact' | 'normal' | 'relaxed';
  commandPaletteShortcut?: string;
}

/**
 * Component instance state
 */
export interface ComponentInstance {
  id: string;
  type: string;
  props: Record<string, unknown>;
  createdAt?: number;
}

/**
 * Components collection state
 */
export interface ComponentsState {
  instances: ComponentInstance[];
}

/**
 * Dashboard state - a named layout of windows with optional theme override
 */
export interface DashboardState {
  id: string;
  name: string;
  projectId: string;
  windows: WindowState[];
  themeOverride?: ThemeState;
  createdAt: number;
  updatedAt: number;
}

/**
 * Project state - groups dashboards with a default theme
 */
export interface ProjectState {
  id: string;
  name: string;
  defaultTheme: ThemeState;
  dashboardIds: string[];
  activeDashboardId: string | null;
  createdAt: number;
  updatedAt: number;
}

/**
 * Projects collection state
 */
export interface ProjectsState {
  list: ProjectState[];
  activeProjectId: string | null;
  nextProjectId: number;
  nextDashboardId: number;
}

/**
 * Dashboards collection state
 */
export interface DashboardsState {
  list: DashboardState[];
}

/**
 * Complete application state
 */
export interface AppState {
  windows: WindowsState;
  theme: ThemeState;
  settings: SettingsState;
  components: ComponentsState;
  projects: ProjectsState;
  dashboards: DashboardsState;
}

/**
 * Command result types
 */
export interface CommandResult {
  success?: boolean;
  noChange?: boolean;
  error?: string;
  id?: string;
}

/**
 * Window create payload
 */
export interface WindowCreatePayload {
  title?: string;
  component: string;
  props?: Record<string, unknown>;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

/**
 * Window update payload
 */
export interface WindowUpdatePayload {
  id: string;
  title?: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  props?: Record<string, unknown>;
}

/**
 * Theme set variable payload
 */
export interface ThemeSetVariablePayload {
  variable: string;
  value: string;
}

/**
 * Settings update payload
 */
export interface SettingsUpdatePayload {
  key: keyof SettingsState;
  value: SettingsState[keyof SettingsState];
}

/**
 * Component add payload
 */
export interface ComponentAddPayload {
  id?: string;
  type: string;
  props?: Record<string, unknown>;
}

/**
 * Component update props payload
 */
export interface ComponentUpdatePropsPayload {
  id: string;
  props: Record<string, unknown>;
}

/**
 * Project create payload
 */
export interface ProjectCreatePayload {
  name: string;
}

/**
 * Project rename payload
 */
export interface ProjectRenamePayload {
  id: string;
  name: string;
}

/**
 * Project set theme payload
 */
export interface ProjectSetThemePayload {
  id: string;
  theme: ThemeState;
}

/**
 * Dashboard create payload
 */
export interface DashboardCreatePayload {
  projectId: string;
  name: string;
}

/**
 * Dashboard rename payload
 */
export interface DashboardRenamePayload {
  id: string;
  name: string;
}

/**
 * Dashboard set theme override payload
 */
export interface DashboardSetThemeOverridePayload {
  id: string;
  themeOverride?: ThemeState;
}

/**
 * StateStore interface
 */
export interface IStateStore {
  events: IEventStore;
  getState(path?: string): unknown;
  setState(path: string, value: unknown): void;
  handleCommand(type: string, payload: unknown): CommandResult;
  save(): void;
}
