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
  pinned?: boolean;
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
  widgetState: Record<string, unknown>;  // widgetId -> widget-specific state
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
  rootDir?: string;  // Root directory for this project (used by Claude for cwd)
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

// ========== Global UI State ==========

/**
 * Slot position configuration
 */
export type SlotPosition =
  | { type: 'corner'; corner: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' }
  | { type: 'bar'; edge: 'top' | 'bottom' }
  | { type: 'panel'; side: 'left' | 'right'; width?: number };

/**
 * Slot definition - a named position where widgets can be placed
 */
export interface SlotState {
  id: string;
  position: SlotPosition;
  zIndex?: number;
}

/**
 * Widget instance - a configured widget in a slot
 */
export interface WidgetState {
  id: string;
  type: string;
  slot: string;
  props: Record<string, unknown>;
  priority?: number;
  visible?: boolean;
}

/**
 * Global UI state - slots and widget instances
 */
export interface GlobalUIState {
  slots: SlotState[];
  widgets: WidgetState[];
}

/**
 * Widget state storage - persists widget state across dashboard switches
 * Keyed by widget ID (scope::path)
 */
export interface WidgetStateStorage {
  data: Record<string, unknown>;
}

/**
 * Custom widget definition - a dynamically registered widget type
 */
export interface CustomWidgetDefinition {
  /** Unique name for the widget type (e.g., "my-chart") */
  name: string;
  /** Display description for discovery */
  description: string;
  /** Category for grouping (e.g., "display", "data", "input") */
  category: string;
  /** React component code as string - will be compiled at runtime */
  code: string;
  /** Default props for the widget */
  defaultProps: Record<string, unknown>;
  /** Props schema for documentation */
  propsSchema: Record<string, string>;
  /** Creation timestamp */
  createdAt: number;
  /** Last update timestamp */
  updatedAt: number;
}

/**
 * Custom widgets state - stores dynamically registered widget types
 */
export interface CustomWidgetsState {
  list: CustomWidgetDefinition[];
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
  globalUI: GlobalUIState;
  widgetState: WidgetStateStorage;
  customWidgets: CustomWidgetsState;
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
  pinned?: boolean;
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
  rootDir?: string;
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

// ========== Global UI Payloads ==========

/**
 * Slot add payload
 */
export interface SlotAddPayload {
  id: string;
  position: SlotPosition;
  zIndex?: number;
}

/**
 * Widget add payload
 */
export interface WidgetAddPayload {
  id?: string;
  type: string;
  slot: string;
  props?: Record<string, unknown>;
  priority?: number;
}

/**
 * Widget update payload
 */
export interface WidgetUpdatePayload {
  id: string;
  props?: Record<string, unknown>;
  slot?: string;
  priority?: number;
  visible?: boolean;
}

// ========== Widget State Payloads ==========

/**
 * Widget state set payload - save widget state to active dashboard
 */
export interface WidgetStateSetPayload {
  widgetId: string;
  state: unknown;
}

/**
 * Widget state get payload
 */
export interface WidgetStateGetPayload {
  widgetId: string;
}

/**
 * Widget state clear payload
 */
export interface WidgetStateClearPayload {
  widgetId: string;
}

// ========== Custom Widgets Payloads ==========

/**
 * Custom widget register payload
 */
export interface CustomWidgetRegisterPayload {
  name: string;
  description: string;
  category?: string;
  code: string;
  defaultProps?: Record<string, unknown>;
  propsSchema?: Record<string, string>;
}

/**
 * Custom widget unregister payload
 */
export interface CustomWidgetUnregisterPayload {
  name: string;
}

/**
 * Custom widget update payload
 */
export interface CustomWidgetUpdatePayload {
  name: string;
  code?: string;
  description?: string;
  defaultProps?: Record<string, unknown>;
  propsSchema?: Record<string, string>;
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
