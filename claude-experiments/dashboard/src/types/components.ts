/**
 * Component Types
 *
 * Type definitions for React components and the component registry
 */

import type { ReactNode, ComponentType } from 'react';

/**
 * Window component props
 */
export interface WindowProps {
  id: string;
  title: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  minWidth?: number;
  minHeight?: number;
  zIndex?: number;
  isFocused?: boolean;
  children?: ReactNode;
  onClose?: (id: string) => void;
  onFocus?: (id: string) => void;
  onUpdate?: (id: string, updates: WindowUpdates) => void;
}

/**
 * Window position/size updates
 */
export interface WindowUpdates {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  props?: Record<string, unknown>;
}

/**
 * Registered component entry
 */
export interface RegisteredComponent {
  id: string;
  name: string;
  component: ComponentType<unknown>;
  defaultTitle: string;
  defaultWidth?: number;
  defaultHeight?: number;
  description?: string;
}

/**
 * Component registry interface
 */
export interface IComponentRegistry {
  register(entry: RegisteredComponent): void;
  unregister(id: string): void;
  get(id: string): RegisteredComponent | undefined;
  getAll(): RegisteredComponent[];
  has(id: string): boolean;
}

/**
 * Desktop props
 */
export interface DesktopProps {
  children?: ReactNode;
}

/**
 * Window manager context value
 */
export interface WindowManagerContextValue {
  windows: Array<{
    id: string;
    title: string;
    component: string;
    props: Record<string, unknown>;
    x: number;
    y: number;
    width: number;
    height: number;
    zIndex: number;
  }>;
  focusedId: string | null;
  createWindow: (options: {
    title?: string;
    component: string;
    props?: Record<string, unknown>;
    x?: number;
    y?: number;
    width?: number;
    height?: number;
  }) => Promise<{ id: string }>;
  closeWindow: (id: string) => Promise<void>;
  focusWindow: (id: string) => Promise<void>;
  updateWindow: (id: string, updates: WindowUpdates) => Promise<void>;
}

/**
 * Theme context value
 */
export interface ThemeContextValue {
  currentTheme: string;
  overrides: Record<string, string>;
  setTheme: (theme: string) => Promise<void>;
  setVariable: (variable: string, value: string) => Promise<void>;
  resetVariable: (variable: string) => Promise<void>;
  resetOverrides: () => Promise<void>;
  loading: boolean;
}

/**
 * Settings context value
 */
export interface SettingsContextValue {
  settings: {
    fontSize: 'small' | 'medium' | 'large';
    fontScale: number;
    spacing: 'compact' | 'normal' | 'relaxed';
  };
  updateSetting: (key: string, value: unknown) => Promise<void>;
  resetSettings: () => Promise<void>;
  loading: boolean;
}

/**
 * Event log entry for display
 */
export interface EventLogEntry {
  id: string;
  type: string;
  payload: unknown;
  timestamp: number;
  source: string;
}

/**
 * Git diff viewer props
 */
export interface GitDiffViewerProps {
  filePath?: string;
  showStaged?: boolean;
}

/**
 * Code editor props
 */
export interface CodeMirrorEditorProps {
  value: string;
  onChange?: (value: string) => void;
  language?: 'javascript' | 'typescript' | 'json' | 'css' | 'html';
  readOnly?: boolean;
  lineNumbers?: boolean;
}

/**
 * Theme editor props
 */
export interface ThemeEditorProps {
  onClose?: () => void;
}

/**
 * Settings editor props
 */
export interface SettingsEditorProps {
  onClose?: () => void;
}

/**
 * Command palette props
 */
export interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
}

/**
 * Command definition
 */
export interface Command {
  id: string;
  label: string;
  description?: string;
  shortcut?: string;
  action: () => void | Promise<void>;
  category?: string;
}
