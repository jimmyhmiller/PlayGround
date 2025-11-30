import type { Theme } from './theme';

export type WidgetType =
  | 'bar-chart' | 'chat' | 'code-editor' | 'command-runner'
  | 'diff-list' | 'error-test' | 'file-list' | 'flippable-test'
  | 'json-viewer' | 'key-value' | 'layout-settings' | 'markdown'
  | 'progress' | 'stat' | 'test-results' | 'todo-list'
  | 'claude-todo-list' | 'webview';

export interface WidgetDimensions {
  x: number;
  y: number;
  w: number;
  h: number;
  width?: number;
  height?: number;
}

export interface BaseWidgetConfig {
  id: string;
  type: WidgetType;
  label?: string | [string, string];
  dimensions: WidgetDimensions;
  data?: any;
  dataSource?: string;
  command?: string;
  cwd?: string;
}

// Specific widget configs
export interface BarChartConfig extends BaseWidgetConfig {
  type: 'bar-chart';
  data?: number[] | { value: number }[] | { values: number[] };
}

export interface ChatConfig extends BaseWidgetConfig {
  type: 'chat';
  projectId?: string;
}

export interface CodeEditorConfig extends BaseWidgetConfig {
  type: 'code-editor';
  language?: string;
  filePath?: string;
  readOnly?: boolean;
}

export interface CommandRunnerConfig extends BaseWidgetConfig {
  type: 'command-runner';
  command: string;
  cwd?: string;
}

export interface MarkdownConfig extends BaseWidgetConfig {
  type: 'markdown';
  dataSource?: string;
  content?: string;
}

export interface WebViewConfig extends BaseWidgetConfig {
  type: 'webview';
  url?: string;
  backgroundColor?: string;
}

export type WidgetConfig =
  | BarChartConfig | ChatConfig | CodeEditorConfig
  | CommandRunnerConfig | MarkdownConfig | WebViewConfig
  | BaseWidgetConfig;

export interface LayoutSettings {
  gap?: number;
  buffer?: number;
  gridSize?: number;
  widgetGap?: number;
  mode?: string;
}

export interface Dashboard {
  id: string;
  name?: string;
  title?: string;
  subtitle?: string;
  icon?: string;
  theme?: Theme;
  widgets: WidgetConfig[];
  layout?: LayoutSettings;
  layoutSettings?: LayoutSettings;
  _sourcePath?: string;
  _projectId?: string;
  _projectRoot?: string;
}
