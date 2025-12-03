import type { Theme } from './theme';

export type WidgetType =
  | 'bar-chart' | 'chat' | 'code-editor' | 'command-runner'
  | 'diff-list' | 'error-test' | 'file-list' | 'flippable-test'
  | 'json-viewer' | 'key-value' | 'layout-settings' | 'markdown'
  | 'nested-dashboard' | 'progress' | 'stat' | 'test-results' | 'todo-list'
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
  dimensions?: WidgetDimensions;
  x?: number;
  y?: number;
  w?: number;
  h?: number;
  width?: number;
  height?: number;
  dataSource?: string;
  command?: string;
  cwd?: string;
  readOnly?: boolean;
  derived?: boolean;
  regenerateCommand?: string;
  regenerateScript?: string;
  regenerate?: string;
  testRunner?: string;
}

// Specific widget configs
export interface BarChartConfig extends BaseWidgetConfig {
  type: 'bar-chart';
  data?: number[] | { value: number }[] | { values: number[] };
}

export interface ChatConfig extends BaseWidgetConfig {
  type: 'chat';
  projectId?: string;
  backend?: 'claude';
  claudeOptions?: {
    model?: string;
  };
}

export interface CodeEditorConfig extends BaseWidgetConfig {
  type: 'code-editor';
  language?: string;
  filePath?: string;
  content?: string;
}

export interface CommandRunnerConfig extends BaseWidgetConfig {
  type: 'command-runner';
  command: string;
  autoRun?: boolean;
  showOutput?: boolean;
}

export interface MarkdownConfig extends BaseWidgetConfig {
  type: 'markdown';
  content?: string;
}

export interface WebViewConfig extends BaseWidgetConfig {
  type: 'webview';
  url?: string;
  backgroundColor?: string;
}

export interface StatConfig extends BaseWidgetConfig {
  type: 'stat';
  value?: string | number;
}

export interface ProgressConfig extends BaseWidgetConfig {
  type: 'progress';
  value?: number;
  text?: string;
}

export interface TodoListConfig extends BaseWidgetConfig {
  type: 'todo-list';
  items?: Array<{ text: string; done: boolean }>;
}

export interface ClaudeTodoListConfig extends BaseWidgetConfig {
  type: 'claude-todo-list';
  chatWidgetId?: string;
}

export interface KeyValueConfig extends BaseWidgetConfig {
  type: 'key-value';
  items?: Array<{ key: string; value: string }>;
}

export interface FileListConfig extends BaseWidgetConfig {
  type: 'file-list';
  files?: Array<{ name: string; status: string }>;
}

export interface DiffListConfig extends BaseWidgetConfig {
  type: 'diff-list';
  items?: Array<[string, number, number]>;
}

export interface TestResultsConfig extends BaseWidgetConfig {
  type: 'test-results';
  tests?: Array<{ name: string; status: 'passed' | 'failed' | 'skipped'; duration?: number; error?: string }>;
}

export interface JsonViewerConfig extends BaseWidgetConfig {
  type: 'json-viewer';
  data?: unknown;
}

export interface LayoutSettingsConfig extends BaseWidgetConfig {
  type: 'layout-settings';
}

export interface NestedDashboardConfig extends BaseWidgetConfig {
  type: 'nested-dashboard';
  dashboard: Dashboard;
}

export type WidgetConfig =
  | BarChartConfig
  | ChatConfig
  | CodeEditorConfig
  | CommandRunnerConfig
  | MarkdownConfig
  | WebViewConfig
  | StatConfig
  | ProgressConfig
  | TodoListConfig
  | ClaudeTodoListConfig
  | KeyValueConfig
  | FileListConfig
  | DiffListConfig
  | TestResultsConfig
  | JsonViewerConfig
  | LayoutSettingsConfig
  | NestedDashboardConfig;

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
