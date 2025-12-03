import { FC } from 'react';
import type { Theme, WidgetConfig, Dashboard, LayoutSettings as LayoutSettingsType } from '../types';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';

// Import ALL widget implementations
import { Stat } from './Stat';
import { Progress } from './Progress';
import { TodoList } from './TodoList';
import { FileList } from './FileList';
import { KeyValue } from './KeyValue';
import { DiffList } from './DiffList';
import { Markdown } from './Markdown';
import { BarChart } from './BarChart';
import { Chat } from './Chat';
import { CodeEditor } from './CodeEditor';
import { CommandRunner } from './CommandRunner';
import { ClaudeTodoList } from './ClaudeTodoList';
import { ErrorTest } from './ErrorTest';
import { FlippableTest } from './FlippableTest';
import { JsonViewer } from './JsonViewer';
import { LayoutSettings } from './LayoutSettings';
import { TestResults } from './TestResults';
import { WebView } from './WebView';
import { NestedDashboard } from './NestedDashboard';

// Re-export them all
export {
  Stat,
  Progress,
  TodoList,
  FileList,
  KeyValue,
  DiffList,
  Markdown,
  BarChart,
  Chat,
  CodeEditor,
  CommandRunner,
  ClaudeTodoList,
  ErrorTest,
  FlippableTest,
  JsonViewer,
  LayoutSettings,
  TestResults,
  WebView,
  NestedDashboard
};

export type WidgetProps = BaseWidgetComponentProps;

// Alias for LayoutSettings to match registry key
export const LayoutSettingsWidget = LayoutSettings;

// Widget Registry - supports both kebab-case and camelCase
export const WIDGET_REGISTRY: Record<string, FC<WidgetProps>> = {
  // Kebab-case (legacy)
  'bar-chart': BarChart,
  'chat': Chat,
  'code-editor': CodeEditor,
  'command-runner': CommandRunner,
  'diff-list': DiffList,
  'error-test': ErrorTest,
  'file-list': FileList,
  'flippable-test': FlippableTest,
  'json-viewer': JsonViewer,
  'key-value': KeyValue,
  'layout-settings': LayoutSettingsWidget,
  'markdown': Markdown,
  'progress': Progress,
  'stat': Stat,
  'test-results': TestResults,
  'todo-list': TodoList,
  'claude-todo-list': ClaudeTodoList,
  'webview': WebView,
  'nested-dashboard': NestedDashboard,

  // CamelCase (current standard)
  'barChart': BarChart,
  'codeEditor': CodeEditor,
  'commandRunner': CommandRunner,
  'diffList': DiffList,
  'errorTest': ErrorTest,
  'fileList': FileList,
  'flippableTest': FlippableTest,
  'jsonViewer': JsonViewer,
  'keyValue': KeyValue,
  'layoutSettings': LayoutSettingsWidget,
  'testResults': TestResults,
  'todoList': TodoList,
  'claudeTodos': ClaudeTodoList,
  'webView': WebView,
  'nestedDashboard': NestedDashboard,
};
