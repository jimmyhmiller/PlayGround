import { FC } from 'react';
import type { Theme, WidgetConfig, Dashboard, LayoutSettings } from '../types';

export interface WidgetProps {
  theme: Theme;
  config: WidgetConfig;
  dashboardId?: string;
  dashboard?: Dashboard;
  layout?: LayoutSettings;
  widgetKey?: string;
  currentConversationId?: string | null;
  setCurrentConversationId?: (id: string | null) => void;
  widgetConversations?: Record<string, string | null>;
  reloadTrigger?: number;
}

// Placeholder widget component for migration
const PlaceholderWidget: FC<WidgetProps & { widgetType: string }> = ({ theme, widgetType }) => (
  <div style={{
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    fontFamily: theme.textBody,
    color: theme.accent,
    fontSize: '1.2rem'
  }}>
    {widgetType} (Under Migration)
  </div>
);

// Widget implementations (stubs for now)
export const BarChart: FC<WidgetProps> = (props) => <PlaceholderWidget {...props} widgetType="BarChart" />;
export const Chat: FC<WidgetProps> = (props) => <PlaceholderWidget {...props} widgetType="Chat" />;
export const CodeEditor: FC<WidgetProps> = (props) => <PlaceholderWidget {...props} widgetType="CodeEditor" />;
export const CommandRunner: FC<WidgetProps> = (props) => <PlaceholderWidget {...props} widgetType="CommandRunner" />;
export const DiffList: FC<WidgetProps> = (props) => <PlaceholderWidget {...props} widgetType="DiffList" />;
export const ErrorTest: FC<WidgetProps> = (props) => <PlaceholderWidget {...props} widgetType="ErrorTest" />;
export const FileList: FC<WidgetProps> = (props) => <PlaceholderWidget {...props} widgetType="FileList" />;
export const FlippableTest: FC<WidgetProps> = (props) => <PlaceholderWidget {...props} widgetType="FlippableTest" />;
export const JsonViewer: FC<WidgetProps> = (props) => <PlaceholderWidget {...props} widgetType="JsonViewer" />;
export const KeyValue: FC<WidgetProps> = (props) => <PlaceholderWidget {...props} widgetType="KeyValue" />;
export const LayoutSettingsWidget: FC<WidgetProps> = (props) => <PlaceholderWidget {...props} widgetType="LayoutSettings" />;
export const Markdown: FC<WidgetProps> = (props) => <PlaceholderWidget {...props} widgetType="Markdown" />;
export const Progress: FC<WidgetProps> = (props) => <PlaceholderWidget {...props} widgetType="Progress" />;
export const Stat: FC<WidgetProps> = (props) => <PlaceholderWidget {...props} widgetType="Stat" />;
export const TestResults: FC<WidgetProps> = (props) => <PlaceholderWidget {...props} widgetType="TestResults" />;
export const TodoList: FC<WidgetProps> = (props) => <PlaceholderWidget {...props} widgetType="TodoList" />;
export const ClaudeTodoList: FC<WidgetProps> = (props) => <PlaceholderWidget {...props} widgetType="ClaudeTodoList" />;
export const WebView: FC<WidgetProps> = (props) => <PlaceholderWidget {...props} widgetType="WebView" />;

// Widget Registry
export const WIDGET_REGISTRY: Record<string, FC<WidgetProps>> = {
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
};
