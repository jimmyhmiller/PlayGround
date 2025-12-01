import { z } from 'zod';

// Base schema for all widgets
const BaseWidgetConfigSchema = z.object({
  id: z.string(),
  label: z.union([z.string(), z.tuple([z.string(), z.string()])]).optional(),
  x: z.number().optional(),
  y: z.number().optional(),
  w: z.number().optional(),
  h: z.number().optional(),
  width: z.number().optional(),
  height: z.number().optional(),
  dataSource: z.string().optional(),
  command: z.string().optional(),
  cwd: z.string().optional(),
  readOnly: z.boolean().optional(),
  derived: z.boolean().optional(),
  regenerateCommand: z.string().optional(),
  regenerateScript: z.string().optional(),
  regenerate: z.string().optional(),
  testRunner: z.string().optional(),
});

// Stat widget schema
export const StatConfigSchema = BaseWidgetConfigSchema.extend({
  type: z.literal('stat'),
  value: z.union([z.string(), z.number()]).optional(),
});

// BarChart widget schema
export const BarChartConfigSchema = BaseWidgetConfigSchema.extend({
  type: z.literal('bar-chart'),
  data: z.union([
    z.array(z.number()),
    z.array(z.object({ value: z.number() })),
    z.object({ values: z.array(z.number()) })
  ]).optional(),
});

// Progress widget schema
export const ProgressConfigSchema = BaseWidgetConfigSchema.extend({
  type: z.literal('progress'),
  value: z.number().min(0).max(100).optional(),
  text: z.string().optional(),
});

// Chat widget schema
export const ChatConfigSchema = BaseWidgetConfigSchema.extend({
  type: z.literal('chat'),
  projectId: z.string().optional(),
  backend: z.literal('claude').optional(),
  claudeOptions: z.object({
    model: z.string().optional(),
  }).optional(),
});

// CodeEditor widget schema
export const CodeEditorConfigSchema = BaseWidgetConfigSchema.extend({
  type: z.union([z.literal('code-editor'), z.literal('codeEditor')]),
  language: z.string().optional(),
  filePath: z.string().optional(),
  content: z.string().optional(),
  // Explicitly disallow common mistakes
  runCommand: z.never().optional().describe('Use "command" instead of "runCommand"'),
  defaultCode: z.never().optional().describe('Use "content" instead of "defaultCode"'),
  minimap: z.boolean().optional(),
  showLineNumbers: z.boolean().optional(),
});

// CommandRunner widget schema
export const CommandRunnerConfigSchema = BaseWidgetConfigSchema.extend({
  type: z.union([z.literal('command-runner'), z.literal('commandRunner')]),
  command: z.string(),
  autoRun: z.boolean().optional(),
  showOutput: z.boolean().optional(),
});

// TodoList widget schema
export const TodoListConfigSchema = BaseWidgetConfigSchema.extend({
  type: z.union([z.literal('todo-list'), z.literal('todoList')]),
  items: z.array(z.object({
    text: z.string(),
    done: z.boolean(),
  })).optional(),
});

// ClaudeTodoList widget schema
export const ClaudeTodoListConfigSchema = BaseWidgetConfigSchema.extend({
  type: z.union([z.literal('claude-todo-list'), z.literal('claudeTodos')]),
  chatWidgetId: z.string().optional(),
});

// FileList widget schema
export const FileListConfigSchema = BaseWidgetConfigSchema.extend({
  type: z.union([z.literal('file-list'), z.literal('fileList')]),
  files: z.array(z.object({
    name: z.string(),
    status: z.string(),
  })).optional(),
});

// KeyValue widget schema
export const KeyValueConfigSchema = BaseWidgetConfigSchema.extend({
  type: z.union([z.literal('key-value'), z.literal('keyValue')]),
  items: z.array(z.object({
    key: z.string(),
    value: z.string(),
  })).optional(),
});

// DiffList widget schema
export const DiffListConfigSchema = BaseWidgetConfigSchema.extend({
  type: z.union([z.literal('diff-list'), z.literal('diffList')]),
  items: z.array(z.tuple([z.string(), z.number(), z.number()])).optional(),
});

// Markdown widget schema
export const MarkdownConfigSchema = BaseWidgetConfigSchema.extend({
  type: z.literal('markdown'),
  content: z.string().optional(),
  filePath: z.string().optional(),
});

// WebView widget schema
export const WebViewConfigSchema = BaseWidgetConfigSchema.extend({
  type: z.literal('webview'),
  url: z.string().optional(),
  backgroundColor: z.string().optional(),
});

// TestResults widget schema
export const TestResultsConfigSchema = BaseWidgetConfigSchema.extend({
  type: z.union([z.literal('test-results'), z.literal('testResults')]),
  tests: z.array(z.object({
    name: z.string(),
    status: z.enum(['passed', 'failed', 'skipped']),
    duration: z.number().optional(),
    error: z.string().optional(),
  })).optional(),
  testRunner: z.enum(['cargo', 'jest', 'pytest']).optional(),
  autoRun: z.boolean().optional(),
});

// JsonViewer widget schema
export const JsonViewerConfigSchema = BaseWidgetConfigSchema.extend({
  type: z.union([z.literal('json-viewer'), z.literal('jsonViewer')]),
  data: z.unknown().optional(),
});

// LayoutSettings widget schema
export const LayoutSettingsConfigSchema = BaseWidgetConfigSchema.extend({
  type: z.union([z.literal('layout-settings'), z.literal('layoutSettings')]),
});

// Discriminated union of all widget schemas
export const WidgetConfigSchema = z.discriminatedUnion('type', [
  StatConfigSchema,
  BarChartConfigSchema,
  ProgressConfigSchema,
  ChatConfigSchema,
  CodeEditorConfigSchema,
  CommandRunnerConfigSchema,
  TodoListConfigSchema,
  ClaudeTodoListConfigSchema,
  FileListConfigSchema,
  KeyValueConfigSchema,
  DiffListConfigSchema,
  MarkdownConfigSchema,
  WebViewConfigSchema,
  TestResultsConfigSchema,
  JsonViewerConfigSchema,
  LayoutSettingsConfigSchema,
]);

// Export type inference
export type ValidatedWidgetConfig = z.infer<typeof WidgetConfigSchema>;
