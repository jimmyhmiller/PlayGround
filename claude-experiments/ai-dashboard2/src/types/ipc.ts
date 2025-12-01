import type { Dashboard, WidgetConfig, WidgetDimensions, LayoutSettings } from './dashboard';
import type { ChatMessage, ChatOptions, Conversation, Todo } from './chat';
import type { Project, ProjectType } from './project';
import type { Theme } from './theme';

// Dashboard API
export interface DashboardAPI {
  loadDashboards: () => Promise<Dashboard[]>;
  addConfigPath: (filePath: string) => Promise<void>;
  removeConfigPath: (filePath: string) => Promise<void>;
  getWatchedPaths: () => Promise<string[]>;
  updateWidgetDimensions: (dashboardId: string, widgetId: string, dimensions: Partial<WidgetDimensions>) => Promise<void>;
  updateWidget: (dashboardId: string, widgetId: string, config: Partial<WidgetConfig>) => Promise<void>;
  deleteWidget: (dashboardId: string, widgetId: string) => Promise<void>;
  regenerateWidget: (dashboardId: string, widgetId: string) => Promise<void>;
  regenerateAllWidgets: (dashboardId: string) => Promise<void>;
  updateLayoutSettings: (dashboardId: string, settings: LayoutSettings) => Promise<void>;
  onDashboardUpdate: (callback: (dashboards: Dashboard[]) => void) => void;
  onError: (callback: (error: string) => void) => void;
  loadDataFile: (filePath: string) => Promise<unknown>;
  loadTextFile: (filePath: string) => Promise<string>;
  writeCodeFile: (filePath: string, content: string) => Promise<{ success: boolean }>;
}

// Command API
export interface CommandOutput {
  widgetId: string;
  data: string;
  isError?: boolean;
}

export interface CommandExit {
  widgetId: string;
  code: number;
}

export interface CommandAPI {
  runCommand: (command: string, cwd?: string) => Promise<{ stdout: string; stderr: string; code: number }>;
  startStreaming: (widgetId: string, command: string, cwd?: string) => Promise<void>;
  stopStreaming: (widgetId: string) => Promise<void>;
  isRunning: (widgetId: string) => Promise<boolean>;
  onOutput: (callback: (data: CommandOutput) => void) => () => void;
  onExit: (callback: (data: CommandExit) => void) => () => void;
  onError: (callback: (data: CommandOutput) => void) => () => void;
  offOutput: (handler: (data: CommandOutput) => void) => void;
  offExit: (handler: (data: CommandExit) => void) => void;
  offError: (handler: (data: CommandOutput) => void) => void;
}

// Claude API
export interface QuestionOption {
  label: string;
  description: string;
}

export interface Question {
  question: string;
  header: string;
  options: QuestionOption[];
  multiSelect: boolean;
}

export interface UserQuestion {
  id: string;
  chatId: string;
  questions: Question[];
}

export interface ClaudeAPI {
  sendMessage: (chatId: string, message: string, options?: ChatOptions) => Promise<void>;
  interrupt: (chatId: string) => Promise<void>;
  getMessages: (chatId: string) => Promise<ChatMessage[]>;
  saveMessage: (chatId: string, message: ChatMessage) => Promise<void>;
  clearMessages: (chatId: string) => Promise<void>;
  listConversations: (widgetKey: string) => Promise<Conversation[]>;
  createConversation: (widgetKey: string, title: string) => Promise<Conversation>;
  updateConversation: (widgetKey: string, conversationId: string, updates: Partial<Conversation>) => Promise<void>;
  deleteConversation: (widgetKey: string, conversationId: string) => Promise<void>;
  getTodos: (conversationId: string) => Promise<Todo[]>;
  onMessage: (callback: (data: { chatId: string; message: ChatMessage }) => void) => () => void;
  onComplete: (callback: (data: { chatId: string }) => void) => () => void;
  onError: (callback: (data: { chatId: string; error: string }) => void) => () => void;
  onTodoUpdate: (callback: (data: { conversationId: string; todos: Todo[] }) => void) => () => void;
  onUserQuestion: (callback: (data: UserQuestion) => void) => () => void;
  sendQuestionAnswer: (questionId: string, answer: Record<string, string | string[]>) => void;
  offMessage: (handler: (data: { chatId: string; message: ChatMessage }) => void) => void;
  offComplete: (handler: (data: { chatId: string }) => void) => void;
  offError: (handler: (data: { chatId: string; error: string }) => void) => void;
  offTodoUpdate: (handler: (data: { conversationId: string; todos: Todo[] }) => void) => void;
  offUserQuestion: (handler: (data: UserQuestion) => void) => void;
  removeAllListeners: () => void;
}

// WebContentsView API
export interface WebViewBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface WebContentsViewAPI {
  create: (widgetId: string, url: string, bounds: WebViewBounds, backgroundColor?: string) => Promise<void>;
  navigate: (widgetId: string, url: string) => Promise<void>;
  goBack: (widgetId: string) => Promise<void>;
  goForward: (widgetId: string) => Promise<void>;
  reload: (widgetId: string) => Promise<void>;
  canNavigate: (widgetId: string) => Promise<{ canGoBack: boolean; canGoForward: boolean }>;
  updateBounds: (widgetId: string, bounds: WebViewBounds) => Promise<void>;
  destroy: (widgetId: string) => Promise<void>;
  onNavigated: (callback: (data: { widgetId: string; url: string }) => void) => () => void;
  offNavigated: (handler: (data: { widgetId: string; url: string }) => void) => void;
}

// Project API
export interface ProjectAPI {
  listProjects: () => Promise<Project[]>;
  getProject: (projectId: string) => Promise<Project | null>;
  addProject: (projectPath: string, type: ProjectType, name: string, description?: string) => Promise<Project>;
  removeProject: (projectId: string) => Promise<void>;
  initProject: (projectPath: string, name: string, description?: string) => Promise<Project>;
  openProjectFolder: (projectId: string) => Promise<void>;
  selectFolder: () => Promise<string | null>;
  generateDesign: (projectName: string) => Promise<{ icon: string; theme: Partial<Theme> }>;
  updateProject: (projectId: string, updates: Partial<Project>) => Promise<void>;
}
