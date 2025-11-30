import { contextBridge, ipcRenderer, IpcRendererEvent } from 'electron';
import type {
  DashboardAPI,
  CommandAPI,
  ClaudeAPI,
  WebContentsViewAPI,
  ProjectAPI,
  Dashboard,
  WidgetConfig,
  WidgetDimensions,
  LayoutSettings,
  ChatMessage,
  ChatOptions,
  Conversation,
  Todo,
  Project,
  ProjectType,
  CommandOutput,
  CommandExit,
  UserQuestion,
  WebViewBounds
} from './src/types';

const dashboardAPI: DashboardAPI = {
  loadDashboards: () => ipcRenderer.invoke('load-dashboards'),
  addConfigPath: (filePath: string) => ipcRenderer.invoke('add-config-path', filePath),
  removeConfigPath: (filePath: string) => ipcRenderer.invoke('remove-config-path', filePath),
  getWatchedPaths: () => ipcRenderer.invoke('get-watched-paths'),
  updateWidgetDimensions: (dashboardId: string, widgetId: string, dimensions: WidgetDimensions) =>
    ipcRenderer.invoke('update-widget-dimensions', { dashboardId, widgetId, dimensions }),
  updateWidget: (dashboardId: string, widgetId: string, config: Partial<WidgetConfig>) =>
    ipcRenderer.invoke('update-widget', { dashboardId, widgetId, config }),
  deleteWidget: (dashboardId: string, widgetId: string) =>
    ipcRenderer.invoke('delete-widget', { dashboardId, widgetId }),
  regenerateWidget: (dashboardId: string, widgetId: string) =>
    ipcRenderer.invoke('regenerate-widget', { dashboardId, widgetId }),
  regenerateAllWidgets: (dashboardId: string) =>
    ipcRenderer.invoke('regenerate-all-widgets', { dashboardId }),
  updateLayoutSettings: (dashboardId: string, settings: LayoutSettings) =>
    ipcRenderer.invoke('update-layout-settings', { dashboardId, settings }),
  onDashboardUpdate: (callback: (dashboards: Dashboard[]) => void) => {
    ipcRenderer.on('dashboard-updated', (_event: IpcRendererEvent, dashboards: Dashboard[]) => callback(dashboards));
  },
  onError: (callback: (error: string) => void) => {
    ipcRenderer.on('dashboard-error', (_event: IpcRendererEvent, error: string) => callback(error));
  },
  loadDataFile: async (filePath: string) => {
    const result = await ipcRenderer.invoke('load-data-file', filePath);
    if (!result.success) {
      throw new Error(result.error);
    }
    return result.data;
  },
  loadTextFile: async (filePath: string) => {
    const result = await ipcRenderer.invoke('load-text-file', filePath);
    if (!result.success) {
      throw new Error(result.error);
    }
    return result.content;
  },
  writeCodeFile: async (filePath: string, content: string) => {
    const result = await ipcRenderer.invoke('write-code-file', { filePath, content });
    if (!result.success) {
      throw new Error(result.error);
    }
    return result;
  },
};

const commandAPI: CommandAPI = {
  runCommand: (command: string, cwd?: string) => ipcRenderer.invoke('run-command', { command, cwd }),
  startStreaming: (widgetId: string, command: string, cwd?: string) =>
    ipcRenderer.invoke('start-streaming-command', { widgetId, command, cwd }),
  stopStreaming: (widgetId: string) =>
    ipcRenderer.invoke('stop-streaming-command', { widgetId }),
  isRunning: (widgetId: string) =>
    ipcRenderer.invoke('is-command-running', { widgetId }),
  onOutput: (callback: (data: CommandOutput) => void) => {
    const handler = (_event: IpcRendererEvent, data: CommandOutput) => callback(data);
    ipcRenderer.on('command-output', handler);
    return () => ipcRenderer.off('command-output', handler);
  },
  onExit: (callback: (data: CommandExit) => void) => {
    const handler = (_event: IpcRendererEvent, data: CommandExit) => callback(data);
    ipcRenderer.on('command-exit', handler);
    return () => ipcRenderer.off('command-exit', handler);
  },
  onError: (callback: (data: CommandOutput) => void) => {
    const handler = (_event: IpcRendererEvent, data: CommandOutput) => callback(data);
    ipcRenderer.on('command-error', handler);
    return () => ipcRenderer.off('command-error', handler);
  },
  offOutput: (handler: any) => ipcRenderer.off('command-output', handler),
  offExit: (handler: any) => ipcRenderer.off('command-exit', handler),
  offError: (handler: any) => ipcRenderer.off('command-error', handler),
};

const claudeAPI: ClaudeAPI = {
  sendMessage: (chatId: string, message: string, options?: ChatOptions) =>
    ipcRenderer.invoke('claude-chat-send', { chatId, message, options }),
  interrupt: (chatId: string) =>
    ipcRenderer.invoke('claude-chat-interrupt', { chatId }),
  getMessages: (chatId: string) =>
    ipcRenderer.invoke('chat-get-messages', { chatId }),
  saveMessage: (chatId: string, message: ChatMessage) =>
    ipcRenderer.invoke('chat-save-message', { chatId, message }),
  clearMessages: (chatId: string) =>
    ipcRenderer.invoke('chat-clear-messages', { chatId }),
  listConversations: (widgetKey: string) =>
    ipcRenderer.invoke('conversation-list', { widgetKey }),
  createConversation: (widgetKey: string, title: string) =>
    ipcRenderer.invoke('conversation-create', { widgetKey, title }),
  updateConversation: (widgetKey: string, conversationId: string, updates: Partial<Conversation>) =>
    ipcRenderer.invoke('conversation-update', { widgetKey, conversationId, updates }),
  deleteConversation: (widgetKey: string, conversationId: string) =>
    ipcRenderer.invoke('conversation-delete', { widgetKey, conversationId }),
  getTodos: (conversationId: string) =>
    ipcRenderer.invoke('get-conversation-todos', { conversationId }),
  onMessage: (callback: (data: { chatId: string; message: ChatMessage }) => void) => {
    const handler = (_event: IpcRendererEvent, data: { chatId: string; message: ChatMessage }) => callback(data);
    ipcRenderer.on('claude-chat-message', handler);
    return () => ipcRenderer.off('claude-chat-message', handler);
  },
  onComplete: (callback: (data: { chatId: string }) => void) => {
    const handler = (_event: IpcRendererEvent, data: { chatId: string }) => callback(data);
    ipcRenderer.on('claude-chat-complete', handler);
    return () => ipcRenderer.off('claude-chat-complete', handler);
  },
  onError: (callback: (data: { chatId: string; error: string }) => void) => {
    const handler = (_event: IpcRendererEvent, data: { chatId: string; error: string }) => callback(data);
    ipcRenderer.on('claude-chat-error', handler);
    return () => ipcRenderer.off('claude-chat-error', handler);
  },
  onTodoUpdate: (callback: (data: { conversationId: string; todos: Todo[] }) => void) => {
    const handler = (_event: IpcRendererEvent, data: { conversationId: string; todos: Todo[] }) => callback(data);
    ipcRenderer.on('todo-update', handler);
    return () => ipcRenderer.off('todo-update', handler);
  },
  onUserQuestion: (callback: (data: UserQuestion) => void) => {
    const handler = (_event: IpcRendererEvent, data: UserQuestion) => callback(data);
    ipcRenderer.on('ask-user-question', handler);
    return () => ipcRenderer.off('ask-user-question', handler);
  },
  sendQuestionAnswer: (questionId: string, answer: Record<string, string | string[]>) => {
    ipcRenderer.send(`question-answer-${questionId}`, answer);
  },
  offMessage: (handler: any) => {
    ipcRenderer.off('claude-chat-message', handler);
  },
  offComplete: (handler: any) => {
    ipcRenderer.off('claude-chat-complete', handler);
  },
  offError: (handler: any) => {
    ipcRenderer.off('claude-chat-error', handler);
  },
  offTodoUpdate: (handler: any) => {
    ipcRenderer.off('todo-update', handler);
  },
  offUserQuestion: (handler: any) => {
    ipcRenderer.off('ask-user-question', handler);
  },
  removeAllListeners: () => {
    ipcRenderer.removeAllListeners('claude-chat-message');
    ipcRenderer.removeAllListeners('claude-chat-complete');
    ipcRenderer.removeAllListeners('claude-chat-error');
    ipcRenderer.removeAllListeners('todo-update');
  },
};

const webContentsViewAPI: WebContentsViewAPI = {
  create: (widgetId: string, url: string, bounds: WebViewBounds, backgroundColor?: string) =>
    ipcRenderer.invoke('create-webcontentsview', { widgetId, url, bounds, backgroundColor }),
  navigate: (widgetId: string, url: string) =>
    ipcRenderer.invoke('navigate-webcontentsview', { widgetId, url }),
  goBack: (widgetId: string) =>
    ipcRenderer.invoke('webcontentsview-go-back', { widgetId }),
  goForward: (widgetId: string) =>
    ipcRenderer.invoke('webcontentsview-go-forward', { widgetId }),
  reload: (widgetId: string) =>
    ipcRenderer.invoke('webcontentsview-reload', { widgetId }),
  canNavigate: (widgetId: string) =>
    ipcRenderer.invoke('webcontentsview-can-navigate', { widgetId }),
  updateBounds: (widgetId: string, bounds: WebViewBounds) =>
    ipcRenderer.invoke('update-webcontentsview-bounds', { widgetId, bounds }),
  destroy: (widgetId: string) =>
    ipcRenderer.invoke('destroy-webcontentsview', { widgetId }),
  onNavigated: (callback: (data: { widgetId: string; url: string }) => void) => {
    const handler = (_event: IpcRendererEvent, data: { widgetId: string; url: string }) => callback(data);
    ipcRenderer.on('webcontentsview-navigated', handler);
    return () => ipcRenderer.off('webcontentsview-navigated', handler);
  },
  offNavigated: (handler: any) =>
    ipcRenderer.off('webcontentsview-navigated', handler),
};

const projectAPI: ProjectAPI = {
  listProjects: () => ipcRenderer.invoke('project-list'),
  getProject: (projectId: string) => ipcRenderer.invoke('project-get', { projectId }),
  addProject: (projectPath: string, type: ProjectType, name: string, description?: string) =>
    ipcRenderer.invoke('project-add', { projectPath, type, name, description }),
  removeProject: (projectId: string) =>
    ipcRenderer.invoke('project-remove', { projectId }),
  initProject: (projectPath: string, name: string, description?: string) =>
    ipcRenderer.invoke('project-init', { projectPath, name, description }),
  openProjectFolder: (projectId: string) =>
    ipcRenderer.invoke('project-open-folder', { projectId }),
  selectFolder: () =>
    ipcRenderer.invoke('project-select-folder'),
  generateDesign: (projectName: string) =>
    ipcRenderer.invoke('project-generate-design', { projectName }),
  updateProject: (projectId: string, updates: Partial<Project>) =>
    ipcRenderer.invoke('project-update', { projectId, updates }),
};

contextBridge.exposeInMainWorld('dashboardAPI', dashboardAPI);
contextBridge.exposeInMainWorld('commandAPI', commandAPI);
contextBridge.exposeInMainWorld('claudeAPI', claudeAPI);
contextBridge.exposeInMainWorld('webContentsViewAPI', webContentsViewAPI);
contextBridge.exposeInMainWorld('projectAPI', projectAPI);
