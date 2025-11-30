const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('dashboardAPI', {
  // Load dashboards from config paths
  loadDashboards: () => ipcRenderer.invoke('load-dashboards'),

  // Add a new config file path to watch
  addConfigPath: (filePath) => ipcRenderer.invoke('add-config-path', filePath),

  // Remove a config file path
  removeConfigPath: (filePath) => ipcRenderer.invoke('remove-config-path', filePath),

  // Get list of watched paths
  getWatchedPaths: () => ipcRenderer.invoke('get-watched-paths'),

  // Update widget dimensions in the JSON file
  updateWidgetDimensions: (dashboardId, widgetId, dimensions) =>
    ipcRenderer.invoke('update-widget-dimensions', { dashboardId, widgetId, dimensions }),

  // Update entire widget configuration
  updateWidget: (dashboardId, widgetId, config) =>
    ipcRenderer.invoke('update-widget', { dashboardId, widgetId, config }),

  // Delete a widget from the dashboard
  deleteWidget: (dashboardId, widgetId) =>
    ipcRenderer.invoke('delete-widget', { dashboardId, widgetId }),

  // Regenerate widget data by running its command
  regenerateWidget: (dashboardId, widgetId) =>
    ipcRenderer.invoke('regenerate-widget', { dashboardId, widgetId }),

  // Regenerate all widgets with commands in a dashboard
  regenerateAllWidgets: (dashboardId) =>
    ipcRenderer.invoke('regenerate-all-widgets', { dashboardId }),

  // Update layout settings (gap, buffer, etc.)
  updateLayoutSettings: (dashboardId, settings) =>
    ipcRenderer.invoke('update-layout-settings', { dashboardId, settings }),

  // Listen for dashboard updates when files change
  onDashboardUpdate: (callback) => {
    ipcRenderer.on('dashboard-updated', (event, dashboards) => callback(dashboards));
  },

  // Listen for errors
  onError: (callback) => {
    ipcRenderer.on('dashboard-error', (event, error) => callback(error));
  },

  // Load data from a file (for widget data sources)
  loadDataFile: async (filePath) => {
    const result = await ipcRenderer.invoke('load-data-file', filePath);
    if (!result.success) {
      throw new Error(result.error);
    }
    return result.data;
  },

  // Load text from a file (for markdown, code, etc.)
  loadTextFile: async (filePath) => {
    const result = await ipcRenderer.invoke('load-text-file', filePath);
    if (!result.success) {
      throw new Error(result.error);
    }
    return result.content;
  },

  // Write code to a temp file (for code editor widget)
  writeCodeFile: async (filePath, content) => {
    const result = await ipcRenderer.invoke('write-code-file', { filePath, content });
    if (!result.success) {
      throw new Error(result.error);
    }
    return result;
  },
});

contextBridge.exposeInMainWorld('commandAPI', {
  // Run a command and get the output (waits for completion)
  runCommand: (command, cwd) => ipcRenderer.invoke('run-command', { command, cwd }),

  // Start a long-running command with streaming output
  startStreaming: (widgetId, command, cwd) =>
    ipcRenderer.invoke('start-streaming-command', { widgetId, command, cwd }),

  // Stop a running command
  stopStreaming: (widgetId) =>
    ipcRenderer.invoke('stop-streaming-command', { widgetId }),

  // Check if command is running
  isRunning: (widgetId) =>
    ipcRenderer.invoke('is-command-running', { widgetId }),

  // Listen for command output
  onOutput: (callback) => {
    const handler = (event, data) => callback(data);
    ipcRenderer.on('command-output', handler);
    return handler;
  },

  // Listen for command exit
  onExit: (callback) => {
    const handler = (event, data) => callback(data);
    ipcRenderer.on('command-exit', handler);
    return handler;
  },

  // Listen for command errors
  onError: (callback) => {
    const handler = (event, data) => callback(data);
    ipcRenderer.on('command-error', handler);
    return handler;
  },

  // Remove listeners
  offOutput: (handler) => ipcRenderer.off('command-output', handler),
  offExit: (handler) => ipcRenderer.off('command-exit', handler),
  offError: (handler) => ipcRenderer.off('command-error', handler),
});

contextBridge.exposeInMainWorld('claudeAPI', {
  // Send a message to Claude and receive streaming responses
  sendMessage: (chatId, message, options) =>
    ipcRenderer.invoke('claude-chat-send', { chatId, message, options }),

  // Interrupt an ongoing chat
  interrupt: (chatId) =>
    ipcRenderer.invoke('claude-chat-interrupt', { chatId }),

  // Get messages for a chat
  getMessages: (chatId) =>
    ipcRenderer.invoke('chat-get-messages', { chatId }),

  // Save a message to chat history
  saveMessage: (chatId, message) =>
    ipcRenderer.invoke('chat-save-message', { chatId, message }),

  // Clear chat history
  clearMessages: (chatId) =>
    ipcRenderer.invoke('chat-clear-messages', { chatId }),

  // Conversation management
  listConversations: (widgetKey) =>
    ipcRenderer.invoke('conversation-list', { widgetKey }),

  createConversation: (widgetKey, title) =>
    ipcRenderer.invoke('conversation-create', { widgetKey, title }),

  updateConversation: (widgetKey, conversationId, updates) =>
    ipcRenderer.invoke('conversation-update', { widgetKey, conversationId, updates }),

  deleteConversation: (widgetKey, conversationId) =>
    ipcRenderer.invoke('conversation-delete', { widgetKey, conversationId }),

  // Listen for streaming message chunks
  onMessage: (callback) => {
    const handler = (event, data) => callback(data);
    ipcRenderer.on('claude-chat-message', handler);
    return handler;
  },

  // Listen for completion
  onComplete: (callback) => {
    const handler = (event, data) => callback(data);
    ipcRenderer.on('claude-chat-complete', handler);
    return handler;
  },

  // Listen for errors
  onError: (callback) => {
    const handler = (event, data) => callback(data);
    ipcRenderer.on('claude-chat-error', handler);
    return handler;
  },

  // Remove specific listeners
  offMessage: (handler) => {
    ipcRenderer.off('claude-chat-message', handler);
  },

  offComplete: (handler) => {
    ipcRenderer.off('claude-chat-complete', handler);
  },

  offError: (handler) => {
    ipcRenderer.off('claude-chat-error', handler);
  },

  // Get todos for a conversation
  getTodos: (conversationId) =>
    ipcRenderer.invoke('get-conversation-todos', { conversationId }),

  // Listen for todo updates
  onTodoUpdate: (callback) => {
    const handler = (event, data) => callback(data);
    ipcRenderer.on('todo-update', handler);
    return handler;
  },

  // Remove todo update listener
  offTodoUpdate: (handler) => {
    ipcRenderer.off('todo-update', handler);
  },

  // Remove all listeners (kept for backward compatibility)
  removeAllListeners: () => {
    ipcRenderer.removeAllListeners('claude-chat-message');
    ipcRenderer.removeAllListeners('claude-chat-complete');
    ipcRenderer.removeAllListeners('claude-chat-error');
    ipcRenderer.removeAllListeners('todo-update');
  },

  // Listen for user questions (plan mode)
  onUserQuestion: (callback) => {
    const handler = (event, data) => callback(data);
    ipcRenderer.on('ask-user-question', handler);
    return handler;
  },

  // Send answer to a question
  sendQuestionAnswer: (questionId, answer) => {
    ipcRenderer.send(`question-answer-${questionId}`, answer);
  },

  // Remove question listener
  offUserQuestion: (handler) => {
    ipcRenderer.off('ask-user-question', handler);
  }
});

contextBridge.exposeInMainWorld('webContentsViewAPI', {
  // Create a WebContentsView for a widget
  create: (widgetId, url, bounds, backgroundColor) =>
    ipcRenderer.invoke('create-webcontentsview', { widgetId, url, bounds, backgroundColor }),

  // Navigate to a URL
  navigate: (widgetId, url) =>
    ipcRenderer.invoke('navigate-webcontentsview', { widgetId, url }),

  // Go back
  goBack: (widgetId) =>
    ipcRenderer.invoke('webcontentsview-go-back', { widgetId }),

  // Go forward
  goForward: (widgetId) =>
    ipcRenderer.invoke('webcontentsview-go-forward', { widgetId }),

  // Reload
  reload: (widgetId) =>
    ipcRenderer.invoke('webcontentsview-reload', { widgetId }),

  // Check if can navigate back/forward
  canNavigate: (widgetId) =>
    ipcRenderer.invoke('webcontentsview-can-navigate', { widgetId }),

  // Update bounds
  updateBounds: (widgetId, bounds) =>
    ipcRenderer.invoke('update-webcontentsview-bounds', { widgetId, bounds }),

  // Destroy view
  destroy: (widgetId) =>
    ipcRenderer.invoke('destroy-webcontentsview', { widgetId }),

  // Listen for navigation events
  onNavigated: (callback) => {
    const handler = (event, data) => callback(data);
    ipcRenderer.on('webcontentsview-navigated', handler);
    return handler;
  },

  // Remove listener
  offNavigated: (handler) =>
    ipcRenderer.off('webcontentsview-navigated', handler),
});

contextBridge.exposeInMainWorld('projectAPI', {
  // List all projects
  listProjects: () => ipcRenderer.invoke('project-list'),

  // Get a specific project by ID
  getProject: (projectId) => ipcRenderer.invoke('project-get', { projectId }),

  // Add a project from an existing folder
  addProject: (projectPath, type, name, description) =>
    ipcRenderer.invoke('project-add', { projectPath, type, name, description }),

  // Remove a project from the registry
  removeProject: (projectId) =>
    ipcRenderer.invoke('project-remove', { projectId }),

  // Initialize .ai-dashboard structure in an existing folder
  initProject: (projectPath, name, description) =>
    ipcRenderer.invoke('project-init', { projectPath, name, description }),

  // Open project folder in file explorer
  openProjectFolder: (projectId) =>
    ipcRenderer.invoke('project-open-folder', { projectId }),

  // Show folder picker dialog
  selectFolder: () =>
    ipcRenderer.invoke('project-select-folder'),

  // Generate AI-powered icon and theme for a project
  generateDesign: (projectName) =>
    ipcRenderer.invoke('project-generate-design', { projectName }),

  // Update a project's properties
  updateProject: (projectId, updates) =>
    ipcRenderer.invoke('project-update', { projectId, updates }),
});
