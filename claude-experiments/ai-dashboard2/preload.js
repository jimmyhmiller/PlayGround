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
  }
});
