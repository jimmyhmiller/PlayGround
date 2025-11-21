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

  // Listen for dashboard updates when files change
  onDashboardUpdate: (callback) => {
    ipcRenderer.on('dashboard-updated', (event, dashboards) => callback(dashboards));
  },

  // Listen for errors
  onError: (callback) => {
    ipcRenderer.on('dashboard-error', (event, error) => callback(error));
  },
});

contextBridge.exposeInMainWorld('claudeAPI', {
  // Send a message to Claude and receive streaming responses
  sendMessage: (chatId, message, options) =>
    ipcRenderer.invoke('claude-chat-send', { chatId, message, options }),

  // Interrupt an ongoing chat
  interrupt: (chatId) =>
    ipcRenderer.invoke('claude-chat-interrupt', { chatId }),

  // Listen for streaming message chunks
  onMessage: (callback) => {
    ipcRenderer.on('claude-chat-message', (event, data) => callback(data));
  },

  // Listen for completion
  onComplete: (callback) => {
    ipcRenderer.on('claude-chat-complete', (event, data) => callback(data));
  },

  // Listen for errors
  onError: (callback) => {
    ipcRenderer.on('claude-chat-error', (event, data) => callback(data));
  },

  // Remove all listeners
  removeAllListeners: () => {
    ipcRenderer.removeAllListeners('claude-chat-message');
    ipcRenderer.removeAllListeners('claude-chat-complete');
    ipcRenderer.removeAllListeners('claude-chat-error');
  }
});
