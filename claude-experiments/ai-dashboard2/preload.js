"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
const dashboardAPI = {
    loadDashboards: () => electron_1.ipcRenderer.invoke('load-dashboards'),
    addConfigPath: (filePath) => electron_1.ipcRenderer.invoke('add-config-path', filePath),
    removeConfigPath: (filePath) => electron_1.ipcRenderer.invoke('remove-config-path', filePath),
    getWatchedPaths: () => electron_1.ipcRenderer.invoke('get-watched-paths'),
    updateWidgetDimensions: (dashboardId, widgetId, dimensions) => electron_1.ipcRenderer.invoke('update-widget-dimensions', { dashboardId, widgetId, dimensions }),
    updateWidget: (dashboardId, widgetId, config) => electron_1.ipcRenderer.invoke('update-widget', { dashboardId, widgetId, config }),
    deleteWidget: (dashboardId, widgetId) => electron_1.ipcRenderer.invoke('delete-widget', { dashboardId, widgetId }),
    regenerateWidget: (dashboardId, widgetId) => electron_1.ipcRenderer.invoke('regenerate-widget', { dashboardId, widgetId }),
    regenerateAllWidgets: (dashboardId) => electron_1.ipcRenderer.invoke('regenerate-all-widgets', { dashboardId }),
    updateLayoutSettings: (dashboardId, settings) => electron_1.ipcRenderer.invoke('update-layout-settings', { dashboardId, settings }),
    onDashboardUpdate: (callback) => {
        electron_1.ipcRenderer.on('dashboard-updated', (_event, dashboards) => callback(dashboards));
    },
    onError: (callback) => {
        electron_1.ipcRenderer.on('dashboard-error', (_event, error) => callback(error));
    },
    loadDataFile: async (filePath) => {
        const result = await electron_1.ipcRenderer.invoke('load-data-file', filePath);
        if (!result.success) {
            throw new Error(result.error);
        }
        return result.data;
    },
    loadTextFile: async (filePath) => {
        const result = await electron_1.ipcRenderer.invoke('load-text-file', filePath);
        if (!result.success) {
            throw new Error(result.error);
        }
        return result.content;
    },
    writeCodeFile: async (filePath, content) => {
        const result = await electron_1.ipcRenderer.invoke('write-code-file', { filePath, content });
        if (!result.success) {
            throw new Error(result.error);
        }
        return result;
    },
};
const commandAPI = {
    runCommand: (command, cwd) => electron_1.ipcRenderer.invoke('run-command', { command, cwd }),
    startStreaming: (widgetId, command, cwd) => electron_1.ipcRenderer.invoke('start-streaming-command', { widgetId, command, cwd }),
    stopStreaming: (widgetId) => electron_1.ipcRenderer.invoke('stop-streaming-command', { widgetId }),
    isRunning: (widgetId) => electron_1.ipcRenderer.invoke('is-command-running', { widgetId }),
    onOutput: (callback) => {
        const handler = (_event, data) => callback(data);
        electron_1.ipcRenderer.on('command-output', handler);
        return () => electron_1.ipcRenderer.off('command-output', handler);
    },
    onExit: (callback) => {
        const handler = (_event, data) => callback(data);
        electron_1.ipcRenderer.on('command-exit', handler);
        return () => electron_1.ipcRenderer.off('command-exit', handler);
    },
    onError: (callback) => {
        const handler = (_event, data) => callback(data);
        electron_1.ipcRenderer.on('command-error', handler);
        return () => electron_1.ipcRenderer.off('command-error', handler);
    },
    offOutput: (handler) => electron_1.ipcRenderer.off('command-output', handler),
    offExit: (handler) => electron_1.ipcRenderer.off('command-exit', handler),
    offError: (handler) => electron_1.ipcRenderer.off('command-error', handler),
};
const claudeAPI = {
    sendMessage: (chatId, message, options) => electron_1.ipcRenderer.invoke('claude-chat-send', { chatId, message, options }),
    interrupt: (chatId) => electron_1.ipcRenderer.invoke('claude-chat-interrupt', { chatId }),
    getMessages: (chatId) => electron_1.ipcRenderer.invoke('chat-get-messages', { chatId }),
    saveMessage: (chatId, message) => electron_1.ipcRenderer.invoke('chat-save-message', { chatId, message }),
    clearMessages: (chatId) => electron_1.ipcRenderer.invoke('chat-clear-messages', { chatId }),
    listConversations: (widgetKey) => electron_1.ipcRenderer.invoke('conversation-list', { widgetKey }),
    createConversation: (widgetKey, title) => electron_1.ipcRenderer.invoke('conversation-create', { widgetKey, title }),
    updateConversation: (widgetKey, conversationId, updates) => electron_1.ipcRenderer.invoke('conversation-update', { widgetKey, conversationId, updates }),
    deleteConversation: (widgetKey, conversationId) => electron_1.ipcRenderer.invoke('conversation-delete', { widgetKey, conversationId }),
    getTodos: (conversationId) => electron_1.ipcRenderer.invoke('get-conversation-todos', { conversationId }),
    onMessage: (callback) => {
        const handler = (_event, data) => callback(data);
        electron_1.ipcRenderer.on('claude-chat-message', handler);
        return () => electron_1.ipcRenderer.off('claude-chat-message', handler);
    },
    onComplete: (callback) => {
        const handler = (_event, data) => callback(data);
        electron_1.ipcRenderer.on('claude-chat-complete', handler);
        return () => electron_1.ipcRenderer.off('claude-chat-complete', handler);
    },
    onError: (callback) => {
        const handler = (_event, data) => callback(data);
        electron_1.ipcRenderer.on('claude-chat-error', handler);
        return () => electron_1.ipcRenderer.off('claude-chat-error', handler);
    },
    onTodoUpdate: (callback) => {
        const handler = (_event, data) => callback(data);
        electron_1.ipcRenderer.on('todo-update', handler);
        return () => electron_1.ipcRenderer.off('todo-update', handler);
    },
    onUserQuestion: (callback) => {
        const handler = (_event, data) => callback(data);
        electron_1.ipcRenderer.on('ask-user-question', handler);
        return () => electron_1.ipcRenderer.off('ask-user-question', handler);
    },
    sendQuestionAnswer: (questionId, answer) => {
        electron_1.ipcRenderer.send(`question-answer-${questionId}`, answer);
    },
    offMessage: (handler) => {
        electron_1.ipcRenderer.off('claude-chat-message', handler);
    },
    offComplete: (handler) => {
        electron_1.ipcRenderer.off('claude-chat-complete', handler);
    },
    offError: (handler) => {
        electron_1.ipcRenderer.off('claude-chat-error', handler);
    },
    offTodoUpdate: (handler) => {
        electron_1.ipcRenderer.off('todo-update', handler);
    },
    offUserQuestion: (handler) => {
        electron_1.ipcRenderer.off('ask-user-question', handler);
    },
    removeAllListeners: () => {
        electron_1.ipcRenderer.removeAllListeners('claude-chat-message');
        electron_1.ipcRenderer.removeAllListeners('claude-chat-complete');
        electron_1.ipcRenderer.removeAllListeners('claude-chat-error');
        electron_1.ipcRenderer.removeAllListeners('todo-update');
    },
};
const webContentsViewAPI = {
    create: (widgetId, url, bounds, backgroundColor) => electron_1.ipcRenderer.invoke('create-webcontentsview', { widgetId, url, bounds, backgroundColor }),
    navigate: (widgetId, url) => electron_1.ipcRenderer.invoke('navigate-webcontentsview', { widgetId, url }),
    goBack: (widgetId) => electron_1.ipcRenderer.invoke('webcontentsview-go-back', { widgetId }),
    goForward: (widgetId) => electron_1.ipcRenderer.invoke('webcontentsview-go-forward', { widgetId }),
    reload: (widgetId) => electron_1.ipcRenderer.invoke('webcontentsview-reload', { widgetId }),
    canNavigate: (widgetId) => electron_1.ipcRenderer.invoke('webcontentsview-can-navigate', { widgetId }),
    updateBounds: (widgetId, bounds) => electron_1.ipcRenderer.invoke('update-webcontentsview-bounds', { widgetId, bounds }),
    destroy: (widgetId) => electron_1.ipcRenderer.invoke('destroy-webcontentsview', { widgetId }),
    onNavigated: (callback) => {
        const handler = (_event, data) => callback(data);
        electron_1.ipcRenderer.on('webcontentsview-navigated', handler);
        return () => electron_1.ipcRenderer.off('webcontentsview-navigated', handler);
    },
    offNavigated: (handler) => electron_1.ipcRenderer.off('webcontentsview-navigated', handler),
};
const projectAPI = {
    listProjects: () => electron_1.ipcRenderer.invoke('project-list'),
    getProject: (projectId) => electron_1.ipcRenderer.invoke('project-get', { projectId }),
    addProject: (projectPath, type, name, description) => electron_1.ipcRenderer.invoke('project-add', { projectPath, type, name, description }),
    removeProject: (projectId) => electron_1.ipcRenderer.invoke('project-remove', { projectId }),
    initProject: (projectPath, name, description) => electron_1.ipcRenderer.invoke('project-init', { projectPath, name, description }),
    openProjectFolder: (projectId) => electron_1.ipcRenderer.invoke('project-open-folder', { projectId }),
    selectFolder: () => electron_1.ipcRenderer.invoke('project-select-folder'),
    generateDesign: (projectName) => electron_1.ipcRenderer.invoke('project-generate-design', { projectName }),
    updateProject: (projectId, updates) => electron_1.ipcRenderer.invoke('project-update', { projectId, updates }),
};
electron_1.contextBridge.exposeInMainWorld('dashboardAPI', dashboardAPI);
electron_1.contextBridge.exposeInMainWorld('commandAPI', commandAPI);
electron_1.contextBridge.exposeInMainWorld('claudeAPI', claudeAPI);
electron_1.contextBridge.exposeInMainWorld('webContentsViewAPI', webContentsViewAPI);
electron_1.contextBridge.exposeInMainWorld('projectAPI', projectAPI);
