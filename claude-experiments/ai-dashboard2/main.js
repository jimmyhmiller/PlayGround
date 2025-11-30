"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const util = __importStar(require("util"));
const child_process_1 = require("child_process");
const claude_agent_sdk_1 = require("@anthropic-ai/claude-agent-sdk");
const dashboard_tools_1 = require("./dashboard-tools");
const projectUtils = __importStar(require("./project-utils"));
const isDev = process.env.NODE_ENV === 'development';
let mainWindow = null;
// Global state Maps with proper types
const watchedPaths = new Map();
const projects = new Map();
const chatMessages = new Map();
const sessionIds = new Map(); // chatId -> SDK session ID
const conversations = new Map();
const conversationTodos = new Map();
const webContentsViews = new Map();
const activeChats = new Map();
const runningCommands = new Map();
// File paths
const configFilePath = path.join(electron_1.app.getPath('userData'), 'dashboard-paths.json');
const projectsPath = path.join(electron_1.app.getPath('userData'), 'projects.json');
const chatStoragePath = path.join(electron_1.app.getPath('userData'), 'chat-history.json');
const conversationsPath = path.join(electron_1.app.getPath('userData'), 'conversations.json');
// Load saved config paths
function loadSavedPaths() {
    try {
        if (fs.existsSync(configFilePath)) {
            const data = JSON.parse(fs.readFileSync(configFilePath, 'utf-8'));
            return data.paths || [];
        }
    }
    catch (e) {
        console.error('Error loading saved paths:', e);
    }
    return [];
}
// Save config paths
function savePaths() {
    try {
        const paths = Array.from(watchedPaths.keys());
        fs.writeFileSync(configFilePath, JSON.stringify({ paths }, null, 2));
    }
    catch (e) {
        console.error('Error saving paths:', e);
    }
}
// Load chat history from disk
function loadChatHistory() {
    try {
        if (fs.existsSync(chatStoragePath)) {
            const data = JSON.parse(fs.readFileSync(chatStoragePath, 'utf-8'));
            Object.entries(data).forEach(([chatId, messages]) => {
                chatMessages.set(chatId, messages);
            });
            console.log(`[Chat] Loaded history for ${chatMessages.size} chats`);
        }
    }
    catch (e) {
        console.error('Error loading chat history:', e);
    }
}
// Save chat history to disk
function saveChatHistory() {
    try {
        const data = Object.fromEntries(chatMessages);
        fs.writeFileSync(chatStoragePath, JSON.stringify(data, null, 2));
        console.log(`[Chat] Saved history for ${chatMessages.size} chats`);
    }
    catch (e) {
        console.error('Error saving chat history:', e);
    }
}
// Load conversations metadata from disk
function loadConversations() {
    try {
        if (fs.existsSync(conversationsPath)) {
            const data = JSON.parse(fs.readFileSync(conversationsPath, 'utf-8'));
            Object.entries(data).forEach(([widgetId, convos]) => {
                conversations.set(widgetId, convos);
                convos.forEach(convo => {
                    if (convo.sessionId) {
                        sessionIds.set(convo.id, convo.sessionId);
                        console.log(`[Conversations] Loaded session ID for conversation ${convo.id}`);
                    }
                });
            });
            console.log(`[Conversations] Loaded ${conversations.size} widget conversation lists`);
            console.log(`[Conversations] Loaded ${sessionIds.size} session IDs`);
        }
    }
    catch (e) {
        console.error('Error loading conversations:', e);
    }
}
// Save conversations metadata to disk
function saveConversations() {
    try {
        const data = Object.fromEntries(conversations);
        fs.writeFileSync(conversationsPath, JSON.stringify(data, null, 2));
        console.log(`[Conversations] Saved ${conversations.size} widget conversation lists`);
    }
    catch (e) {
        console.error('Error saving conversations:', e);
    }
}
// Load projects from disk
function loadProjects() {
    try {
        if (fs.existsSync(projectsPath)) {
            const data = JSON.parse(fs.readFileSync(projectsPath, 'utf-8'));
            Object.entries(data).forEach(([projectId, project]) => {
                projects.set(projectId, project);
            });
            console.log(`[Projects] Loaded ${projects.size} projects`);
        }
    }
    catch (e) {
        console.error('Error loading projects:', e);
    }
}
// Save projects to disk
function saveProjects() {
    try {
        const data = Object.fromEntries(projects);
        fs.writeFileSync(projectsPath, JSON.stringify(data, null, 2));
        console.log(`[Projects] Saved ${projects.size} projects`);
    }
    catch (e) {
        console.error('Error saving projects:', e);
    }
}
// Parse a dashboard JSON file
function parseDashboardFile(filePath) {
    try {
        const content = fs.readFileSync(filePath, 'utf-8');
        const dashboard = JSON.parse(content);
        dashboard._sourcePath = filePath;
        return dashboard;
    }
    catch (e) {
        console.error(`Error parsing ${filePath}:`, e);
        return null;
    }
}
// Watch a file for changes
function watchFile(filePath) {
    if (watchedPaths.has(filePath))
        return;
    const dashboard = parseDashboardFile(filePath);
    if (!dashboard)
        return;
    const watcher = fs.watchFile(filePath, { interval: 100 }, (curr, prev) => {
        if (curr.mtime > prev.mtime) {
            const entry = watchedPaths.get(filePath);
            if (!entry)
                return;
            const timeSinceWrite = curr.mtime.getTime() - (entry.lastWriteTime || 0);
            if (timeSinceWrite < 1000 && entry.lastWriteTime) {
                console.log(`[FileWatcher] Ignoring self-triggered change: ${filePath} (${timeSinceWrite}ms since write)`);
                return;
            }
            console.log(`[FileWatcher] External file modification detected: ${filePath}`);
            const updated = parseDashboardFile(filePath);
            if (updated) {
                console.log(`[FileWatcher] Successfully reloaded ${filePath}`);
                entry.dashboard = updated;
                broadcastDashboards();
            }
            else {
                console.error(`[FileWatcher] Failed to parse ${filePath}`);
            }
        }
    });
    watchedPaths.set(filePath, { watcher, dashboard, lastWriteTime: null });
}
// Stop watching a file
function unwatchFile(filePath) {
    const entry = watchedPaths.get(filePath);
    if (entry) {
        fs.unwatchFile(filePath);
        watchedPaths.delete(filePath);
    }
}
// Get all loaded dashboards
function getAllDashboards() {
    return Array.from(watchedPaths.values())
        .map(entry => {
        if (!entry.dashboard)
            return null;
        const dashboard = { ...entry.dashboard };
        if (entry.projectId) {
            dashboard._projectId = entry.projectId;
            const project = projects.get(entry.projectId);
            if (project) {
                if (project.icon)
                    dashboard.icon = project.icon;
                if (project.theme)
                    dashboard.theme = project.theme;
                if (project.rootPath)
                    dashboard._projectRoot = project.rootPath;
            }
        }
        return dashboard;
    })
        .filter((d) => d !== null);
}
// Broadcast dashboards to renderer
function broadcastDashboards() {
    const dashboards = getAllDashboards();
    console.log(`[Broadcast] Sending ${dashboards.length} dashboards to renderer`);
    if (mainWindow) {
        mainWindow.webContents.send('dashboard-updated', dashboards);
    }
    else {
        console.warn('[Broadcast] No main window available');
    }
}
function createWindow() {
    mainWindow = new electron_1.BrowserWindow({
        width: 950,
        height: 700,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js'),
        },
        backgroundColor: '#050505',
        titleBarStyle: 'hiddenInset',
    });
    if (isDev) {
        mainWindow.loadURL('http://localhost:5173');
    }
    else {
        mainWindow.loadFile(path.join(__dirname, 'dist', 'index.html'));
    }
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}
// IPC Handlers
electron_1.ipcMain.handle('load-dashboards', () => {
    return getAllDashboards();
});
electron_1.ipcMain.handle('add-config-path', (_event, filePath) => {
    if (!fs.existsSync(filePath)) {
        return { success: false, error: 'File does not exist' };
    }
    watchFile(filePath);
    savePaths();
    broadcastDashboards();
    return { success: true };
});
electron_1.ipcMain.handle('remove-config-path', (_event, filePath) => {
    unwatchFile(filePath);
    savePaths();
    broadcastDashboards();
    return { success: true };
});
electron_1.ipcMain.handle('get-watched-paths', () => {
    return Array.from(watchedPaths.keys());
});
electron_1.ipcMain.handle('update-widget-dimensions', async (_event, { dashboardId, widgetId, dimensions }) => {
    try {
        let targetPath = null;
        let entry = null;
        for (const [path, e] of watchedPaths.entries()) {
            if (e.dashboard.id === dashboardId) {
                targetPath = path;
                entry = e;
                break;
            }
        }
        if (!targetPath || !entry) {
            return { success: false, error: 'Dashboard not found' };
        }
        const widget = entry.dashboard.widgets.find(w => w.id === widgetId);
        if (!widget) {
            return { success: false, error: 'Widget not found' };
        }
        const widgetAny = widget;
        if (dimensions.width !== undefined)
            widgetAny.width = dimensions.width;
        if (dimensions.height !== undefined)
            widgetAny.height = dimensions.height;
        if (dimensions.x !== undefined)
            widgetAny.x = dimensions.x;
        if (dimensions.y !== undefined)
            widgetAny.y = dimensions.y;
        entry.lastWriteTime = Date.now();
        fs.writeFileSync(targetPath, JSON.stringify(entry.dashboard, null, 2), 'utf-8');
        return { success: true };
    }
    catch (error) {
        console.error('Error updating widget dimensions:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('update-widget', async (_event, { dashboardId, widgetId, config }) => {
    try {
        let targetPath = null;
        let entry = null;
        for (const [path, e] of watchedPaths.entries()) {
            if (e.dashboard.id === dashboardId) {
                targetPath = path;
                entry = e;
                break;
            }
        }
        if (!targetPath || !entry) {
            return { success: false, error: 'Dashboard not found' };
        }
        const widgetIndex = entry.dashboard.widgets.findIndex(w => w.id === widgetId);
        if (widgetIndex === -1) {
            return { success: false, error: 'Widget not found' };
        }
        entry.dashboard.widgets[widgetIndex] = { ...config, id: widgetId };
        entry.lastWriteTime = Date.now();
        fs.writeFileSync(targetPath, JSON.stringify(entry.dashboard, null, 2), 'utf-8');
        electron_1.BrowserWindow.getAllWindows().forEach(win => {
            win.webContents.send('dashboard-updated', {
                dashboardId: entry.dashboard.id,
                dashboard: entry.dashboard
            });
        });
        return { success: true };
    }
    catch (error) {
        console.error('Error updating widget:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('delete-widget', async (_event, { dashboardId, widgetId }) => {
    try {
        let targetPath = null;
        let entry = null;
        for (const [path, e] of watchedPaths.entries()) {
            if (e.dashboard.id === dashboardId) {
                targetPath = path;
                entry = e;
                break;
            }
        }
        if (!targetPath || !entry) {
            return { success: false, error: 'Dashboard not found' };
        }
        entry.dashboard.widgets = entry.dashboard.widgets.filter(w => w.id !== widgetId);
        entry.lastWriteTime = Date.now();
        fs.writeFileSync(targetPath, JSON.stringify(entry.dashboard, null, 2), 'utf-8');
        broadcastDashboards();
        console.log(`[Dashboard] Deleted widget ${widgetId} from dashboard ${dashboardId}`);
        return { success: true };
    }
    catch (error) {
        console.error('Error deleting widget:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('update-layout-settings', async (_event, { dashboardId, settings }) => {
    try {
        let targetPath = null;
        let entry = null;
        for (const [path, e] of watchedPaths.entries()) {
            if (e.dashboard.id === dashboardId) {
                targetPath = path;
                entry = e;
                break;
            }
        }
        if (!targetPath || !entry) {
            return { success: false, error: 'Dashboard not found' };
        }
        if (!entry.dashboard.layout) {
            entry.dashboard.layout = {};
        }
        if (settings.widgetGap !== undefined)
            entry.dashboard.layout.widgetGap = settings.widgetGap;
        if (settings.buffer !== undefined)
            entry.dashboard.layout.buffer = settings.buffer;
        if (settings.mode !== undefined)
            entry.dashboard.layout.mode = settings.mode;
        entry.lastWriteTime = Date.now();
        fs.writeFileSync(targetPath, JSON.stringify(entry.dashboard, null, 2), 'utf-8');
        broadcastDashboards();
        return { success: true };
    }
    catch (error) {
        console.error('Error updating layout settings:', error);
        return { success: false, error: error.message };
    }
});
// Claude Agent SDK Chat Handlers
electron_1.ipcMain.handle('claude-chat-send', async (_event, { chatId, message, options = {} }) => {
    try {
        const responseId = `${chatId}-${Date.now()}`;
        (async () => {
            try {
                console.log(`[Claude] Starting chat for ${chatId} with message: ${message.substring(0, 50)}...`);
                const existingSessionId = sessionIds.get(chatId);
                const history = chatMessages.get(chatId) || [];
                console.log(`[Claude] ${existingSessionId ? 'Resuming' : 'Starting'} conversation ${chatId}`);
                if (existingSessionId) {
                    console.log(`[Claude] Using session ID: ${existingSessionId}`);
                }
                let systemPrompt = '';
                if (options.dashboardContext) {
                    const ctx = options.dashboardContext;
                    const widgetToolsList = ctx.config.widgets.map((w) => {
                        const label = Array.isArray(w.label) ? w.label.join(' - ') : w.label;
                        const tools = [];
                        if (w.type === 'stat') {
                            tools.push(`- update_stat_${w.id}(value) - Update "${label}" stat value`);
                        }
                        if (w.type === 'barChart') {
                            tools.push(`- update_chart_${w.id}(data) - Update "${label}" chart data`);
                        }
                        if (w.type === 'progress') {
                            tools.push(`- update_progress_${w.id}(value, text?) - Update "${label}" progress`);
                        }
                        if (w.type === 'todoList') {
                            tools.push(`- add_todo_${w.id}(text, done?) - Add todo to "${label}"`);
                            tools.push(`- toggle_todo_${w.id}(index) - Toggle todo in "${label}"`);
                        }
                        if (w.type === 'keyValue') {
                            tools.push(`- update_keyvalue_${w.id}(key, value) - Update key-value in "${label}"`);
                        }
                        return tools.join('\n');
                    }).filter((t) => t).join('\n');
                    systemPrompt = `You are assisting with a dashboard application. Here is the current dashboard context:

Dashboard File Path: ${ctx.filePath}
Dashboard ID: ${ctx.id}
Dashboard Title: ${ctx.title}

Dashboard Configuration:
\`\`\`json
${JSON.stringify(ctx.config, null, 2)}
\`\`\`

You can help edit this dashboard by providing instructions or generating updated JSON configurations.

## Available Widget Types

When adding widgets, these types are available:

1. **stat** - Display a single statistic
   - Config: { label: "Label", value: "Value" }

2. **barChart** - Animated bar chart
   - Config: { label: "Label" or ["Label", "Sublabel"], data: [numbers] or dataSource: "file.json" }

3. **progress** - Progress bar
   - Config: { label: "Label", value: 0-100, text: "Optional text" }

4. **chat** - AI chat interface with Claude
   - Config: { label: "Label", backend: "claude", claudeOptions: {model: "claude-sonnet-4-5-20250929"} }

5. **claudeTodos** - Display AI agent tasks
   - Config: { label: "Agent Tasks", chatWidgetId: "chat-widget-id" }

6. **todoList** - Simple todo list
   - Config: { label: "Label", items: [{text: "Task", done: false}] }

7. **fileList** - Display file list with status
   - Config: { label: "Label", files: [{name: "file.txt", status: "created"}] }

8. **keyValue** - Key-value pairs
   - Config: { label: "Label", items: [{key: "Key", value: "Value"}] }

9. **diffList** - Git-style diff view
   - Config: { label: "Label", items: [["file.txt", 10, 5]] }

10. **layoutSettings** - Grid layout controls
    - Config: { label: "Layout Settings" }

11. **commandRunner** - Run shell commands and display output
    - Config: { label: "Label", command: "npm test", cwd: "/path/to/dir", autoRun: true/false, showOutput: true/false }

12. **webView** - Embedded web browser
    - Config: { label: "Label", url: "https://example.com" }

## Available Custom Tools

You have access to specialized tools for this dashboard that allow you to quickly modify widgets without using the Edit tool:

### Widget-Specific Tools:
${widgetToolsList}

### General Tools:
- update_widget(widgetId, property, value) - Update any widget property
- add_widget(type, id, x, y, width, height, config?) - Add a new widget (use widget types above)
- remove_widget(widgetId) - Remove a widget

**Usage**: When the user asks to update a widget value (like "set uptime to 99%"), prefer using the widget-specific tool (e.g., update_stat_uptime("99%")) instead of manually editing the JSON file. This is faster and more reliable.`;
                }
                const { dashboardContext, ...queryOptions } = options;
                let mcpServers = queryOptions.mcpServers || {};
                if (dashboardContext) {
                    try {
                        const dashboardMcpServer = (0, dashboard_tools_1.createDashboardTools)({
                            config: dashboardContext.config,
                            filePath: dashboardContext.filePath
                        }, watchedPaths, broadcastDashboards, queryOptions.permissionMode || 'bypassPermissions');
                        mcpServers[`dashboard-${dashboardContext.id}`] = dashboardMcpServer;
                        console.log(`[Claude] Created ${Object.keys(mcpServers).length} MCP tools for dashboard ${dashboardContext.id} (mode: ${queryOptions.permissionMode || 'bypassPermissions'})`);
                    }
                    catch (error) {
                        console.error('[Claude] Error creating dashboard tools:', error);
                    }
                }
                let workingDirectory = queryOptions.cwd || process.cwd();
                if (dashboardContext?.filePath) {
                    const dashboardEntry = watchedPaths.get(dashboardContext.filePath);
                    if (dashboardEntry?.projectId) {
                        const project = projects.get(dashboardEntry.projectId);
                        if (project?.rootPath) {
                            workingDirectory = project.rootPath;
                            console.log(`[Claude] Using project root as cwd: ${workingDirectory}`);
                        }
                    }
                }
                const result = (0, claude_agent_sdk_1.query)({
                    prompt: message,
                    options: {
                        model: queryOptions.model || 'claude-sonnet-4-5-20250929',
                        cwd: workingDirectory,
                        systemPrompt: systemPrompt || undefined,
                        resume: existingSessionId || undefined,
                        mcpServers,
                        permissionMode: queryOptions.permissionMode || 'bypassPermissions',
                        canUseTool: async (toolName, input, options) => {
                            console.log('[Claude] canUseTool called for:', toolName);
                            if (toolName === 'AskUserQuestion' || toolName.endsWith('__AskUserQuestion')) {
                                console.log('[Claude] Auto-approving AskUserQuestion tool');
                                return { behavior: 'allow', updatedInput: input };
                            }
                            return undefined;
                        },
                        ...queryOptions
                    }
                });
                activeChats.set(chatId, { query: result });
                for await (const msg of result) {
                    console.log(`[Claude] Received message type: ${msg.type}`);
                    console.log(util.inspect(msg, { depth: null, colors: true }));
                    if (msg.type === 'system' && msg.subtype === 'init' && msg.session_id) {
                        console.log(`[Claude] Captured session ID: ${msg.session_id}`);
                        sessionIds.set(chatId, msg.session_id);
                        for (const [widgetKey, convos] of conversations.entries()) {
                            const convo = convos.find(c => c.id === chatId);
                            if (convo) {
                                convo.sessionId = msg.session_id;
                                saveConversations();
                                console.log(`[Claude] Stored session ID in conversation metadata for ${chatId}`);
                                break;
                            }
                        }
                    }
                    if (msg.type === 'assistant' && msg.message?.content) {
                        const content = msg.message.content;
                        console.log(`[Claude] Assistant message for ${chatId}, content type:`, typeof content, 'isArray:', Array.isArray(content));
                        if (Array.isArray(content)) {
                            console.log(`[Claude] Content blocks: ${content.length}`);
                            content.forEach((block, idx) => {
                                console.log(`[Claude] Block ${idx}: type=${block.type}, name=${block.name}`);
                                if (block.type === 'tool_use' && block.name === 'TodoWrite' && block.input?.todos) {
                                    console.log(`[Claude] ✓ Received TodoWrite for ${chatId}:`, block.input.todos);
                                    conversationTodos.set(chatId, block.input.todos);
                                    mainWindow?.webContents.send('todo-update', {
                                        chatId,
                                        todos: block.input.todos
                                    });
                                }
                            });
                        }
                    }
                    mainWindow?.webContents.send('claude-chat-message', {
                        chatId,
                        responseId,
                        message: msg
                    });
                }
                console.log(`[Claude] Streaming complete for ${chatId}`);
                mainWindow?.webContents.send('claude-chat-complete', {
                    chatId,
                    responseId
                });
                activeChats.delete(chatId);
            }
            catch (streamError) {
                console.error('[Claude] Streaming error:', streamError);
                mainWindow?.webContents.send('claude-chat-error', {
                    chatId,
                    responseId,
                    error: streamError.message
                });
                activeChats.delete(chatId);
            }
        })();
        return { success: true, responseId };
    }
    catch (error) {
        console.error('[Claude] Handler error:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('claude-chat-interrupt', async (_event, { chatId }) => {
    const chat = activeChats.get(chatId);
    if (chat?.query) {
        try {
            await chat.query.interrupt();
            activeChats.delete(chatId);
            return { success: true };
        }
        catch (error) {
            return { success: false, error: error.message };
        }
    }
    return { success: false, error: 'No active chat found' };
});
electron_1.ipcMain.handle('get-conversation-todos', async (_event, { conversationId }) => {
    try {
        console.log(`[Claude] getTodos called for conversationId: ${conversationId}`);
        console.log(`[Claude] Available sessionIds keys:`, Array.from(sessionIds.keys()));
        console.log(`[Claude] Available conversationTodos keys:`, Array.from(conversationTodos.keys()));
        if (conversationTodos.has(conversationId)) {
            const todos = conversationTodos.get(conversationId);
            console.log(`[Claude] Retrieved cached todos for ${conversationId}:`, todos.length);
            return { success: true, todos };
        }
        const sessionId = sessionIds.get(conversationId);
        console.log(`[Claude] Found sessionId for ${conversationId}:`, sessionId);
        if (sessionId) {
            console.log(`[Claude] No cached todos, resuming session ${sessionId} to extract todos`);
            const result = (0, claude_agent_sdk_1.query)({
                prompt: '',
                options: {
                    resume: sessionId,
                    continue: false,
                    cwd: process.cwd(),
                    permissionMode: 'acceptEdits'
                }
            });
            let latestTodos = [];
            for await (const msg of result) {
                if (msg.type === 'assistant' && msg.message?.content) {
                    const content = msg.message.content;
                    if (Array.isArray(content)) {
                        content.forEach((block) => {
                            if (block.type === 'tool_use' && block.name === 'TodoWrite' && block.input?.todos) {
                                console.log(`[Claude] Found TodoWrite in replay:`, block.input.todos.length, 'todos');
                                latestTodos = block.input.todos;
                            }
                        });
                    }
                }
            }
            if (latestTodos.length > 0) {
                conversationTodos.set(conversationId, latestTodos);
            }
            console.log(`[Claude] Extracted ${latestTodos.length} todos from session replay`);
            return { success: true, todos: latestTodos };
        }
        console.log(`[Claude] No todos or session ID for ${conversationId}`);
        return { success: true, todos: [] };
    }
    catch (error) {
        console.error('[Claude] Error getting todos:', error);
        return { success: false, error: error.message, todos: [] };
    }
});
// Chat message handlers
electron_1.ipcMain.handle('chat-get-messages', async (_event, { chatId }) => {
    try {
        const messages = chatMessages.get(chatId) || [];
        const isStreaming = activeChats.has(chatId);
        console.log(`[Chat] Retrieved ${messages.length} messages for ${chatId}, streaming: ${isStreaming}`);
        console.log(`[Chat] Messages:`, messages);
        return { success: true, messages, isStreaming };
    }
    catch (error) {
        console.error('[Chat] Error getting messages:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('chat-save-message', async (_event, { chatId, message }) => {
    try {
        if (!chatMessages.has(chatId)) {
            chatMessages.set(chatId, []);
        }
        console.log(`[Chat] Saving message for ${chatId}:`, message);
        chatMessages.get(chatId).push(message);
        saveChatHistory();
        const allMessages = chatMessages.get(chatId);
        console.log(`[Chat] Saved message for ${chatId}, total: ${allMessages.length}`);
        console.log(`[Chat] All messages:`, allMessages);
        return { success: true };
    }
    catch (error) {
        console.error('[Chat] Error saving message:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('chat-clear-messages', async (_event, { chatId }) => {
    try {
        chatMessages.delete(chatId);
        saveChatHistory();
        console.log(`[Chat] Cleared messages for ${chatId}`);
        return { success: true };
    }
    catch (error) {
        console.error('[Chat] Error clearing messages:', error);
        return { success: false, error: error.message };
    }
});
// Conversation handlers
electron_1.ipcMain.handle('conversation-list', async (_event, { widgetKey }) => {
    try {
        const convos = conversations.get(widgetKey) || [];
        console.log(`[Conversations] Retrieved ${convos.length} conversations for ${widgetKey}`);
        return { success: true, conversations: convos };
    }
    catch (error) {
        console.error('[Conversations] Error listing conversations:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('conversation-create', async (_event, { widgetKey, title }) => {
    try {
        const conversationId = `${widgetKey}-${Date.now()}`;
        const conversation = {
            id: conversationId,
            title: title || 'New Conversation',
            createdAt: new Date().toISOString(),
            lastMessageAt: new Date().toISOString()
        };
        if (!conversations.has(widgetKey)) {
            conversations.set(widgetKey, []);
        }
        conversations.get(widgetKey).unshift(conversation);
        saveConversations();
        console.log(`[Conversations] Created new conversation ${conversationId}`);
        return { success: true, conversation };
    }
    catch (error) {
        console.error('[Conversations] Error creating conversation:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('conversation-update', async (_event, { widgetKey, conversationId, updates }) => {
    try {
        const convos = conversations.get(widgetKey);
        if (!convos) {
            return { success: false, error: 'Widget not found' };
        }
        const convoIndex = convos.findIndex(c => c.id === conversationId);
        if (convoIndex === -1) {
            return { success: false, error: 'Conversation not found' };
        }
        convos[convoIndex] = { ...convos[convoIndex], ...updates };
        saveConversations();
        console.log(`[Conversations] Updated conversation ${conversationId}`);
        return { success: true, conversation: convos[convoIndex] };
    }
    catch (error) {
        console.error('[Conversations] Error updating conversation:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('conversation-delete', async (_event, { widgetKey, conversationId }) => {
    try {
        const convos = conversations.get(widgetKey);
        if (!convos) {
            return { success: false, error: 'Widget not found' };
        }
        const filtered = convos.filter(c => c.id !== conversationId);
        conversations.set(widgetKey, filtered);
        saveConversations();
        chatMessages.delete(conversationId);
        saveChatHistory();
        console.log(`[Conversations] Deleted conversation ${conversationId}`);
        return { success: true };
    }
    catch (error) {
        console.error('[Conversations] Error deleting conversation:', error);
        return { success: false, error: error.message };
    }
});
// File loading handlers
electron_1.ipcMain.handle('load-data-file', async (_event, filePath) => {
    try {
        let resolvedPath = filePath;
        if (!path.isAbsolute(filePath)) {
            for (const dashboardPath of watchedPaths.keys()) {
                const dashboardDir = path.dirname(dashboardPath);
                const testPath = path.join(dashboardDir, filePath);
                if (fs.existsSync(testPath)) {
                    resolvedPath = testPath;
                    break;
                }
            }
        }
        console.log(`[DataLoader] Loading data from: ${resolvedPath}`);
        if (!fs.existsSync(resolvedPath)) {
            throw new Error(`File not found: ${filePath}`);
        }
        const content = fs.readFileSync(resolvedPath, 'utf-8');
        const data = JSON.parse(content);
        return { success: true, data };
    }
    catch (error) {
        console.error('[DataLoader] Error loading data file:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('load-text-file', async (_event, filePath) => {
    try {
        let resolvedPath = filePath;
        if (!path.isAbsolute(filePath)) {
            for (const dashboardPath of watchedPaths.keys()) {
                const dashboardDir = path.dirname(dashboardPath);
                const testPath = path.join(dashboardDir, filePath);
                if (fs.existsSync(testPath)) {
                    resolvedPath = testPath;
                    break;
                }
            }
        }
        console.log(`[TextLoader] Loading text file from: ${resolvedPath}`);
        if (!fs.existsSync(resolvedPath)) {
            throw new Error(`File not found: ${filePath}`);
        }
        const content = fs.readFileSync(resolvedPath, 'utf-8');
        return { success: true, content };
    }
    catch (error) {
        console.error('[TextLoader] Error loading text file:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('write-code-file', async (_event, { filePath, content }) => {
    try {
        console.log(`[CodeWriter] Writing code file to: ${filePath}`);
        const dir = path.dirname(filePath);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
        fs.writeFileSync(filePath, content, 'utf-8');
        return { success: true };
    }
    catch (error) {
        console.error('[CodeWriter] Error writing code file:', error);
        return { success: false, error: error.message };
    }
});
// Command Runner Handler
electron_1.ipcMain.handle('run-command', async (_event, { command, cwd }) => {
    try {
        console.log(`[CommandRunner] Executing: ${command}`);
        const workingDir = cwd || process.cwd();
        return new Promise((resolve) => {
            const options = {
                cwd: workingDir,
                shell: true
            };
            const child = (0, child_process_1.spawn)(command, [], options);
            let output = '';
            let hasError = false;
            let errorMessage = '';
            child.stdout?.on('data', (data) => {
                output += data.toString();
            });
            child.stderr?.on('data', (data) => {
                output += data.toString();
            });
            child.on('error', (error) => {
                hasError = true;
                errorMessage = error.message;
            });
            child.on('close', (code) => {
                console.log(`[CommandRunner] Command completed with code ${code}`);
                if (hasError) {
                    resolve({ success: false, error: errorMessage });
                }
                else if (code !== 0) {
                    resolve({ success: false, error: output || `Command exited with code ${code}` });
                }
                else {
                    resolve({ success: true, output: output || 'Command completed successfully' });
                }
            });
        });
    }
    catch (error) {
        console.error('[CommandRunner] Error executing command:', error);
        return { success: false, error: error.message };
    }
});
// Streaming Command Runners
electron_1.ipcMain.handle('start-streaming-command', async (event, { widgetId, command, cwd }) => {
    try {
        console.log(`[StreamingCommand] Starting command for widget ${widgetId}: ${command}`);
        if (runningCommands.has(widgetId)) {
            const existing = runningCommands.get(widgetId);
            existing.process.kill();
            runningCommands.delete(widgetId);
        }
        const workingDir = cwd || process.cwd();
        const options = {
            cwd: workingDir,
            shell: true
        };
        const child = (0, child_process_1.spawn)(command, [], options);
        runningCommands.set(widgetId, {
            process: child,
            output: ''
        });
        child.stdout?.on('data', (data) => {
            const text = data.toString();
            const entry = runningCommands.get(widgetId);
            if (entry) {
                entry.output += text;
                event.sender.send('command-output', { widgetId, output: text, type: 'stdout' });
            }
        });
        child.stderr?.on('data', (data) => {
            const text = data.toString();
            const entry = runningCommands.get(widgetId);
            if (entry) {
                entry.output += text;
                event.sender.send('command-output', { widgetId, output: text, type: 'stderr' });
            }
        });
        child.on('close', (code) => {
            console.log(`[StreamingCommand] Command for widget ${widgetId} exited with code ${code}`);
            event.sender.send('command-exit', { widgetId, code });
            runningCommands.delete(widgetId);
        });
        child.on('error', (error) => {
            console.error(`[StreamingCommand] Error for widget ${widgetId}:`, error);
            event.sender.send('command-error', { widgetId, error: error.message });
            runningCommands.delete(widgetId);
        });
        return { success: true, message: 'Command started' };
    }
    catch (error) {
        console.error('[StreamingCommand] Error starting command:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('stop-streaming-command', async (_event, { widgetId }) => {
    try {
        console.log(`[StreamingCommand] Stopping command for widget ${widgetId}`);
        if (!runningCommands.has(widgetId)) {
            return { success: false, error: 'No command running for this widget' };
        }
        const entry = runningCommands.get(widgetId);
        entry.process.kill();
        runningCommands.delete(widgetId);
        return { success: true, message: 'Command stopped' };
    }
    catch (error) {
        console.error('[StreamingCommand] Error stopping command:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('is-command-running', async (_event, { widgetId }) => {
    return { running: runningCommands.has(widgetId) };
});
// WebContentsView Handlers
electron_1.ipcMain.handle('create-webcontentsview', async (event, { widgetId, url, bounds, backgroundColor }) => {
    try {
        console.log(`[WebContentsView] Creating view for widget ${widgetId}`);
        if (webContentsViews.has(widgetId)) {
            const existing = webContentsViews.get(widgetId);
            mainWindow.contentView.removeChildView(existing.view);
            existing.view.webContents.close();
            webContentsViews.delete(widgetId);
        }
        const view = new electron_1.WebContentsView({
            webPreferences: {
                nodeIntegration: false,
                contextIsolation: true,
                sandbox: true
            }
        });
        view.setBackgroundColor(backgroundColor || '#000000');
        view.setBounds(bounds);
        mainWindow.contentView.addChildView(view);
        const viewData = {
            view,
            history: url ? [url] : [],
            historyIndex: 0
        };
        webContentsViews.set(widgetId, viewData);
        const ensureTransparent = () => {
            if (backgroundColor) {
                view.setBackgroundColor(backgroundColor);
            }
        };
        view.webContents.on('did-start-loading', ensureTransparent);
        view.webContents.on('did-finish-load', ensureTransparent);
        view.webContents.on('did-navigate', ensureTransparent);
        view.webContents.on('did-navigate-in-page', ensureTransparent);
        if (url) {
            view.webContents.loadURL(url);
        }
        view.webContents.on('did-navigate', (_ev, navigationUrl) => {
            console.log(`[WebContentsView] ${widgetId} navigated to ${navigationUrl}`);
            event.sender.send('webcontentsview-navigated', {
                widgetId,
                url: navigationUrl,
                canGoBack: view.webContents.canGoBack(),
                canGoForward: view.webContents.canGoForward()
            });
        });
        view.webContents.on('did-navigate-in-page', (_ev, navigationUrl) => {
            event.sender.send('webcontentsview-navigated', {
                widgetId,
                url: navigationUrl,
                canGoBack: view.webContents.canGoBack(),
                canGoForward: view.webContents.canGoForward()
            });
        });
        view.webContents.setWindowOpenHandler(({ url: newUrl }) => {
            electron_1.shell.openExternal(newUrl);
            return { action: 'deny' };
        });
        return { success: true };
    }
    catch (error) {
        console.error('[WebContentsView] Error creating view:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('navigate-webcontentsview', async (_event, { widgetId, url }) => {
    try {
        const viewData = webContentsViews.get(widgetId);
        if (!viewData) {
            return { success: false, error: 'View not found' };
        }
        viewData.view.webContents.loadURL(url);
        return { success: true };
    }
    catch (error) {
        console.error('[WebContentsView] Error navigating:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('webcontentsview-go-back', async (_event, { widgetId }) => {
    try {
        const viewData = webContentsViews.get(widgetId);
        if (!viewData) {
            return { success: false, error: 'View not found' };
        }
        if (viewData.view.webContents.canGoBack()) {
            viewData.view.webContents.goBack();
            return { success: true };
        }
        return { success: false, error: 'Cannot go back' };
    }
    catch (error) {
        console.error('[WebContentsView] Error going back:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('webcontentsview-go-forward', async (_event, { widgetId }) => {
    try {
        const viewData = webContentsViews.get(widgetId);
        if (!viewData) {
            return { success: false, error: 'View not found' };
        }
        if (viewData.view.webContents.canGoForward()) {
            viewData.view.webContents.goForward();
            return { success: true };
        }
        return { success: false, error: 'Cannot go forward' };
    }
    catch (error) {
        console.error('[WebContentsView] Error going forward:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('webcontentsview-reload', async (_event, { widgetId }) => {
    try {
        const viewData = webContentsViews.get(widgetId);
        if (!viewData) {
            return { success: false, error: 'View not found' };
        }
        viewData.view.webContents.reload();
        return { success: true };
    }
    catch (error) {
        console.error('[WebContentsView] Error reloading:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('webcontentsview-can-navigate', async (_event, { widgetId }) => {
    try {
        const viewData = webContentsViews.get(widgetId);
        if (!viewData) {
            return { success: false, error: 'View not found' };
        }
        return {
            success: true,
            canGoBack: viewData.view.webContents.canGoBack(),
            canGoForward: viewData.view.webContents.canGoForward()
        };
    }
    catch (error) {
        console.error('[WebContentsView] Error checking navigation state:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('update-webcontentsview-bounds', async (_event, { widgetId, bounds }) => {
    try {
        const viewData = webContentsViews.get(widgetId);
        if (!viewData) {
            return { success: false, error: 'View not found' };
        }
        viewData.view.setBounds(bounds);
        return { success: true };
    }
    catch (error) {
        console.error('[WebContentsView] Error updating bounds:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('destroy-webcontentsview', async (_event, { widgetId }) => {
    try {
        const viewData = webContentsViews.get(widgetId);
        if (!viewData) {
            return { success: true };
        }
        mainWindow.contentView.removeChildView(viewData.view);
        viewData.view.webContents.close();
        webContentsViews.delete(widgetId);
        return { success: true };
    }
    catch (error) {
        console.error('[WebContentsView] Error destroying view:', error);
        return { success: false, error: error.message };
    }
});
// Widget Regeneration Handlers
electron_1.ipcMain.handle('regenerate-widget', async (_event, { dashboardId, widgetId }) => {
    try {
        console.log(`[Regenerate] Regenerating widget ${widgetId} in dashboard ${dashboardId}`);
        let targetPath = null;
        let entry = null;
        for (const [filePath, e] of watchedPaths.entries()) {
            if (e.dashboard.id === dashboardId) {
                targetPath = filePath;
                entry = e;
                break;
            }
        }
        if (!targetPath || !entry) {
            return { success: false, error: 'Dashboard not found' };
        }
        const widget = entry.dashboard.widgets.find(w => w.id === widgetId);
        if (!widget) {
            return { success: false, error: 'Widget not found' };
        }
        // Built-in test runners with parsers
        const testRunners = {
            cargo: {
                command: 'cargo test 2>&1',
                parser: (output) => {
                    const tests = [];
                    const lines = output.split('\n');
                    for (const line of lines) {
                        const match = line.match(/^test\s+(\S+)\s+\.\.\.\s+(\w+)/);
                        if (match) {
                            const name = match[1];
                            const result = match[2];
                            let status;
                            if (result === 'ok') {
                                status = 'passed';
                            }
                            else if (result === 'FAILED') {
                                status = 'failed';
                            }
                            else if (result === 'ignored') {
                                status = 'skipped';
                            }
                            else {
                                status = 'passed';
                            }
                            tests.push({ name, status });
                        }
                    }
                    return tests;
                }
            },
            jest: {
                command: 'npm test -- --json 2>&1',
                parser: (output) => {
                    try {
                        const result = JSON.parse(output);
                        const tests = [];
                        if (result.testResults) {
                            result.testResults.forEach((file) => {
                                file.assertionResults?.forEach((test) => {
                                    tests.push({
                                        name: test.title || test.fullName,
                                        status: test.status === 'passed' ? 'passed' : test.status === 'failed' ? 'failed' : 'skipped',
                                        duration: test.duration,
                                        error: test.failureMessages?.[0]
                                    });
                                });
                            });
                        }
                        return tests;
                    }
                    catch (e) {
                        console.error('[TestRunner] Failed to parse Jest output:', e);
                        return [];
                    }
                }
            },
            pytest: {
                command: 'pytest --tb=short -v 2>&1',
                parser: (output) => {
                    const tests = [];
                    const lines = output.split('\n');
                    for (const line of lines) {
                        const match = line.match(/^(.+?)::([\w_]+)\s+(PASSED|FAILED|SKIPPED)/);
                        if (match) {
                            const name = `${match[1]}::${match[2]}`;
                            const result = match[3];
                            let status;
                            if (result === 'PASSED') {
                                status = 'passed';
                            }
                            else if (result === 'FAILED') {
                                status = 'failed';
                            }
                            else if (result === 'SKIPPED') {
                                status = 'skipped';
                            }
                            tests.push({ name, status });
                        }
                    }
                    return tests;
                }
            }
        };
        let command = widget.regenerateCommand || widget.regenerate;
        let parser = null;
        if (widget.testRunner && testRunners[widget.testRunner]) {
            const runner = testRunners[widget.testRunner];
            command = widget.regenerateCommand || runner.command;
            parser = runner.parser;
            console.log(`[Regenerate] Using built-in test runner: ${widget.testRunner}`);
        }
        if (!command && widget.regenerateScript) {
            const dashboardDir = path.dirname(targetPath);
            const scriptPath = path.join(dashboardDir, widget.regenerateScript);
            if (!fs.existsSync(scriptPath)) {
                return { success: false, error: `Script file not found: ${widget.regenerateScript}` };
            }
            command = fs.readFileSync(scriptPath, 'utf-8').trim();
        }
        if (!command) {
            return { success: false, error: 'No testRunner, regenerateCommand, regenerate, or regenerateScript specified' };
        }
        let workingDir = process.cwd();
        if (entry.projectId) {
            const project = projects.get(entry.projectId);
            if (project?.rootPath) {
                workingDir = project.rootPath;
            }
        }
        console.log(`[Regenerate] Running command: ${command}`);
        console.log(`[Regenerate] Working directory: ${workingDir}`);
        return new Promise((resolve) => {
            const options = {
                cwd: workingDir,
                shell: true
            };
            const child = (0, child_process_1.spawn)(command, [], options);
            let stdout = '';
            let stderr = '';
            let hasError = false;
            let errorMessage = '';
            child.stdout?.on('data', (data) => {
                stdout += data.toString();
            });
            child.stderr?.on('data', (data) => {
                stderr += data.toString();
            });
            child.on('error', (error) => {
                hasError = true;
                errorMessage = error.message;
            });
            child.on('close', (code) => {
                console.log(`[Regenerate] Command completed with code ${code}`);
                if (hasError) {
                    resolve({ success: false, error: errorMessage });
                    return;
                }
                const allowNonZeroExit = parser !== null;
                if (code !== 0 && !allowNonZeroExit) {
                    resolve({ success: false, error: stderr || stdout || `Command exited with code ${code}` });
                    return;
                }
                let data;
                const outputTrimmed = stdout.trim();
                if (parser) {
                    console.log('[Regenerate] Using built-in parser for test runner output');
                    try {
                        data = parser(stdout);
                    }
                    catch (parseError) {
                        console.error('[Regenerate] Parser error:', parseError);
                        resolve({ success: false, error: `Parser error: ${parseError.message}` });
                        return;
                    }
                }
                else if (widget.type === 'stat') {
                    try {
                        data = JSON.parse(outputTrimmed);
                    }
                    catch {
                        data = outputTrimmed;
                    }
                }
                else {
                    try {
                        data = JSON.parse(outputTrimmed);
                    }
                    catch (parseError) {
                        console.error('[Regenerate] JSON parse error:', parseError);
                        resolve({ success: false, error: `Invalid JSON output: ${parseError.message}` });
                        return;
                    }
                }
                try {
                    switch (widget.type) {
                        case 'barChart':
                            widget.data = data;
                            break;
                        case 'stat':
                            widget.value = data;
                            break;
                        case 'progress':
                            if (typeof data === 'object' && data.value !== undefined) {
                                widget.value = data.value;
                                if (data.text !== undefined)
                                    widget.text = data.text;
                            }
                            else {
                                widget.value = data;
                            }
                            break;
                        case 'diffList':
                        case 'fileList':
                        case 'todoList':
                        case 'keyValue':
                            widget.items = data;
                            break;
                        case 'testResults':
                            widget.tests = data;
                            break;
                        case 'jsonViewer':
                            widget.data = data;
                            break;
                        default:
                            if ('data' in widget) {
                                widget.data = data;
                            }
                            else if ('items' in widget) {
                                widget.items = data;
                            }
                            else if ('tests' in widget) {
                                widget.tests = data;
                            }
                            else {
                                resolve({ success: false, error: `Don't know how to update widget type: ${widget.type}` });
                                return;
                            }
                    }
                    entry.lastWriteTime = Date.now();
                    fs.writeFileSync(targetPath, JSON.stringify(entry.dashboard, null, 2), 'utf-8');
                    broadcastDashboards();
                    console.log(`[Regenerate] Successfully regenerated widget ${widgetId}`);
                    resolve({ success: true });
                }
                catch (updateError) {
                    console.error('[Regenerate] Error updating widget:', updateError);
                    resolve({ success: false, error: `Failed to update widget: ${updateError.message}` });
                }
            });
        });
    }
    catch (error) {
        console.error('[Regenerate] Error regenerating widget:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('regenerate-all-widgets', async (event, { dashboardId }) => {
    try {
        console.log(`[Regenerate] Regenerating all widgets in dashboard ${dashboardId}`);
        let entry = null;
        for (const [filePath, e] of watchedPaths.entries()) {
            if (e.dashboard.id === dashboardId) {
                entry = e;
                break;
            }
        }
        if (!entry) {
            return { success: false, error: 'Dashboard not found' };
        }
        const regeneratableWidgets = entry.dashboard.widgets.filter((w) => w.regenerateCommand || w.regenerateScript || w.regenerate);
        if (regeneratableWidgets.length === 0) {
            return { success: true, message: 'No widgets have regenerate commands', results: [] };
        }
        console.log(`[Regenerate] Found ${regeneratableWidgets.length} widgets to regenerate`);
        const results = [];
        for (const widget of regeneratableWidgets) {
            // Call regenerate-widget handler for each widget
            const handlers = electron_1.ipcMain._events['regenerate-widget'];
            const handler = Array.isArray(handlers) ? handlers[0] : handlers;
            const result = await handler?.(event, {
                dashboardId,
                widgetId: widget.id
            });
            results.push({
                widgetId: widget.id,
                label: Array.isArray(widget.label) ? widget.label.join(' - ') : widget.label,
                ...result
            });
        }
        const successCount = results.filter(r => r.success).length;
        const failureCount = results.length - successCount;
        return {
            success: true,
            message: `Regenerated ${successCount} widgets, ${failureCount} failed`,
            results
        };
    }
    catch (error) {
        console.error('[Regenerate] Error regenerating all widgets:', error);
        return { success: false, error: error.message };
    }
});
// Project Management Handlers
electron_1.ipcMain.handle('project-list', async () => {
    try {
        const projectList = Array.from(projects.values());
        console.log(`[Projects] Listing ${projectList.length} projects`);
        return { success: true, projects: projectList };
    }
    catch (error) {
        console.error('[Projects] Error listing projects:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('project-get', async (_event, { projectId }) => {
    try {
        const project = projects.get(projectId);
        if (!project) {
            return { success: false, error: 'Project not found' };
        }
        console.log(`[Projects] Retrieved project ${projectId}`);
        return { success: true, project };
    }
    catch (error) {
        console.error('[Projects] Error getting project:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('project-add', async (_event, { projectPath, type, name, description }) => {
    try {
        console.log(`[Projects] Adding project from ${projectPath}`);
        const validation = projectUtils.validateProjectPath(projectPath);
        if (!validation.valid) {
            return { success: false, error: validation.error };
        }
        // Detect the project structure type (embedded or standalone)
        const structureType = projectUtils.detectProjectType(projectPath);
        const configDir = projectUtils.getProjectConfigDir(projectPath, structureType);
        let projectConfig = projectUtils.loadProjectConfig(configDir);
        if (!projectConfig) {
            const result = projectUtils.initializeProject(projectPath, structureType, { name, description });
            if (!result.success) {
                return result;
            }
            projectConfig = result.projectConfig;
        }
        const dashboardFiles = projectUtils.findDashboardFiles(configDir);
        projectConfig.dashboards = dashboardFiles;
        // Create a Project object that includes both structure type and project type
        const project = {
            ...projectConfig,
            type: type || projectConfig.type
        };
        projects.set(project.id, project);
        saveProjects();
        dashboardFiles.forEach(dashboardPath => {
            watchFile(dashboardPath);
            const entry = watchedPaths.get(dashboardPath);
            if (entry) {
                entry.projectId = projectConfig.id;
            }
        });
        broadcastDashboards();
        console.log(`[Projects] Added project ${projectConfig.id} with ${dashboardFiles.length} dashboards`);
        return { success: true, project: projectConfig };
    }
    catch (error) {
        console.error('[Projects] Error adding project:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('project-update', async (_event, { projectId, updates }) => {
    try {
        const project = projects.get(projectId);
        if (!project) {
            return { success: false, error: 'Project not found' };
        }
        Object.assign(project, updates);
        projects.set(projectId, project);
        saveProjects();
        broadcastDashboards();
        console.log(`[Projects] Updated project ${projectId}`);
        return { success: true, project };
    }
    catch (error) {
        console.error('[Projects] Error updating project:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('project-remove', async (_event, { projectId }) => {
    try {
        console.log(`[Projects] Attempting to remove project ${projectId}`);
        const project = projects.get(projectId);
        if (!project) {
            console.log(`[Projects] Project ${projectId} not found`);
            return { success: false, error: 'Project not found' };
        }
        console.log(`[Projects] Project found:`, project);
        console.log(`[Projects] Dashboard paths before removal:`, Array.from(watchedPaths.keys()));
        if (project.dashboards && Array.isArray(project.dashboards)) {
            console.log(`[Projects] Unwatching ${project.dashboards.length} dashboard(s)`);
            project.dashboards.forEach(dashboardPath => {
                console.log(`[Projects] Unwatching: ${dashboardPath}`);
                unwatchFile(dashboardPath);
            });
        }
        else {
            console.log(`[Projects] No dashboards to unwatch`);
        }
        console.log(`[Projects] Dashboard paths after unwatching:`, Array.from(watchedPaths.keys()));
        projects.delete(projectId);
        saveProjects();
        savePaths();
        const remainingDashboards = getAllDashboards();
        console.log(`[Projects] Broadcasting ${remainingDashboards.length} remaining dashboards`);
        broadcastDashboards();
        console.log(`[Projects] Successfully removed project ${projectId}`);
        return { success: true };
    }
    catch (error) {
        console.error('[Projects] Error removing project:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('project-init', async (_event, { projectPath, name, description }) => {
    try {
        console.log(`[Projects] Initializing project in ${projectPath}`);
        const validation = projectUtils.validateProjectPath(projectPath);
        if (!validation.valid) {
            return { success: false, error: validation.error };
        }
        if (projectUtils.hasEmbeddedProject(projectPath)) {
            return { success: false, error: 'Project already has .ai-dashboard folder' };
        }
        const result = projectUtils.initializeProject(projectPath, 'embedded', { name, description });
        if (result.success) {
            console.log(`[Projects] Initialized project structure at ${result.configDir}`);
        }
        return result;
    }
    catch (error) {
        console.error('[Projects] Error initializing project:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('project-open-folder', async (_event, { projectId }) => {
    try {
        const project = projects.get(projectId);
        if (!project) {
            return { success: false, error: 'Project not found' };
        }
        const folderPath = project.rootPath || project.path;
        if (!folderPath) {
            return { success: false, error: 'Project has no path configured' };
        }
        if (!fs.existsSync(folderPath)) {
            return { success: false, error: 'Project folder does not exist' };
        }
        await electron_1.shell.openPath(folderPath);
        console.log(`[Projects] Opened folder for project ${projectId}`);
        return { success: true };
    }
    catch (error) {
        console.error('[Projects] Error opening project folder:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('project-select-folder', async () => {
    try {
        const result = await electron_1.dialog.showOpenDialog(mainWindow, {
            properties: ['openDirectory'],
            title: 'Select Project Folder'
        });
        if (result.canceled || result.filePaths.length === 0) {
            return { success: false, canceled: true };
        }
        return { success: true, path: result.filePaths[0] };
    }
    catch (error) {
        console.error('[Projects] Error showing folder picker:', error);
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('project-generate-design', async (_event, { projectName }) => {
    try {
        console.log(`[Projects] Generating design for project: ${projectName}`);
        const prompt = `Generate an abstract SVG icon and color theme for a project named "${projectName}".

CRITICAL SVG Requirements:
1. The SVG will be styled with CSS, so DO NOT include any color attributes:
   - NO stroke attribute
   - NO fill attribute
   - NO color attribute
   - NO style attribute
2. Use viewBox="0 0 60 60" and xmlns="http://www.w3.org/2000/svg"
3. Be creative with complexity - intricate designs are welcome
4. Design for visibility at 48x48px - ensure shapes are bold enough to see
5. Make it abstract, unique, and inspired by the project name's meaning or vibe
6. Use stroke-linecap="round" and stroke-linejoin="round" for smooth appearance

Theme Requirements - Create a UNIQUE, cohesive visual identity:
- accent: A vibrant, distinctive primary color (hex format, avoid generic blue)
- textColor: Light gray for text (#c9d1d9 or similar)
- positive: Success color (greenish, hex format)
- negative: Error color (reddish, hex format)
- bgApp: Dark background color (hex format)
- textHead: Font for headings - be creative! (e.g., "Georgia, serif" or "Courier New, monospace" or system fonts)
- textBody: Font for body text - should complement textHead
- widgetBg: Widget background with transparency (rgba format, e.g., "rgba(22, 27, 34, 0.8)")
- widgetBorder: Widget border style (e.g., "1px solid rgba(255, 255, 255, 0.1)" or "2px dashed rgba(...)")
- widgetRadius: Widget corner radius - vary this! (e.g., "0px" for sharp, "12px" for round, "20px" for very round)
- chartRadius: Chart element corner radius (e.g., "0px", "4px", "8px")
- bgLayer: Background layer CSS object for gradients/effects (can be empty {} or include background, opacity, etc.)

Return ONLY a JSON object with this exact structure (no markdown, no explanation):
{
  "icon": "<svg viewBox=\\"0 0 60 60\\" xmlns=\\"http://www.w3.org/2000/svg\\"><path d=\\"...\\" stroke-linecap=\\"round\\" stroke-linejoin=\\"round\\"/></svg>",
  "theme": {
    "accent": "#hexcolor",
    "textColor": "#hexcolor",
    "positive": "#hexcolor",
    "negative": "#hexcolor",
    "bgApp": "#hexcolor",
    "textHead": "font-family string",
    "textBody": "font-family string",
    "widgetBg": "rgba(...)",
    "widgetBorder": "border style string",
    "widgetRadius": "Xpx",
    "chartRadius": "Xpx",
    "bgLayer": {}
  }
}

Example good SVG (notice NO color attributes):
<svg viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"><circle cx="30" cy="30" r="20" stroke-linecap="round" stroke-linejoin="round"/><path d="M20 30 L30 40 L40 20" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
        const queryInstance = (0, claude_agent_sdk_1.query)({
            prompt,
            options: {
                model: 'claude-haiku-4-5-20251001'
            }
        });
        console.log('[Projects] Waiting for Claude response...');
        let responseText = '';
        for await (const message of queryInstance) {
            console.log('[Projects] Received message:', JSON.stringify(message, null, 2));
            if (message.type === 'result') {
                // Try different ways to get the response text from the SDK
                if (message.result) {
                    responseText = message.result;
                }
                else if (message.content && Array.isArray(message.content)) {
                    const textBlocks = message.content.filter((block) => block.type === 'text');
                    responseText = textBlocks.map((block) => block.text).join('');
                }
                else if (message.text) {
                    responseText = message.text;
                }
                console.log('[Projects] Got response text:', responseText.substring(0, 100) + '...');
                if (responseText) {
                    break;
                }
            }
            else if (message.type === 'assistant' && message.message?.content) {
                const content = message.message.content;
                if (Array.isArray(content)) {
                    const textBlocks = content.filter((block) => block.type === 'text');
                    if (textBlocks.length > 0) {
                        responseText = textBlocks.map((block) => block.text).join('');
                        console.log('[Projects] Got response text from assistant message:', responseText.substring(0, 100) + '...');
                        break;
                    }
                }
            }
        }
        if (!responseText) {
            throw new Error('No response text received from Claude');
        }
        const text = responseText.trim();
        const jsonText = text.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
        const design = JSON.parse(jsonText);
        console.log('[Projects] Generated design:', design);
        return { success: true, design };
    }
    catch (error) {
        console.error('[Projects] Error generating design:', error);
        return {
            success: true,
            design: {
                icon: '<svg viewBox="0 0 60 60"><rect x="10" y="10" width="40" height="40" /></svg>',
                theme: {
                    accent: '#58a6ff',
                    textColor: '#c9d1d9',
                    positive: '#3fb950',
                    negative: '#f85149',
                    bgApp: '#0d1117'
                }
            }
        };
    }
});
electron_1.app.whenReady().then(() => {
    const savedPaths = loadSavedPaths();
    savedPaths.forEach(watchFile);
    loadChatHistory();
    loadConversations();
    loadProjects();
    projects.forEach((project) => {
        if (project.dashboards && Array.isArray(project.dashboards)) {
            project.dashboards.forEach(dashboardPath => {
                console.log(`[Projects] Watching dashboard from project ${project.id}: ${dashboardPath}`);
                watchFile(dashboardPath);
                const entry = watchedPaths.get(dashboardPath);
                if (entry) {
                    entry.projectId = project.id;
                }
            });
        }
    });
    createWindow();
});
electron_1.app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        watchedPaths.forEach((entry) => entry.watcher?.close());
        electron_1.app.quit();
    }
});
electron_1.app.on('activate', () => {
    if (mainWindow === null) {
        createWindow();
    }
});
