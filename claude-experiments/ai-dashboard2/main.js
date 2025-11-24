const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const util = require('util');
const { query } = require('@anthropic-ai/claude-agent-sdk');
const { createDashboardTools } = require('./dashboard-tools');

const isDev = process.env.NODE_ENV === 'development';

let mainWindow = null;
const watchedPaths = new Map(); // path -> { watcher, dashboard, lastWriteTime }
const configFilePath = path.join(app.getPath('userData'), 'dashboard-paths.json');
const chatStoragePath = path.join(app.getPath('userData'), 'chat-history.json');
const conversationsPath = path.join(app.getPath('userData'), 'conversations.json');
const chatMessages = new Map(); // chatId -> array of messages
const sessionIds = new Map(); // chatId -> SDK session ID for resuming
const conversations = new Map(); // widgetId -> array of conversation metadata
const conversationTodos = new Map(); // conversationId -> array of todos from SDK

// Load saved config paths
function loadSavedPaths() {
  try {
    if (fs.existsSync(configFilePath)) {
      const data = JSON.parse(fs.readFileSync(configFilePath, 'utf-8'));
      return data.paths || [];
    }
  } catch (e) {
    console.error('Error loading saved paths:', e);
  }
  return [];
}

// Save config paths
function savePaths() {
  try {
    const paths = Array.from(watchedPaths.keys());
    fs.writeFileSync(configFilePath, JSON.stringify({ paths }, null, 2));
  } catch (e) {
    console.error('Error saving paths:', e);
  }
}

// Load chat history from disk
function loadChatHistory() {
  try {
    if (fs.existsSync(chatStoragePath)) {
      const data = JSON.parse(fs.readFileSync(chatStoragePath, 'utf-8'));
      // Convert object back to Map
      Object.entries(data).forEach(([chatId, messages]) => {
        chatMessages.set(chatId, messages);
      });
      console.log(`[Chat] Loaded history for ${chatMessages.size} chats`);
    }
  } catch (e) {
    console.error('Error loading chat history:', e);
  }
}

// Save chat history to disk
function saveChatHistory() {
  try {
    // Convert Map to object for JSON serialization
    const data = Object.fromEntries(chatMessages);
    fs.writeFileSync(chatStoragePath, JSON.stringify(data, null, 2));
    console.log(`[Chat] Saved history for ${chatMessages.size} chats`);
  } catch (e) {
    console.error('Error saving chat history:', e);
  }
}

// Load conversations metadata from disk
function loadConversations() {
  try {
    if (fs.existsSync(conversationsPath)) {
      const data = JSON.parse(fs.readFileSync(conversationsPath, 'utf-8'));
      // Convert object back to Map
      Object.entries(data).forEach(([widgetId, convos]) => {
        conversations.set(widgetId, convos);
        // Load sessionIds back into memory
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
  } catch (e) {
    console.error('Error loading conversations:', e);
  }
}

// Save conversations metadata to disk
function saveConversations() {
  try {
    // Convert Map to object for JSON serialization
    const data = Object.fromEntries(conversations);
    fs.writeFileSync(conversationsPath, JSON.stringify(data, null, 2));
    console.log(`[Conversations] Saved ${conversations.size} widget conversation lists`);
  } catch (e) {
    console.error('Error saving conversations:', e);
  }
}

// Parse a dashboard JSON file
function parseDashboardFile(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const dashboard = JSON.parse(content);
    // Add source path for identification
    dashboard._sourcePath = filePath;
    return dashboard;
  } catch (e) {
    console.error(`Error parsing ${filePath}:`, e);
    return null;
  }
}

// Watch a file for changes
function watchFile(filePath) {
  if (watchedPaths.has(filePath)) return;

  const dashboard = parseDashboardFile(filePath);
  if (!dashboard) return;

  // Use fs.watchFile instead of fs.watch to handle atomic writes (temp file + rename)
  // This is common with editors like VS Code, Claude Code, etc.
  const watcher = fs.watchFile(filePath, { interval: 100 }, (curr, prev) => {
    // Check if the file was actually modified (not just accessed)
    if (curr.mtime > prev.mtime) {
      const entry = watchedPaths.get(filePath);
      if (!entry) return;

      // Check if this change was made by us (within 1 second of our last write)
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
      } else {
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
    fs.unwatchFile(filePath); // Use unwatchFile for fs.watchFile
    watchedPaths.delete(filePath);
  }
}

// Get all loaded dashboards
function getAllDashboards() {
  return Array.from(watchedPaths.values())
    .map(entry => entry.dashboard)
    .filter(Boolean);
}

// Broadcast dashboards to renderer
function broadcastDashboards() {
  const dashboards = getAllDashboards();
  console.log(`[Broadcast] Sending ${dashboards.length} dashboards to renderer`);
  if (mainWindow) {
    mainWindow.webContents.send('dashboard-updated', dashboards);
  } else {
    console.warn('[Broadcast] No main window available');
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
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
  } else {
    mainWindow.loadFile(path.join(__dirname, 'dist', 'index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// IPC Handlers
ipcMain.handle('load-dashboards', () => {
  return getAllDashboards();
});

ipcMain.handle('add-config-path', (event, filePath) => {
  if (!fs.existsSync(filePath)) {
    return { success: false, error: 'File does not exist' };
  }
  watchFile(filePath);
  savePaths();
  broadcastDashboards();
  return { success: true };
});

ipcMain.handle('remove-config-path', (event, filePath) => {
  unwatchFile(filePath);
  savePaths();
  broadcastDashboards();
  return { success: true };
});

ipcMain.handle('get-watched-paths', () => {
  return Array.from(watchedPaths.keys());
});

ipcMain.handle('update-widget-dimensions', async (event, { dashboardId, widgetId, dimensions }) => {
  try {
    // Find the dashboard in memory
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

    // Update the in-memory dashboard
    const widget = entry.dashboard.widgets.find(w => w.id === widgetId);
    if (!widget) {
      return { success: false, error: 'Widget not found' };
    }

    // Only update the changed properties
    if (dimensions.width !== undefined) widget.width = dimensions.width;
    if (dimensions.height !== undefined) widget.height = dimensions.height;
    if (dimensions.x !== undefined) widget.x = dimensions.x;
    if (dimensions.y !== undefined) widget.y = dimensions.y;

    // Record the time of this write so we can ignore the file watcher event
    entry.lastWriteTime = Date.now();

    // Write the updated in-memory dashboard to file
    fs.writeFileSync(targetPath, JSON.stringify(entry.dashboard, null, 2), 'utf-8');

    // DON'T broadcast - the frontend already has this state from the drag action
    // Broadcasting would cause unnecessary re-renders and state resets

    return { success: true };
  } catch (error) {
    console.error('Error updating widget dimensions:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('update-layout-settings', async (event, { dashboardId, settings }) => {
  try {
    // Find the dashboard in memory
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

    // Update layout settings
    if (!entry.dashboard.layout) {
      entry.dashboard.layout = {};
    }

    if (settings.widgetGap !== undefined) {
      entry.dashboard.layout.widgetGap = settings.widgetGap;
    }
    if (settings.buffer !== undefined) {
      entry.dashboard.layout.buffer = settings.buffer;
    }

    // Record the time of this write
    entry.lastWriteTime = Date.now();

    // Write to file
    fs.writeFileSync(targetPath, JSON.stringify(entry.dashboard, null, 2), 'utf-8');

    // Broadcast this change so all widgets update with new settings
    broadcastDashboards();

    return { success: true };
  } catch (error) {
    console.error('Error updating layout settings:', error);
    return { success: false, error: error.message };
  }
});

// Claude Agent SDK Chat Handlers
const activeChats = new Map(); // chatId -> { query, abortController }

ipcMain.handle('claude-chat-send', async (event, { chatId, message, options = {} }) => {
  try {
    const responseId = `${chatId}-${Date.now()}`;

    // Start streaming in the background - don't await it
    (async () => {
      try {
        console.log(`[Claude] Starting chat for ${chatId} with message: ${message.substring(0, 50)}...`);

        // Check if we have an SDK session ID to resume
        const existingSessionId = sessionIds.get(chatId);
        const history = chatMessages.get(chatId) || [];

        console.log(`[Claude] ${existingSessionId ? 'Resuming' : 'Starting'} conversation ${chatId}`);
        if (existingSessionId) {
          console.log(`[Claude] Using session ID: ${existingSessionId}`);
        }

        // Build system prompt with dashboard context if provided
        let systemPrompt = '';
        if (options.dashboardContext) {
          const ctx = options.dashboardContext;

          // Build list of available widget-specific tools
          const widgetToolsList = ctx.config.widgets.map(w => {
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
          }).filter(t => t).join('\n');

          systemPrompt = `You are assisting with a dashboard application. Here is the current dashboard context:

Dashboard File Path: ${ctx.filePath}
Dashboard ID: ${ctx.id}
Dashboard Title: ${ctx.title}

Dashboard Configuration:
\`\`\`json
${JSON.stringify(ctx.config, null, 2)}
\`\`\`

You can help edit this dashboard by providing instructions or generating updated JSON configurations.

## Available Custom Tools

You have access to specialized tools for this dashboard that allow you to quickly modify widgets without using the Edit tool:

### Widget-Specific Tools:
${widgetToolsList}

### General Tools:
- update_widget(widgetId, property, value) - Update any widget property
- add_widget(type, id, x, y, width, height, config?) - Add a new widget
- remove_widget(widgetId) - Remove a widget

**Usage**: When the user asks to update a widget value (like "set uptime to 99%"), prefer using the widget-specific tool (e.g., update_stat_uptime("99%")) instead of manually editing the JSON file. This is faster and more reliable.`;
        }

        // Extract dashboardContext from options before passing to query
        const { dashboardContext, ...queryOptions } = options;

        // Create dashboard-specific MCP tools if we have dashboard context
        let mcpServers = queryOptions.mcpServers || {};
        if (dashboardContext) {
          try {
            const dashboardMcpServer = createDashboardTools(
              {
                config: dashboardContext.config,
                filePath: dashboardContext.filePath
              },
              watchedPaths,
              broadcastDashboards // Pass the broadcast function so tools can update UI
            );
            // Add the dashboard MCP server to the servers config
            mcpServers[`dashboard-${dashboardContext.id}`] = dashboardMcpServer;
            console.log(`[Claude] Created ${Object.keys(mcpServers).length} MCP tools for dashboard ${dashboardContext.id}`);
          } catch (error) {
            console.error('[Claude] Error creating dashboard tools:', error);
          }
        }

        // Create query with streaming enabled
        // Use SDK's session resumption if we have a session ID
        const result = query({
          prompt: message,
          options: {
            model: queryOptions.model || 'claude-sonnet-4-5-20250929',
            cwd: queryOptions.cwd || process.cwd(),
            systemPrompt: systemPrompt || undefined,
            resume: existingSessionId || undefined, // Resume session if we have one
            mcpServers, // Include dashboard-specific tools
            permissionMode: queryOptions.permissionMode || 'bypassPermissions', // Default to bypass all permissions
            ...queryOptions
          }
        });

        // Store the query so we can interrupt it if needed
        activeChats.set(chatId, { query: result });

        // Stream messages back to renderer
        for await (const msg of result) {
          console.log(`[Claude] Received message type: ${msg.type}`);
          console.log(util.inspect(msg, { depth: null, colors: true }));

          // Capture the session ID from the init message
          if (msg.type === 'system' && msg.subtype === 'init' && msg.session_id) {
            console.log(`[Claude] Captured session ID: ${msg.session_id}`);
            sessionIds.set(chatId, msg.session_id);

            // Also persist it in conversation metadata
            // Find which widget this chat belongs to and update the conversation
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

          // Capture todo updates from the SDK (via TodoWrite tool)
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

                  // Send todo update to renderer
                  mainWindow?.webContents.send('todo-update', {
                    chatId,
                    todos: block.input.todos
                  });
                }
              });
            }
          }

          // Send each message chunk to the renderer
          mainWindow?.webContents.send('claude-chat-message', {
            chatId,
            responseId,
            message: msg
          });
        }

        console.log(`[Claude] Streaming complete for ${chatId}`);

        // Signal completion
        mainWindow?.webContents.send('claude-chat-complete', {
          chatId,
          responseId
        });

        // Clean up when done
        activeChats.delete(chatId);
      } catch (streamError) {
        console.error('[Claude] Streaming error:', streamError);
        mainWindow?.webContents.send('claude-chat-error', {
          chatId,
          responseId,
          error: streamError.message
        });
        activeChats.delete(chatId);
      }
    })();

    // Return immediately to not block the IPC call
    return { success: true, responseId };
  } catch (error) {
    console.error('[Claude] Handler error:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('claude-chat-interrupt', async (event, { chatId }) => {
  const chat = activeChats.get(chatId);
  if (chat?.query) {
    try {
      await chat.query.interrupt();
      activeChats.delete(chatId);
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }
  return { success: false, error: 'No active chat found' };
});

// Get todos for a specific conversation
ipcMain.handle('get-conversation-todos', async (event, { conversationId }) => {
  try {
    console.log(`[Claude] getTodos called for conversationId: ${conversationId}`);
    console.log(`[Claude] Available sessionIds keys:`, Array.from(sessionIds.keys()));
    console.log(`[Claude] Available conversationTodos keys:`, Array.from(conversationTodos.keys()));

    // First check if we have todos in memory
    if (conversationTodos.has(conversationId)) {
      const todos = conversationTodos.get(conversationId);
      console.log(`[Claude] Retrieved cached todos for ${conversationId}:`, todos.length);
      return { success: true, todos };
    }

    // If not in memory, check if we have a session ID to resume from
    const sessionId = sessionIds.get(conversationId);
    console.log(`[Claude] Found sessionId for ${conversationId}:`, sessionId);
    if (sessionId) {
      console.log(`[Claude] No cached todos, resuming session ${sessionId} to extract todos`);

      // Resume the session to replay messages and extract todos
      const result = query({
        prompt: '', // Empty prompt - we're just replaying
        options: {
          resume: sessionId,
          continue: false, // Don't continue, just replay
          cwd: process.cwd(),
          permissionMode: 'acceptEdits'
        }
      });

      let latestTodos = [];

      // Iterate through replayed messages to find TodoWrite
      for await (const msg of result) {
        if (msg.type === 'assistant' && msg.message?.content) {
          const content = msg.message.content;
          if (Array.isArray(content)) {
            content.forEach(block => {
              if (block.type === 'tool_use' && block.name === 'TodoWrite' && block.input?.todos) {
                console.log(`[Claude] Found TodoWrite in replay:`, block.input.todos.length, 'todos');
                latestTodos = block.input.todos;
              }
            });
          }
        }
      }

      // Cache the extracted todos
      if (latestTodos.length > 0) {
        conversationTodos.set(conversationId, latestTodos);
      }

      console.log(`[Claude] Extracted ${latestTodos.length} todos from session replay`);
      return { success: true, todos: latestTodos };
    }

    // No session ID, return empty
    console.log(`[Claude] No todos or session ID for ${conversationId}`);
    return { success: true, todos: [] };
  } catch (error) {
    console.error('[Claude] Error getting todos:', error);
    return { success: false, error: error.message, todos: [] };
  }
});

// Get chat messages for a specific chatId
ipcMain.handle('chat-get-messages', async (event, { chatId }) => {
  try {
    const messages = chatMessages.get(chatId) || [];
    const isStreaming = activeChats.has(chatId);
    console.log(`[Chat] Retrieved ${messages.length} messages for ${chatId}, streaming: ${isStreaming}`);
    console.log(`[Chat] Messages:`, messages);
    return { success: true, messages, isStreaming };
  } catch (error) {
    console.error('[Chat] Error getting messages:', error);
    return { success: false, error: error.message };
  }
});

// Save a message to chat history
ipcMain.handle('chat-save-message', async (event, { chatId, message }) => {
  try {
    if (!chatMessages.has(chatId)) {
      chatMessages.set(chatId, []);
    }

    console.log(`[Chat] Saving message for ${chatId}:`, message);
    chatMessages.get(chatId).push(message);

    // Save to disk after each message
    saveChatHistory();

    const allMessages = chatMessages.get(chatId);
    console.log(`[Chat] Saved message for ${chatId}, total: ${allMessages.length}`);
    console.log(`[Chat] All messages:`, allMessages);
    return { success: true };
  } catch (error) {
    console.error('[Chat] Error saving message:', error);
    return { success: false, error: error.message };
  }
});

// Clear chat history for a specific chatId
ipcMain.handle('chat-clear-messages', async (event, { chatId }) => {
  try {
    chatMessages.delete(chatId);
    saveChatHistory();
    console.log(`[Chat] Cleared messages for ${chatId}`);
    return { success: true };
  } catch (error) {
    console.error('[Chat] Error clearing messages:', error);
    return { success: false, error: error.message };
  }
});

// Get all conversations for a widget (dashboard + widget combo)
ipcMain.handle('conversation-list', async (event, { widgetKey }) => {
  try {
    const convos = conversations.get(widgetKey) || [];
    console.log(`[Conversations] Retrieved ${convos.length} conversations for ${widgetKey}`);
    return { success: true, conversations: convos };
  } catch (error) {
    console.error('[Conversations] Error listing conversations:', error);
    return { success: false, error: error.message };
  }
});

// Create a new conversation
ipcMain.handle('conversation-create', async (event, { widgetKey, title }) => {
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
    conversations.get(widgetKey).unshift(conversation); // Add to beginning
    saveConversations();

    console.log(`[Conversations] Created new conversation ${conversationId}`);
    return { success: true, conversation };
  } catch (error) {
    console.error('[Conversations] Error creating conversation:', error);
    return { success: false, error: error.message };
  }
});

// Update conversation metadata (e.g., update lastMessageAt)
ipcMain.handle('conversation-update', async (event, { widgetKey, conversationId, updates }) => {
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
  } catch (error) {
    console.error('[Conversations] Error updating conversation:', error);
    return { success: false, error: error.message };
  }
});

// Delete a conversation
ipcMain.handle('conversation-delete', async (event, { widgetKey, conversationId }) => {
  try {
    const convos = conversations.get(widgetKey);
    if (!convos) {
      return { success: false, error: 'Widget not found' };
    }

    const filtered = convos.filter(c => c.id !== conversationId);
    conversations.set(widgetKey, filtered);
    saveConversations();

    // Also delete messages for this conversation
    chatMessages.delete(conversationId);
    saveChatHistory();

    console.log(`[Conversations] Deleted conversation ${conversationId}`);
    return { success: true };
  } catch (error) {
    console.error('[Conversations] Error deleting conversation:', error);
    return { success: false, error: error.message };
  }
});

// Load data file for widgets
ipcMain.handle('load-data-file', async (event, filePath) => {
  try {
    // Resolve relative paths from the dashboard's directory
    // If path is absolute, use it directly
    let resolvedPath = filePath;

    // If it's a relative path, try to resolve it relative to each dashboard directory
    if (!path.isAbsolute(filePath)) {
      // Try to find the file relative to any watched dashboard directory
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
  } catch (error) {
    console.error('[DataLoader] Error loading data file:', error);
    return { success: false, error: error.message };
  }
});

app.whenReady().then(() => {
  const savedPaths = loadSavedPaths();
  savedPaths.forEach(watchFile);
  loadChatHistory(); // Load persisted chat messages
  loadConversations(); // Load conversation metadata
  createWindow();
});

app.on('window-all-closed', () => {
  // On macOS, apps typically stay open even when all windows are closed
  if (process.platform !== 'darwin') {
    // Clean up watchers on non-macOS platforms
    watchedPaths.forEach((entry) => entry.watcher.close());
    app.quit();
  }
});

// On macOS, recreate window when clicking dock icon while app is running
app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});
