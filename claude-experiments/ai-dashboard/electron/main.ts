import { app, BrowserWindow, ipcMain } from 'electron';
import * as path from 'path';
import { StyleAgent } from './agents/style-agent';
import { AgentBridge } from './bridge/agent-bridge';
import { ProjectStyleStorage } from './storage/project-styles';

let mainWindow: BrowserWindow | null = null;
let styleAgent: StyleAgent;
let agentBridge: AgentBridge;
let styleStorage: ProjectStyleStorage;

// Check if running in development mode
const isDev = !app.isPackaged;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    backgroundColor: '#000000',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // Load the app
  if (isDev) {
    // In development, load from Vite dev server
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    // In production, load from built files
    mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(async () => {
  // Initialize storage
  styleStorage = new ProjectStyleStorage();

  // Initialize style agent (in-process)
  styleAgent = new StyleAgent();
  await styleAgent.initialize();

  // Initialize agent bridge for external agents
  agentBridge = new AgentBridge();
  await agentBridge.start();

  // Set up IPC handlers
  setupIPC();

  createWindow();

  const bridgePort = agentBridge.getPort();
  console.log('[Main] Dashboard initialized');
  console.log('[Main] Style Agent ready (in-process)');
  console.log(`[Main] Agent Bridge listening on ws://localhost:${bridgePort}`);
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

function setupIPC() {
  // Request style generation from in-process style agent (streaming)
  ipcMain.on('style:generate', (event, request) => {
    console.log('[Main] Style generation requested (streaming):', request.prompt);

    styleAgent.generateStyleStreaming(
      request,
      // onChunk: send each chunk to renderer
      (chunk: string) => {
        console.log('[Main] Sending chunk to renderer, length:', chunk.length);
        event.sender.send('style:chunk', chunk);
      },
      // onComplete: send final style
      (style) => {
        console.log('[Main] Style generation complete, sending to renderer');
        event.sender.send('style:complete', style);
      },
      // onError: send error
      (error: Error) => {
        console.error('[Main] Style generation error:', error.message);
        event.sender.send('style:error', error.message);
      }
    );
  });

  // Get current style
  ipcMain.handle('style:getCurrent', async () => {
    return styleAgent.getCurrentStyle();
  });

  // Project style storage
  ipcMain.handle('style:setProjectStyle', async (_event, projectId: string, style: any) => {
    styleStorage.setProjectStyle(projectId, style);
    return { success: true };
  });

  ipcMain.handle('style:getProjectStyle', async (_event, projectId: string) => {
    return styleStorage.getProjectStyle(projectId);
  });

  ipcMain.handle('style:getAllStyles', async () => {
    return styleStorage.getAllStyles();
  });

  // Forward messages to external agents
  ipcMain.handle('agent:send', async (_event, agentId, message) => {
    return await agentBridge.sendToAgent(agentId, message);
  });

  // Get registered components from external agents
  ipcMain.handle('agent:getComponents', async () => {
    return agentBridge.getRegisteredComponents();
  });

  // Get data from external agents
  ipcMain.handle('agent:getData', async (_event, dataSourceId) => {
    return agentBridge.getData(dataSourceId);
  });

  // Listen for external agent registrations and forward to renderer
  agentBridge.on('registration', (message) => {
    if (mainWindow) {
      mainWindow.webContents.send('agent:registration', message);
    }
  });

  // Listen for data updates from external agents
  agentBridge.on('data-update', (message) => {
    if (mainWindow) {
      mainWindow.webContents.send('agent:data-update', message);
    }
  });
}
