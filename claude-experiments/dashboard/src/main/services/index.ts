/**
 * Services Index
 *
 * Initializes and exports all backend services.
 */

import { ipcMain, IpcMainInvokeEvent, BrowserWindow } from 'electron';
import { spawn, ChildProcess } from 'child_process';
import { FileWatcherService } from './fileWatcher';
import { GitService } from './gitService';
import { initEvaluationService, getEvaluationService, EvaluationRequest, ExecutorConfig } from './evaluationService';
import { initPipelineService, setupPipelineIPC, closePipelineService } from '../pipeline';
import { getACPServiceForWidget, removeACPServiceForWidget, shutdownAllACPServices, setAcpDebug, isAcpDebugEnabled } from './acpClientService';

// Track running processes
const runningProcesses: Map<string, ChildProcess> = new Map();

// Type for the events module
interface EventEmitter {
  emit(type: string, payload: unknown): void;
}

interface ServiceOptions {
  repoPath?: string;
}

interface Services {
  fileWatcher: FileWatcherService;
  gitService: GitService;
}

let fileWatcher: FileWatcherService | null = null;
let gitService: GitService | null = null;

/**
 * Initialize all services
 */
export function initServices(events: EventEmitter, options: ServiceOptions = {}): Services {
  const repoPath = options.repoPath ?? process.cwd();

  // Store events for ACP service creation
  globalEvents = events;

  // Initialize file watcher
  fileWatcher = new FileWatcherService(events);

  // Initialize git service
  gitService = new GitService(events, repoPath);

  // Initialize evaluation service with events
  initEvaluationService(events);

  // Initialize pipeline service
  initPipelineService(events as EventEmitter & { subscribe: (pattern: string, callback: (event: { type: string; payload: unknown }) => void) => () => void });

  // Note: ACP services are now created per-widget on demand, not at initialization

  console.log('[services] Initialized');

  return { fileWatcher, gitService };
}

/**
 * Setup IPC handlers for services
 */
export function setupServiceIPC(): void {
  // File operations
  ipcMain.handle('file:load', (_event: IpcMainInvokeEvent, filePath: string) => {
    return fileWatcher!.loadFile(filePath);
  });

  ipcMain.handle('file:watch', (_event: IpcMainInvokeEvent, watchPath: string) => {
    fileWatcher!.watch(watchPath);
    return { success: true, path: watchPath };
  });

  ipcMain.handle('file:unwatch', (_event: IpcMainInvokeEvent, watchPath: string) => {
    fileWatcher!.unwatch(watchPath);
    return { success: true };
  });

  ipcMain.handle('file:getWatched', () => {
    return fileWatcher!.getWatchedPaths();
  });

  // Git operations
  ipcMain.handle('git:refresh', async () => {
    return await gitService!.refresh();
  });

  ipcMain.handle('git:status', async () => {
    return await gitService!.getStatus();
  });

  ipcMain.handle('git:diff', async (_event: IpcMainInvokeEvent, filePath?: string) => {
    return await gitService!.getDiff(filePath ?? null);
  });

  ipcMain.handle('git:diffStaged', async (_event: IpcMainInvokeEvent, filePath?: string) => {
    return await gitService!.getStagedDiff(filePath ?? null);
  });

  ipcMain.handle('git:startPolling', (_event: IpcMainInvokeEvent, intervalMs?: number) => {
    gitService!.startPolling(intervalMs);
    return { success: true };
  });

  ipcMain.handle('git:stopPolling', () => {
    gitService!.stopPolling();
    return { success: true };
  });

  ipcMain.handle('git:stage', async (_event: IpcMainInvokeEvent, filePath: string) => {
    await gitService!.stageFile(filePath);
    return { success: true };
  });

  ipcMain.handle('git:unstage', async (_event: IpcMainInvokeEvent, filePath: string) => {
    await gitService!.unstageFile(filePath);
    return { success: true };
  });

  // Evaluation operations
  ipcMain.handle(
    'eval:execute',
    async (
      _event: IpcMainInvokeEvent,
      code: string,
      language: string = 'javascript',
      context?: Record<string, unknown>
    ) => {
      const evalService = getEvaluationService();
      return await evalService.execute({
        id: `eval-${Date.now()}`,
        code,
        language,
        context,
      });
    }
  );

  ipcMain.handle(
    'eval:batch',
    async (_event: IpcMainInvokeEvent, requests: EvaluationRequest[]) => {
      const evalService = getEvaluationService();
      return await evalService.batch(requests);
    }
  );

  // Executor registration
  ipcMain.handle(
    'eval:registerExecutor',
    (_event: IpcMainInvokeEvent, config: ExecutorConfig) => {
      const evalService = getEvaluationService();
      evalService.registerExecutor(config);
      return { success: true, language: config.language };
    }
  );

  ipcMain.handle(
    'eval:unregisterExecutor',
    (_event: IpcMainInvokeEvent, language: string) => {
      const evalService = getEvaluationService();
      evalService.unregisterExecutor(language);
      return { success: true };
    }
  );

  ipcMain.handle('eval:getExecutors', () => {
    const evalService = getEvaluationService();
    return evalService.getExecutors();
  });

  // Shell command operations
  ipcMain.handle(
    'shell:spawn',
    async (
      _event: IpcMainInvokeEvent,
      id: string,
      command: string,
      args: string[] = [],
      options: { cwd?: string; env?: Record<string, string> } = {}
    ) => {
      // Kill existing process with same ID if any
      const existing = runningProcesses.get(id);
      if (existing) {
        existing.kill();
        runningProcesses.delete(id);
      }

      const cwd = options.cwd ?? process.cwd();
      const env = { ...process.env, ...options.env };

      const proc = spawn(command, args, {
        cwd,
        env,
        shell: true,
        stdio: ['ignore', 'pipe', 'pipe'],
      });

      runningProcesses.set(id, proc);

      // Forward output via events
      const events = fileWatcher ? (fileWatcher as unknown as { events: EventEmitter }).events : null;

      proc.stdout?.on('data', (data: Buffer) => {
        events?.emit(`shell.stdout.${id}`, { id, data: data.toString() });
      });

      proc.stderr?.on('data', (data: Buffer) => {
        events?.emit(`shell.stderr.${id}`, { id, data: data.toString() });
      });

      proc.on('close', (code: number | null) => {
        runningProcesses.delete(id);
        events?.emit(`shell.exit.${id}`, { id, code });
      });

      proc.on('error', (err: Error) => {
        runningProcesses.delete(id);
        events?.emit(`shell.error.${id}`, { id, error: err.message });
      });

      return { success: true, id, pid: proc.pid };
    }
  );

  ipcMain.handle('shell:kill', (_event: IpcMainInvokeEvent, id: string) => {
    const proc = runningProcesses.get(id);
    if (proc) {
      proc.kill();
      runningProcesses.delete(id);
      return { success: true, id };
    }
    return { success: false, error: 'Process not found' };
  });

  ipcMain.handle('shell:isRunning', (_event: IpcMainInvokeEvent, id: string) => {
    return { running: runningProcesses.has(id) };
  });

  // Setup pipeline IPC handlers
  setupPipelineIPC();

  // Setup ACP IPC handlers
  setupACPIPC();

  console.log('[services] IPC handlers registered');
}

// Pending permission requests for ACP
// Response format: { outcome: { outcome: 'selected', optionId: string } | { outcome: 'cancelled' } }
const pendingPermissions: Map<string, {
  resolve: (response: { outcome: { outcome: 'selected'; optionId: string } | { outcome: 'cancelled' } }) => void;
  reject: (error: Error) => void;
}> = new Map();

// Get events emitter for ACP service creation
let globalEvents: EventEmitter | null = null;

// Helper to get widget ID from event
function getWidgetId(event: IpcMainInvokeEvent): string {
  // Use the webContents ID as the widget identifier
  return `widget-${event.sender.id}`;
}

// Helper to get ACP service for the requesting widget
function getACPService(event: IpcMainInvokeEvent): any {
  if (!globalEvents) {
    throw new Error('Events not initialized');
  }
  const widgetId = getWidgetId(event);
  return getACPServiceForWidget(widgetId, globalEvents);
}

/**
 * Setup ACP IPC handlers
 */
function setupACPIPC(): void {
  // Permission callback setup - this needs to be per-service
  // We'll set it up when services are created
  const setupPermissionCallback = (acpService: any) => {
    acpService.setPermissionCallback(async (request: unknown): Promise<{ outcome: { outcome: 'selected'; optionId: string } | { outcome: 'cancelled' } }> => {
    const requestId = `perm-${Date.now()}-${Math.random().toString(36).slice(2)}`;

    // Forward to all renderer windows
    const windows = BrowserWindow.getAllWindows();
    for (const win of windows) {
      win.webContents.send('acp:permissionRequest', { ...request, requestId });
    }

    // Wait for response from renderer
    return new Promise((resolve, reject) => {
      pendingPermissions.set(requestId, { resolve, reject });

      // Timeout after 5 minutes - cancel the request
      setTimeout(() => {
        if (pendingPermissions.has(requestId)) {
          pendingPermissions.delete(requestId);
          resolve({ outcome: { outcome: 'cancelled' } });
        }
      }, 5 * 60 * 1000);
    });
    });
  };

  // Subscribe to session updates and forward to renderer
  // This is done via the event system - the service emits 'acp.session.update' events
  // which get pushed to the renderer via the existing events:push channel

  // Spawn the agent
  ipcMain.handle('acp:spawn', async (event: IpcMainInvokeEvent, cwd?: string) => {
    const service = getACPService(event);
    setupPermissionCallback(service);
    await service.spawn(cwd);
  });

  // Initialize connection
  ipcMain.handle('acp:initialize', async (event: IpcMainInvokeEvent) => {
    const service = getACPService(event);
    await service.initialize();
  });

  // Create new session
  ipcMain.handle('acp:newSession', async (event: IpcMainInvokeEvent, cwd?: string, mcpServers?: unknown[], force?: boolean) => {
    const service = getACPService(event);
    return await service.newSession(cwd, mcpServers, force);
  });

  // Resume existing session
  ipcMain.handle('acp:resumeSession', async (event: IpcMainInvokeEvent, sessionId: string, cwd?: string) => {
    const service = getACPService(event);
    return await service.resumeSession(sessionId, cwd);
  });

  // Send prompt
  ipcMain.handle('acp:prompt', async (event: IpcMainInvokeEvent, sessionId: string, content: string | unknown[]) => {
    const service = getACPService(event);
    return await service.prompt(sessionId, content as string | Array<{ type: string; text?: string }>);
  });

  // Cancel prompt
  ipcMain.handle('acp:cancel', async (event: IpcMainInvokeEvent, sessionId: string) => {
    const service = getACPService(event);
    await service.cancel(sessionId);
  });

  // Set mode
  ipcMain.handle('acp:setMode', async (event: IpcMainInvokeEvent, sessionId: string, modeId: string) => {
    const service = getACPService(event);
    await service.setMode(sessionId, modeId);
  });

  // Shutdown
  ipcMain.handle('acp:shutdown', async (event: IpcMainInvokeEvent) => {
    const widgetId = getWidgetId(event);
    await removeACPServiceForWidget(widgetId);
  });

  // Check connection
  ipcMain.handle('acp:isConnected', (event: IpcMainInvokeEvent) => {
    const service = getACPService(event);
    return service.isConnected();
  });

  // Respond to permission request
  // optionId should be one of: 'allow', 'allow_always', 'reject', or 'cancelled'
  ipcMain.handle('acp:respondPermission', (_event: IpcMainInvokeEvent, requestId: string, optionId: string) => {
    const pending = pendingPermissions.get(requestId);
    if (pending) {
      pendingPermissions.delete(requestId);
      if (optionId === 'cancelled') {
        pending.resolve({ outcome: { outcome: 'cancelled' } });
      } else {
        pending.resolve({ outcome: { outcome: 'selected', optionId } });
      }
    }
  });

  // Load session history from Claude's local files
  ipcMain.handle('acp:loadSessionHistory', async (event: IpcMainInvokeEvent, sessionId: string, cwd?: string) => {
    const service = getACPService(event);
    return await service.loadSessionHistory(sessionId, cwd);
  });

  // Toggle ACP debug logging (logs all JSON-RPC messages)
  ipcMain.handle('acp:setDebug', (_event: IpcMainInvokeEvent, enabled: boolean) => {
    setAcpDebug(enabled);
    return enabled;
  });

  // Get current ACP debug state
  ipcMain.handle('acp:isDebugEnabled', () => {
    return isAcpDebugEnabled();
  });

  console.log('[acp] IPC handlers registered');
}

/**
 * Cleanup services
 */
export async function closeServices(): Promise<void> {
  if (fileWatcher) {
    fileWatcher.close();
  }
  if (gitService) {
    gitService.stopPolling();
  }
  closePipelineService();

  // Shutdown all ACP services
  await shutdownAllACPServices();

  console.log('[services] Closed');
}

export function getFileWatcher(): FileWatcherService | null {
  return fileWatcher;
}

export function getGitService(): GitService | null {
  return gitService;
}
