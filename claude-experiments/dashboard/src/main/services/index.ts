/**
 * Services Index
 *
 * Initializes and exports all backend services.
 */

import { ipcMain, IpcMainInvokeEvent } from 'electron';
import { spawn, ChildProcess } from 'child_process';
import { FileWatcherService } from './fileWatcher';
import { GitService } from './gitService';
import { initEvaluationService, getEvaluationService, EvaluationRequest, ExecutorConfig } from './evaluationService';

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

  // Initialize file watcher
  fileWatcher = new FileWatcherService(events);

  // Initialize git service
  gitService = new GitService(events, repoPath);

  // Initialize evaluation service with events
  initEvaluationService(events);

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

  console.log('[services] IPC handlers registered');
}

/**
 * Cleanup services
 */
export function closeServices(): void {
  if (fileWatcher) {
    fileWatcher.close();
  }
  if (gitService) {
    gitService.stopPolling();
  }
  console.log('[services] Closed');
}

export function getFileWatcher(): FileWatcherService | null {
  return fileWatcher;
}

export function getGitService(): GitService | null {
  return gitService;
}
