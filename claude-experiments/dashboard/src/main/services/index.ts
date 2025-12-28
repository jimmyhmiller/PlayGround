/**
 * Services Index
 *
 * Initializes and exports all backend services.
 */

import { ipcMain, IpcMainInvokeEvent } from 'electron';
import { FileWatcherService } from './fileWatcher';
import { GitService } from './gitService';
import { initEvaluationService, getEvaluationService, EvaluationRequest } from './evaluationService';

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
      language: 'javascript' | 'typescript' = 'javascript',
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
