/**
 * Services Index
 *
 * Initializes and exports all backend services.
 */

const { ipcMain } = require('electron');
const { FileWatcherService } = require('./fileWatcher');
const { GitService } = require('./gitService');

let fileWatcher = null;
let gitService = null;

/**
 * Initialize all services
 */
function initServices(events, options = {}) {
  const repoPath = options.repoPath || process.cwd();

  // Initialize file watcher
  fileWatcher = new FileWatcherService(events);

  // Initialize git service
  gitService = new GitService(events, repoPath);

  console.log('[services] Initialized');

  return { fileWatcher, gitService };
}

/**
 * Setup IPC handlers for services
 */
function setupServiceIPC() {
  // File operations
  ipcMain.handle('file:load', (event, filePath) => {
    return fileWatcher.loadFile(filePath);
  });

  ipcMain.handle('file:watch', (event, watchPath) => {
    fileWatcher.watch(watchPath);
    return { success: true, path: watchPath };
  });

  ipcMain.handle('file:unwatch', (event, watchPath) => {
    fileWatcher.unwatch(watchPath);
    return { success: true };
  });

  ipcMain.handle('file:getWatched', () => {
    return fileWatcher.getWatchedPaths();
  });

  // Git operations
  ipcMain.handle('git:refresh', async () => {
    return await gitService.refresh();
  });

  ipcMain.handle('git:status', async () => {
    return await gitService.getStatus();
  });

  ipcMain.handle('git:diff', async (event, filePath) => {
    return await gitService.getDiff(filePath);
  });

  ipcMain.handle('git:diffStaged', async (event, filePath) => {
    return await gitService.getStagedDiff(filePath);
  });

  ipcMain.handle('git:startPolling', (event, intervalMs) => {
    gitService.startPolling(intervalMs);
    return { success: true };
  });

  ipcMain.handle('git:stopPolling', () => {
    gitService.stopPolling();
    return { success: true };
  });

  ipcMain.handle('git:stage', async (event, filePath) => {
    await gitService.stageFile(filePath);
    return { success: true };
  });

  ipcMain.handle('git:unstage', async (event, filePath) => {
    await gitService.unstageFile(filePath);
    return { success: true };
  });

  console.log('[services] IPC handlers registered');
}

/**
 * Cleanup services
 */
function closeServices() {
  if (fileWatcher) {
    fileWatcher.close();
  }
  if (gitService) {
    gitService.stopPolling();
  }
  console.log('[services] Closed');
}

module.exports = {
  initServices,
  setupServiceIPC,
  closeServices,
  getFileWatcher: () => fileWatcher,
  getGitService: () => gitService,
};
