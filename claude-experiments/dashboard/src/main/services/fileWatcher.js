/**
 * File Watcher Service
 *
 * Watches files/directories and emits events when they change.
 *
 * Events emitted:
 *   - file.content.loaded   { filePath, content }
 *   - file.content.changed  { filePath, content }
 *   - file.created          { filePath }
 *   - file.deleted          { filePath }
 *   - file.watch.started    { path }
 *   - file.watch.stopped    { path }
 */

const fs = require('fs');
const path = require('path');

// Dynamic import for ESM-only chokidar
let chokidar = null;
const chokidarReady = import('chokidar').then((mod) => {
  chokidar = mod.default || mod;
});

class FileWatcherService {
  constructor(eventEmitter) {
    this.events = eventEmitter;
    this.watchers = new Map(); // path -> chokidar instance
  }

  /**
   * Load a file and emit its content
   */
  loadFile(filePath) {
    try {
      const absolutePath = path.resolve(filePath);
      const content = fs.readFileSync(absolutePath, 'utf-8');

      this.events.emit('file.content.loaded', {
        filePath: absolutePath,
        content,
        size: content.length,
      });

      return { success: true, content };
    } catch (err) {
      this.events.emit('file.error', {
        filePath,
        error: err.message,
      });
      return { success: false, error: err.message };
    }
  }

  /**
   * Watch a file or directory for changes
   */
  async watch(watchPath, options = {}) {
    // Ensure chokidar is loaded
    await chokidarReady;

    const absolutePath = path.resolve(watchPath);

    if (this.watchers.has(absolutePath)) {
      console.log(`[fileWatcher] Already watching: ${absolutePath}`);
      return;
    }

    const watcher = chokidar.watch(absolutePath, {
      persistent: true,
      ignoreInitial: options.ignoreInitial !== false,
      ...options,
    });

    watcher.on('add', (filePath) => {
      this.events.emit('file.created', { filePath });
    });

    watcher.on('change', (filePath) => {
      try {
        const content = fs.readFileSync(filePath, 'utf-8');
        this.events.emit('file.content.changed', {
          filePath,
          content,
          size: content.length,
        });
      } catch (err) {
        this.events.emit('file.error', {
          filePath,
          error: err.message,
        });
      }
    });

    watcher.on('unlink', (filePath) => {
      this.events.emit('file.deleted', { filePath });
    });

    watcher.on('error', (err) => {
      this.events.emit('file.error', {
        path: absolutePath,
        error: err.message,
      });
    });

    this.watchers.set(absolutePath, watcher);

    this.events.emit('file.watch.started', { path: absolutePath });
    console.log(`[fileWatcher] Watching: ${absolutePath}`);
  }

  /**
   * Stop watching a path
   */
  async unwatch(watchPath) {
    const absolutePath = path.resolve(watchPath);
    const watcher = this.watchers.get(absolutePath);

    if (watcher) {
      await watcher.close();
      this.watchers.delete(absolutePath);
      this.events.emit('file.watch.stopped', { path: absolutePath });
      console.log(`[fileWatcher] Stopped watching: ${absolutePath}`);
    }
  }

  /**
   * Stop all watchers
   */
  async close() {
    for (const [watchPath, watcher] of this.watchers) {
      await watcher.close();
      this.events.emit('file.watch.stopped', { path: watchPath });
    }
    this.watchers.clear();
    console.log('[fileWatcher] All watchers closed');
  }

  /**
   * Get list of watched paths
   */
  getWatchedPaths() {
    return Array.from(this.watchers.keys());
  }
}

module.exports = { FileWatcherService };
