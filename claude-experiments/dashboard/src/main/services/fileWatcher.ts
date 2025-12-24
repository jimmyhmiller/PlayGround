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

import * as fs from 'fs';
import * as path from 'path';
import type { FSWatcher } from 'chokidar';

// Type for the events module
interface EventEmitter {
  emit(type: string, payload: unknown): void;
}

// Dynamic import for ESM-only chokidar
type ChokidarModule = typeof import('chokidar');
let chokidar: Pick<ChokidarModule, 'watch' | 'FSWatcher'> | null = null;
const chokidarReady = import('chokidar').then((mod) => {
  chokidar = mod.default ?? mod;
});

interface WatchOptions {
  ignoreInitial?: boolean;
  [key: string]: unknown;
}

interface LoadResult {
  success: boolean;
  content?: string;
  error?: string;
}

export class FileWatcherService {
  private events: EventEmitter;
  private watchers: Map<string, FSWatcher> = new Map();

  constructor(eventEmitter: EventEmitter) {
    this.events = eventEmitter;
  }

  /**
   * Load a file and emit its content
   */
  loadFile(filePath: string): LoadResult {
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
      const error = err instanceof Error ? err.message : String(err);
      this.events.emit('file.error', {
        filePath,
        error,
      });
      return { success: false, error };
    }
  }

  /**
   * Watch a file or directory for changes
   */
  async watch(watchPath: string, options: WatchOptions = {}): Promise<void> {
    // Ensure chokidar is loaded
    await chokidarReady;
    if (!chokidar) {
      throw new Error('Chokidar failed to load');
    }

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

    watcher.on('add', (filePath: string) => {
      this.events.emit('file.created', { filePath });
    });

    watcher.on('change', (filePath: string) => {
      try {
        const content = fs.readFileSync(filePath, 'utf-8');
        this.events.emit('file.content.changed', {
          filePath,
          content,
          size: content.length,
        });
      } catch (err) {
        const error = err instanceof Error ? err.message : String(err);
        this.events.emit('file.error', {
          filePath,
          error,
        });
      }
    });

    watcher.on('unlink', (filePath: string) => {
      this.events.emit('file.deleted', { filePath });
    });

    watcher.on('error', (err: unknown) => {
      this.events.emit('file.error', {
        path: absolutePath,
        error: err instanceof Error ? err.message : String(err),
      });
    });

    this.watchers.set(absolutePath, watcher);

    this.events.emit('file.watch.started', { path: absolutePath });
    console.log(`[fileWatcher] Watching: ${absolutePath}`);
  }

  /**
   * Stop watching a path
   */
  async unwatch(watchPath: string): Promise<void> {
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
  async close(): Promise<void> {
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
  getWatchedPaths(): string[] {
    return Array.from(this.watchers.keys());
  }
}
