/**
 * Git Service
 *
 * Monitors git repository and emits events for changes.
 *
 * Events emitted:
 *   - git.status.changed   { files: [{ path, status }], branch }
 *   - git.diff.updated     { diff, filePath? }
 *   - git.branch.changed   { branch, previous }
 *   - git.commit.created   { hash, message, author }
 */

import { exec } from 'child_process';
import * as path from 'path';

// Type for the events module
interface EventEmitter {
  emit(type: string, payload: unknown): void;
}

interface GitFileStatus {
  status: string;
  path: string;
}

interface GitStatus {
  files: GitFileStatus[];
  branch: string | null;
}

interface RefreshResult {
  status: GitStatus;
  hasChanges: boolean;
}

export class GitService {
  private events: EventEmitter;
  private repoPath: string;
  private pollInterval: ReturnType<typeof setInterval> | null = null;
  private lastStatus: GitStatus | null = null;
  private lastBranch: string | null = null;
  private lastDiff: string | null = null;

  constructor(eventEmitter: EventEmitter, repoPath: string = process.cwd()) {
    this.events = eventEmitter;
    this.repoPath = path.resolve(repoPath);
  }

  /**
   * Execute a git command
   */
  private _exec(command: string): Promise<string> {
    return new Promise((resolve, reject) => {
      exec(command, { cwd: this.repoPath }, (err, stdout, stderr) => {
        if (err) {
          reject(new Error(stderr || err.message));
        } else {
          resolve(stdout.trim());
        }
      });
    });
  }

  /**
   * Get current branch name
   */
  async getBranch(): Promise<string | null> {
    try {
      return await this._exec('git rev-parse --abbrev-ref HEAD');
    } catch {
      return null;
    }
  }

  /**
   * Get git status as structured data
   */
  async getStatus(): Promise<GitStatus> {
    try {
      const output = await this._exec('git status --porcelain');
      const files: GitFileStatus[] = output
        .split('\n')
        .filter(Boolean)
        .map((line) => ({
          status: line.substring(0, 2).trim(),
          path: line.substring(3),
        }));

      const branch = await this.getBranch();
      return { files, branch };
    } catch {
      return { files: [], branch: null };
    }
  }

  /**
   * Get diff for all changes or a specific file
   */
  async getDiff(filePath: string | null = null): Promise<string> {
    try {
      const cmd = filePath
        ? `git diff -- "${filePath}"`
        : 'git diff';
      return await this._exec(cmd);
    } catch {
      return '';
    }
  }

  /**
   * Get diff for staged changes
   */
  async getStagedDiff(filePath: string | null = null): Promise<string> {
    try {
      const cmd = filePath
        ? `git diff --cached -- "${filePath}"`
        : 'git diff --cached';
      return await this._exec(cmd);
    } catch {
      return '';
    }
  }

  /**
   * Check if two status objects are equal
   */
  private _statusEqual(a: GitStatus | null, b: GitStatus | null): boolean {
    if (!a || !b) return false;
    if (a.branch !== b.branch) return false;
    if (a.files.length !== b.files.length) return false;
    return a.files.every((file, i) => {
      const bFile = b.files[i];
      return bFile && file.path === bFile.path && file.status === bFile.status;
    });
  }

  /**
   * Refresh and emit current status (only if changed)
   */
  async refresh(): Promise<RefreshResult> {
    const status = await this.getStatus();
    const diff = await this.getDiff();

    let hasChanges = false;

    // Check if branch changed
    if (this.lastBranch !== null && status.branch !== this.lastBranch) {
      this.events.emit('git.branch.changed', {
        branch: status.branch,
        previous: this.lastBranch,
      });
      hasChanges = true;
    }
    this.lastBranch = status.branch;

    // Check if status changed
    if (!this._statusEqual(status, this.lastStatus)) {
      this.events.emit('git.status.changed', {
        files: status.files,
        branch: status.branch,
      });
      hasChanges = true;
    }

    // Check if diff changed
    if (diff !== this.lastDiff) {
      this.events.emit('git.diff.updated', {
        diff,
      });
      hasChanges = true;
    }

    this.lastStatus = status;
    this.lastDiff = diff;

    return { status, hasChanges };
  }

  /**
   * Get diff for a specific file and emit
   */
  async refreshFileDiff(filePath: string): Promise<string> {
    const diff = await this.getDiff(filePath);

    this.events.emit('git.diff.updated', {
      diff,
      filePath,
    });

    return diff;
  }

  /**
   * Start polling for changes
   */
  startPolling(intervalMs: number = 2000): void {
    if (this.pollInterval) {
      console.log('[gitService] Already polling');
      return;
    }

    // Initial refresh
    this.refresh();

    this.pollInterval = setInterval(() => {
      this.refresh();
    }, intervalMs);

    console.log(`[gitService] Polling every ${intervalMs}ms`);
  }

  /**
   * Stop polling
   */
  stopPolling(): void {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
      console.log('[gitService] Polling stopped');
    }
  }

  /**
   * Stage a file
   */
  async stageFile(filePath: string): Promise<void> {
    try {
      await this._exec(`git add "${filePath}"`);
      this.events.emit('git.file.staged', { filePath });
      await this.refresh();
    } catch (err) {
      const error = err instanceof Error ? err.message : String(err);
      this.events.emit('git.error', { error, filePath });
    }
  }

  /**
   * Unstage a file
   */
  async unstageFile(filePath: string): Promise<void> {
    try {
      await this._exec(`git reset HEAD "${filePath}"`);
      this.events.emit('git.file.unstaged', { filePath });
      await this.refresh();
    } catch (err) {
      const error = err instanceof Error ? err.message : String(err);
      this.events.emit('git.error', { error, filePath });
    }
  }
}
