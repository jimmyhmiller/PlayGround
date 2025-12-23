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

const { exec } = require('child_process');
const path = require('path');

class GitService {
  constructor(eventEmitter, repoPath = process.cwd()) {
    this.events = eventEmitter;
    this.repoPath = path.resolve(repoPath);
    this.pollInterval = null;
    this.lastStatus = null;
    this.lastBranch = null;
    this.lastDiff = null;
  }

  /**
   * Execute a git command
   */
  _exec(command) {
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
  async getBranch() {
    try {
      return await this._exec('git rev-parse --abbrev-ref HEAD');
    } catch {
      return null;
    }
  }

  /**
   * Get git status as structured data
   */
  async getStatus() {
    try {
      const output = await this._exec('git status --porcelain');
      const files = output
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
  async getDiff(filePath = null) {
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
  async getStagedDiff(filePath = null) {
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
  _statusEqual(a, b) {
    if (!a || !b) return false;
    if (a.branch !== b.branch) return false;
    if (a.files.length !== b.files.length) return false;
    return a.files.every((file, i) =>
      file.path === b.files[i].path && file.status === b.files[i].status
    );
  }

  /**
   * Refresh and emit current status (only if changed)
   */
  async refresh() {
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
  async refreshFileDiff(filePath) {
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
  startPolling(intervalMs = 2000) {
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
  stopPolling() {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
      console.log('[gitService] Polling stopped');
    }
  }

  /**
   * Stage a file
   */
  async stageFile(filePath) {
    try {
      await this._exec(`git add "${filePath}"`);
      this.events.emit('git.file.staged', { filePath });
      await this.refresh();
    } catch (err) {
      this.events.emit('git.error', { error: err.message, filePath });
    }
  }

  /**
   * Unstage a file
   */
  async unstageFile(filePath) {
    try {
      await this._exec(`git reset HEAD "${filePath}"`);
      this.events.emit('git.file.unstaged', { filePath });
      await this.refresh();
    } catch (err) {
      this.events.emit('git.error', { error: err.message, filePath });
    }
  }
}

module.exports = { GitService };
