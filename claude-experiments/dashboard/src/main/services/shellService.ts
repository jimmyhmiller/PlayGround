/**
 * Shell Service
 *
 * Provides shell command execution functionality that can be used by
 * both IPC handlers and StateStore commands.
 */

import { spawn, exec, ChildProcess } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

// Type for the events module
interface EventEmitter {
  emit(type: string, payload: unknown): void;
}

// Track running processes
const runningProcesses: Map<string, ChildProcess> = new Map();

// Process ID counter
let processIdCounter = 0;

// Event emitter reference
let events: EventEmitter | null = null;

/**
 * Initialize shell service with event emitter
 */
export function initShellService(eventEmitter: EventEmitter): void {
  events = eventEmitter;
}

/**
 * Get the event emitter
 */
export function getShellEvents(): EventEmitter | null {
  return events;
}

/**
 * Get the running processes map (for backward compatibility with IPC handlers)
 */
export function getRunningProcesses(): Map<string, ChildProcess> {
  return runningProcesses;
}

export interface ShellSpawnOptions {
  command: string;
  args?: string[];
  cwd?: string;
  env?: Record<string, string>;
}

export interface ShellSpawnResult {
  success: boolean;
  pid: string;
  error?: string;
}

/**
 * Spawn a background process
 * Returns a process ID that can be used to track output and kill the process
 */
export function shellSpawn(options: ShellSpawnOptions): ShellSpawnResult {
  const { command, args = [], cwd, env } = options;
  const pid = `proc-${++processIdCounter}`;

  try {
    const proc = spawn(command, args, {
      cwd: cwd ?? process.cwd(),
      env: env ? { ...process.env, ...env } : process.env,
      shell: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    runningProcesses.set(pid, proc);

    // Forward output via events
    proc.stdout?.on('data', (data: Buffer) => {
      events?.emit(`shell.stdout.${pid}`, { pid, data: data.toString() });
    });

    proc.stderr?.on('data', (data: Buffer) => {
      events?.emit(`shell.stderr.${pid}`, { pid, data: data.toString() });
    });

    proc.on('close', (code: number | null) => {
      runningProcesses.delete(pid);
      events?.emit(`shell.exit.${pid}`, { pid, code });
    });

    proc.on('error', (err: Error) => {
      runningProcesses.delete(pid);
      events?.emit(`shell.error.${pid}`, { pid, error: err.message });
    });

    return { success: true, pid };
  } catch (err) {
    return {
      success: false,
      pid,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

export interface ShellKillOptions {
  pid?: string;
  pattern?: string;
  signal?: 'SIGTERM' | 'SIGKILL' | 'SIGINT';
}

export interface ShellKillResult {
  success: boolean;
  killed: number;
  error?: string;
}

/**
 * Kill running processes
 * Can kill by PID or by pattern (uses pkill -f)
 */
export async function shellKill(options: ShellKillOptions): Promise<ShellKillResult> {
  const { pid, pattern, signal = 'SIGTERM' } = options;
  let killed = 0;

  // Kill by PID
  if (pid) {
    const proc = runningProcesses.get(pid);
    if (proc) {
      try {
        proc.kill(signal);
        runningProcesses.delete(pid);
        killed++;
      } catch (err) {
        return {
          success: false,
          killed,
          error: err instanceof Error ? err.message : String(err),
        };
      }
    }
  }

  // Kill by pattern using pkill
  if (pattern) {
    try {
      // First, try to kill any tracked processes that match the pattern
      for (const [id, proc] of Array.from(runningProcesses.entries())) {
        // Check if the process command matches the pattern
        const spawnArgs = (proc as unknown as { spawnargs?: string[] }).spawnargs;
        const cmdLine = spawnArgs ? spawnArgs.join(' ') : '';
        if (cmdLine.includes(pattern) || new RegExp(pattern).test(cmdLine)) {
          proc.kill(signal);
          runningProcesses.delete(id);
          killed++;
        }
      }

      // Also use pkill to kill system processes matching the pattern
      const signalFlag = signal === 'SIGKILL' ? '-9' : signal === 'SIGINT' ? '-2' : '-15';
      await execAsync(`pkill ${signalFlag} -f "${pattern.replace(/"/g, '\\"')}"`).catch(() => {
        // pkill returns non-zero if no processes matched, which is fine
      });

      // We don't know exactly how many pkill killed, but mark success
      return { success: true, killed };
    } catch (err) {
      // pkill returning non-zero is common (no processes matched)
      return { success: true, killed };
    }
  }

  if (!pid && !pattern) {
    return { success: false, killed: 0, error: 'Must specify either pid or pattern' };
  }

  return { success: true, killed };
}

export interface ShellExecOptions {
  command: string;
  args?: string[];
  cwd?: string;
  timeout?: number;
}

export interface ShellExecResult {
  success: boolean;
  stdout: string;
  stderr: string;
  exitCode: number;
  error?: string;
}

/**
 * Execute a command and wait for completion
 */
export async function shellExec(options: ShellExecOptions): Promise<ShellExecResult> {
  const { command, args = [], cwd, timeout = 30000 } = options;
  const fullCommand = args.length > 0 ? `${command} ${args.join(' ')}` : command;

  try {
    const { stdout, stderr } = await execAsync(fullCommand, {
      cwd: cwd ?? process.cwd(),
      timeout,
      maxBuffer: 10 * 1024 * 1024, // 10MB buffer
    });

    return {
      success: true,
      stdout,
      stderr,
      exitCode: 0,
    };
  } catch (err: unknown) {
    const error = err as { stdout?: string; stderr?: string; code?: number; message?: string };
    return {
      success: false,
      stdout: error.stdout ?? '',
      stderr: error.stderr ?? '',
      exitCode: error.code ?? 1,
      error: error.message,
    };
  }
}

export interface ShellIsRunningResult {
  running: boolean;
  pid?: string;
}

/**
 * Check if a process is running
 */
export function shellIsRunning(pid: string): ShellIsRunningResult {
  return { running: runningProcesses.has(pid), pid };
}

/**
 * List all running processes
 */
export function shellList(): { processes: Array<{ pid: string; running: boolean }> } {
  const processes = Array.from(runningProcesses.keys()).map((pid) => ({
    pid,
    running: true,
  }));
  return { processes };
}

/**
 * Kill all tracked processes (for cleanup)
 */
export function shellKillAll(): void {
  for (const [id, proc] of Array.from(runningProcesses.entries())) {
    try {
      proc.kill('SIGTERM');
    } catch {
      // Ignore errors during cleanup
    }
    runningProcesses.delete(id);
  }
}
