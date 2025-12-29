/**
 * Evaluation Service
 *
 * Fully event-driven code execution service.
 * Emits eval.request events and waits for eval.result responses.
 *
 * Supports:
 *   - Built-in JavaScript executor (vm module)
 *   - Subprocess executors: register a command, code goes to stdin, result from stdout
 *
 * Events:
 *   - eval.request  { id, code, language, context?, timeout? }
 *   - eval.result   { id, success, value?, displayValue, type, executionTimeMs, error?, language }
 *   - eval.executor.register { language, command, args? }
 */

import * as vm from 'vm';
import { spawn, ChildProcess } from 'child_process';

// Type for the events module
interface EventEmitter {
  emit(type: string, payload: unknown): void;
  subscribe(pattern: string, callback: (event: { type: string; payload: unknown }) => void): () => void;
}

export interface EvaluationRequest {
  id: string;
  code: string;
  language: string;
  context?: Record<string, unknown>;
  timeout?: number;
}

export interface EvaluationResult {
  id: string;
  success: boolean;
  value?: unknown;
  displayValue: string;
  type: string;
  executionTimeMs: number;
  error?: string;
  language?: string;
}

/**
 * Configuration for a subprocess-based executor
 */
export interface ExecutorConfig {
  language: string;
  command: string;
  args?: string[];
  /** If true, keep the process running and reuse it */
  persistent?: boolean;
  /** Working directory for the process */
  cwd?: string;
}

/**
 * A running persistent executor process
 */
interface PersistentExecutor {
  process: ChildProcess;
  config: ExecutorConfig;
}

/**
 * Format a value for display
 */
function formatValue(value: unknown): string {
  if (value === undefined) return 'undefined';
  if (value === null) return 'null';
  if (typeof value === 'function') {
    return `[Function: ${value.name || 'anonymous'}]`;
  }
  if (typeof value === 'symbol') {
    return value.toString();
  }
  if (value instanceof Error) {
    return `${value.name}: ${value.message}`;
  }
  if (Array.isArray(value)) {
    if (value.length <= 10) {
      return JSON.stringify(value);
    }
    return `[Array(${value.length})]`;
  }
  if (typeof value === 'object') {
    try {
      const json = JSON.stringify(value);
      if (json.length <= 100) return json;
      return `{${Object.keys(value).slice(0, 5).join(', ')}${Object.keys(value).length > 5 ? ', ...' : ''}}`;
    } catch {
      return '[Object]';
    }
  }
  return String(value);
}

/**
 * Get the type of a value for display
 */
function getValueType(value: unknown): string {
  if (value === null) return 'null';
  if (value === undefined) return 'undefined';
  if (Array.isArray(value)) return `Array(${value.length})`;
  if (value instanceof Error) return value.constructor.name;
  if (typeof value === 'object') return value.constructor?.name || 'Object';
  return typeof value;
}

/**
 * Create a sandboxed context for JavaScript evaluation
 */
function createSandboxContext(userContext: Record<string, unknown> = {}): vm.Context {
  const sandbox: Record<string, unknown> = {
    // Safe globals
    console: {
      log: (...args: unknown[]) => args.map(formatValue).join(' '),
      error: (...args: unknown[]) => args.map(formatValue).join(' '),
      warn: (...args: unknown[]) => args.map(formatValue).join(' '),
    },
    JSON,
    Math,
    Date,
    Array,
    Object,
    String,
    Number,
    Boolean,
    RegExp,
    Map,
    Set,
    Promise,
    Error,
    TypeError,
    RangeError,
    SyntaxError,

    // Utility functions
    parseInt,
    parseFloat,
    isNaN,
    isFinite,
    encodeURI,
    decodeURI,
    encodeURIComponent,
    decodeURIComponent,

    // User-provided context
    ...userContext,
  };

  return vm.createContext(sandbox);
}

/**
 * Evaluate JavaScript code in a sandboxed context
 */
async function evaluateJavaScript(
  code: string,
  context: Record<string, unknown> = {},
  timeout: number = 5000
): Promise<{ success: boolean; value?: unknown; error?: string }> {
  const sandbox = createSandboxContext(context);

  try {
    const script = new vm.Script(code, {
      filename: 'eval.js',
    });

    const result = script.runInContext(sandbox, {
      timeout,
      displayErrors: true,
    });

    // Handle promises
    if (result instanceof Promise) {
      const resolved = await Promise.race([
        result,
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Promise timeout')), timeout)
        ),
      ]);
      return { success: true, value: resolved };
    }

    return { success: true, value: result };
  } catch (err) {
    const error = err instanceof Error ? err : new Error(String(err));
    return { success: false, error: error.message };
  }
}

/**
 * Execute code via subprocess
 * Spawns the command, writes code to stdin, reads result from stdout
 */
async function executeSubprocess(
  config: ExecutorConfig,
  code: string,
  timeout: number = 5000
): Promise<{ success: boolean; output?: string; error?: string }> {
  return new Promise((resolve) => {
    const proc = spawn(config.command, config.args ?? [], {
      cwd: config.cwd,
      shell: true,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    let killed = false;

    const timeoutHandle = setTimeout(() => {
      killed = true;
      proc.kill('SIGKILL');
      resolve({ success: false, error: `Timeout after ${timeout}ms` });
    }, timeout);

    proc.stdout?.on('data', (data) => {
      stdout += data.toString();
    });

    proc.stderr?.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('close', (exitCode) => {
      clearTimeout(timeoutHandle);
      if (killed) return;

      if (exitCode === 0) {
        resolve({ success: true, output: stdout.trim() });
      } else {
        resolve({
          success: false,
          output: stdout.trim(),
          error: stderr.trim() || `Process exited with code ${exitCode}`,
        });
      }
    });

    proc.on('error', (err) => {
      clearTimeout(timeoutHandle);
      if (killed) return;
      resolve({ success: false, error: err.message });
    });

    // Write code to stdin and close
    proc.stdin?.write(code);
    proc.stdin?.end();
  });
}

/**
 * Evaluation Service class
 *
 * Fully event-driven - emits eval.request and listens for eval.result.
 * Supports:
 *   - Built-in JavaScript executor
 *   - Subprocess executors registered via registerExecutor() or eval.executor.register event
 */
export class EvaluationService {
  private events: EventEmitter;
  private pendingRequests: Map<string, {
    resolve: (result: EvaluationResult) => void;
    timeout: NodeJS.Timeout;
  }> = new Map();
  private executors: Map<string, ExecutorConfig> = new Map();
  private unsubscribes: Array<() => void> = [];

  constructor(eventEmitter: EventEmitter) {
    this.events = eventEmitter;
    this.setupBuiltinExecutor();
    this.setupSubprocessExecutor();
    this.setupExecutorRegistration();
    this.setupResultListener();
  }

  /**
   * Register a subprocess executor for a language
   */
  registerExecutor(config: ExecutorConfig): void {
    this.executors.set(config.language, config);
    this.events.emit('eval.executor.registered', {
      language: config.language,
      command: config.command,
    });
    console.log(`[eval] Registered executor for ${config.language}: ${config.command}`);
  }

  /**
   * Unregister an executor
   */
  unregisterExecutor(language: string): void {
    this.executors.delete(language);
    this.events.emit('eval.executor.unregistered', { language });
  }

  /**
   * Get registered executors
   */
  getExecutors(): ExecutorConfig[] {
    return Array.from(this.executors.values());
  }

  /**
   * Listen for executor registration events
   */
  private setupExecutorRegistration(): void {
    const unsub = this.events.subscribe('eval.executor.register', (event) => {
      const config = event.payload as ExecutorConfig;
      this.registerExecutor(config);
    });
    this.unsubscribes.push(unsub);
  }

  /**
   * Setup built-in JavaScript executor
   */
  private setupBuiltinExecutor(): void {
    const unsub = this.events.subscribe('eval.request', async (event) => {
      const request = event.payload as EvaluationRequest;

      // Only handle JavaScript/TypeScript
      if (request.language !== 'javascript' && request.language !== 'typescript') {
        return;
      }

      const startTime = performance.now();
      const evalResult = await evaluateJavaScript(
        request.code,
        request.context ?? {},
        request.timeout ?? 5000
      );
      const executionTimeMs = performance.now() - startTime;

      let result: EvaluationResult;
      if (evalResult.success) {
        result = {
          id: request.id,
          success: true,
          value: evalResult.value,
          displayValue: formatValue(evalResult.value),
          type: getValueType(evalResult.value),
          executionTimeMs,
          language: request.language,
        };
      } else {
        result = {
          id: request.id,
          success: false,
          displayValue: evalResult.error || 'Unknown error',
          type: 'error',
          executionTimeMs,
          error: evalResult.error,
          language: request.language,
        };
      }

      this.events.emit('eval.result', result);
      this.events.emit(`eval.result.${request.language}`, result);
    });
    this.unsubscribes.push(unsub);
  }

  /**
   * Setup subprocess executor for registered languages
   */
  private setupSubprocessExecutor(): void {
    const unsub = this.events.subscribe('eval.request', async (event) => {
      const request = event.payload as EvaluationRequest;

      // Check if we have a registered executor for this language
      const executor = this.executors.get(request.language);
      if (!executor) {
        return; // No executor registered, let it timeout or be handled elsewhere
      }

      const startTime = performance.now();
      const result = await executeSubprocess(
        executor,
        request.code,
        request.timeout ?? 5000
      );
      const executionTimeMs = performance.now() - startTime;

      const evalResult: EvaluationResult = {
        id: request.id,
        success: result.success,
        displayValue: result.success ? (result.output ?? '') : (result.error ?? 'Unknown error'),
        type: result.success ? 'string' : 'error',
        executionTimeMs,
        error: result.error,
        language: request.language,
      };

      // Try to parse JSON output
      if (result.success && result.output) {
        try {
          const parsed = JSON.parse(result.output);
          evalResult.value = parsed;
          evalResult.displayValue = formatValue(parsed);
          evalResult.type = getValueType(parsed);
        } catch {
          // Not JSON, keep as string
          evalResult.value = result.output;
        }
      }

      this.events.emit('eval.result', evalResult);
      this.events.emit(`eval.result.${request.language}`, evalResult);
    });
    this.unsubscribes.push(unsub);
  }

  /**
   * Setup listener for eval.result to resolve pending requests
   */
  private setupResultListener(): void {
    const unsub = this.events.subscribe('eval.result', (event) => {
      const result = event.payload as EvaluationResult;
      const pending = this.pendingRequests.get(result.id);

      if (pending) {
        clearTimeout(pending.timeout);
        this.pendingRequests.delete(result.id);
        pending.resolve(result);
      }
    });
    this.unsubscribes.push(unsub);
  }

  /**
   * Execute a single evaluation request
   * Emits eval.request and waits for eval.result
   */
  async execute(request: EvaluationRequest): Promise<EvaluationResult> {
    const timeout = request.timeout ?? 5000;

    // Emit the request
    this.events.emit('eval.request', request);

    // Wait for result
    return new Promise((resolve) => {
      const timeoutHandle = setTimeout(() => {
        this.pendingRequests.delete(request.id);
        resolve({
          id: request.id,
          success: false,
          displayValue: `No executor responded for language: ${request.language}`,
          type: 'error',
          executionTimeMs: timeout,
          error: `No executor available for language: ${request.language}. Register one with eval.executor.register event.`,
          language: request.language,
        });
      }, timeout + 1000); // Give executors a bit more time than the eval timeout

      this.pendingRequests.set(request.id, {
        resolve,
        timeout: timeoutHandle,
      });
    });
  }

  /**
   * Execute multiple evaluation requests
   */
  async batch(requests: EvaluationRequest[]): Promise<EvaluationResult[]> {
    const results: EvaluationResult[] = [];

    for (const request of requests) {
      const result = await this.execute(request);
      results.push(result);
    }

    return results;
  }

  /**
   * Cleanup
   */
  destroy(): void {
    for (const unsub of this.unsubscribes) {
      unsub();
    }
    for (const pending of this.pendingRequests.values()) {
      clearTimeout(pending.timeout);
    }
    this.pendingRequests.clear();
    this.executors.clear();
  }
}

// Singleton instance
let evaluationService: EvaluationService | null = null;

export function initEvaluationService(events: EventEmitter): EvaluationService {
  evaluationService = new EvaluationService(events);
  return evaluationService;
}

export function getEvaluationService(): EvaluationService {
  if (!evaluationService) {
    throw new Error('EvaluationService not initialized. Call initEvaluationService first.');
  }
  return evaluationService;
}
