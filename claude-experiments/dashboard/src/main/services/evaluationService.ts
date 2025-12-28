/**
 * Evaluation Service
 *
 * Sandboxed code execution service supporting multiple languages.
 * Uses Node.js vm module for JavaScript evaluation with timeout protection.
 *
 * Events emitted:
 *   - eval.result   { id, success, value, displayValue, type, executionTimeMs, error? }
 */

import * as vm from 'vm';

// Type for the events module
interface EventEmitter {
  emit(type: string, payload: unknown): void;
}

export interface EvaluationRequest {
  id: string;
  code: string;
  language: 'javascript' | 'typescript';
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
export async function evaluateJavaScript(
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
 * Evaluation Service class
 *
 * Emits events:
 *   - eval.result: After each evaluation completes
 */
export class EvaluationService {
  private events: EventEmitter | null = null;

  constructor(eventEmitter?: EventEmitter) {
    this.events = eventEmitter ?? null;
  }

  /**
   * Execute a single evaluation request
   */
  async execute(request: EvaluationRequest): Promise<EvaluationResult> {
    const { id, code, language, context = {}, timeout = 5000 } = request;

    const startTime = performance.now();

    if (language !== 'javascript' && language !== 'typescript') {
      const result: EvaluationResult = {
        id,
        success: false,
        displayValue: '',
        type: 'error',
        executionTimeMs: 0,
        error: `Unsupported language: ${language}. Currently only JavaScript is supported.`,
      };
      this.events?.emit('eval.result', result);
      return result;
    }

    const evalResult = await evaluateJavaScript(code, context, timeout);
    const executionTimeMs = performance.now() - startTime;

    let result: EvaluationResult;
    if (evalResult.success) {
      result = {
        id,
        success: true,
        value: evalResult.value,
        displayValue: formatValue(evalResult.value),
        type: getValueType(evalResult.value),
        executionTimeMs,
      };
    } else {
      result = {
        id,
        success: false,
        displayValue: evalResult.error || 'Unknown error',
        type: 'error',
        executionTimeMs,
        error: evalResult.error,
      };
    }

    // Emit event with result
    this.events?.emit('eval.result', result);

    return result;
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
}

// Singleton instance
let evaluationService: EvaluationService | null = null;

export function initEvaluationService(events: EventEmitter): EvaluationService {
  evaluationService = new EvaluationService(events);
  return evaluationService;
}

export function getEvaluationService(): EvaluationService {
  if (!evaluationService) {
    // Fallback without events
    evaluationService = new EvaluationService();
  }
  return evaluationService;
}
