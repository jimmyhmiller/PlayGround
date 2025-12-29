/**
 * External Processors
 *
 * Processors that execute external commands: shell, python, grep, jq
 */

import { spawn } from 'child_process';
import type { Processor } from '../../../types/pipeline';
import type { ProcessorRegistry } from '../ProcessorRegistry';

/**
 * Execute a command with input piped to stdin
 */
async function execCommand(
  command: string,
  args: string[],
  input: string,
  options: { cwd?: string; timeout?: number; shell?: boolean } = {}
): Promise<{ stdout: string; stderr: string; code: number }> {
  const { cwd, timeout = 30000, shell = true } = options;

  return new Promise((resolve) => {
    const proc = spawn(command, args, {
      cwd,
      shell,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    let killed = false;

    const timeoutHandle = setTimeout(() => {
      killed = true;
      proc.kill('SIGKILL');
      resolve({ stdout, stderr: stderr + '\n[TIMEOUT]', code: -1 });
    }, timeout);

    proc.stdout.on('data', (data: Buffer) => {
      stdout += data.toString();
    });

    proc.stderr.on('data', (data: Buffer) => {
      stderr += data.toString();
    });

    proc.on('close', (code: number | null) => {
      clearTimeout(timeoutHandle);
      if (!killed) {
        resolve({ stdout: stdout.trim(), stderr: stderr.trim(), code: code ?? 0 });
      }
    });

    proc.on('error', (err: Error) => {
      clearTimeout(timeoutHandle);
      if (!killed) {
        resolve({ stdout: '', stderr: err.message, code: -1 });
      }
    });

    // Write input to stdin
    proc.stdin.write(input);
    proc.stdin.end();
  });
}

/**
 * Shell - execute arbitrary shell command
 */
export const shell: Processor = {
  name: 'shell',
  description: 'Execute a shell command. Input is piped to stdin as JSON, stdout is the output.',
  configSchema: {
    command: { type: 'string', description: 'Shell command to execute', required: true },
    args: { type: 'array', items: { type: 'string' }, description: 'Command arguments' },
    cwd: { type: 'string', description: 'Working directory' },
    timeout: { type: 'number', description: 'Timeout in ms', default: 30000 },
    inputField: { type: 'string', description: 'Field to use as stdin (uses JSON of whole input if not specified)' },
    inputFormat: {
      type: 'string',
      enum: ['json', 'text', 'lines'],
      description: 'How to format input for stdin',
      default: 'json',
    },
    outputFormat: {
      type: 'string',
      enum: ['text', 'json', 'lines'],
      description: 'How to parse stdout',
      default: 'text',
    },
  },
  create: (config) => {
    const command = config.config?.command as string;
    const args = (config.config?.args as string[]) ?? [];
    const cwd = config.config?.cwd as string | undefined;
    const timeout = (config.config?.timeout as number) ?? 30000;
    const inputField = config.config?.inputField as string | undefined;
    const inputFormat = (config.config?.inputFormat as string) ?? 'json';
    const outputFormat = (config.config?.outputFormat as string) ?? 'text';

    if (!command) {
      throw new Error('shell processor requires "command" config');
    }

    return {
      process: async (input) => {
        // Format stdin
        let stdin: string;
        let value = input;

        if (inputField && input && typeof input === 'object') {
          value = (input as Record<string, unknown>)[inputField];
        }

        switch (inputFormat) {
          case 'text':
            stdin = String(value ?? '');
            break;
          case 'lines':
            stdin = Array.isArray(value) ? value.join('\n') : String(value ?? '');
            break;
          case 'json':
          default:
            stdin = JSON.stringify(value);
            break;
        }

        const result = await execCommand(command, args, stdin, { cwd, timeout });

        if (result.code !== 0) {
          return {
            error: true,
            code: result.code,
            stderr: result.stderr,
            stdout: result.stdout,
            command,
          };
        }

        // Parse stdout
        switch (outputFormat) {
          case 'json':
            try {
              return JSON.parse(result.stdout);
            } catch {
              return { error: true, message: 'Invalid JSON output', stdout: result.stdout };
            }
          case 'lines':
            return result.stdout.split('\n').filter(Boolean);
          case 'text':
          default:
            return result.stdout;
        }
      },
    };
  },
};

/**
 * Python - execute Python code
 */
export const python: Processor = {
  name: 'python',
  description: 'Execute Python code. Input available as `data` variable (parsed from JSON stdin). Use print() for output.',
  configSchema: {
    code: { type: 'string', description: 'Python code to execute (data variable contains input)' },
    script: { type: 'string', description: 'Path to Python script file (receives JSON on stdin)' },
    python: { type: 'string', description: 'Python interpreter path', default: 'python3' },
    timeout: { type: 'number', description: 'Timeout in ms', default: 30000 },
    outputFormat: {
      type: 'string',
      enum: ['json', 'text', 'lines'],
      description: 'How to parse stdout',
      default: 'json',
    },
  },
  create: (config) => {
    const code = config.config?.code as string | undefined;
    const script = config.config?.script as string | undefined;
    const pythonPath = (config.config?.python as string) ?? 'python3';
    const timeout = (config.config?.timeout as number) ?? 30000;
    const outputFormat = (config.config?.outputFormat as string) ?? 'json';

    if (!code && !script) {
      throw new Error('python processor requires either "code" or "script" config');
    }

    // Wrapper that loads JSON from stdin into `data` variable
    const wrapper = code
      ? `
import sys, json
data = json.load(sys.stdin)
${code}
`.trim()
      : '';

    return {
      process: async (input) => {
        const stdin = JSON.stringify(input);
        let result;

        if (script) {
          result = await execCommand(pythonPath, [script], stdin, { timeout });
        } else {
          result = await execCommand(pythonPath, ['-c', wrapper], stdin, { timeout });
        }

        if (result.code !== 0) {
          return { error: true, stderr: result.stderr, code: result.code };
        }

        switch (outputFormat) {
          case 'json':
            try {
              return JSON.parse(result.stdout);
            } catch {
              // If output isn't JSON, return as text
              return result.stdout;
            }
          case 'lines':
            return result.stdout.split('\n').filter(Boolean);
          case 'text':
          default:
            return result.stdout;
        }
      },
    };
  },
};

/**
 * Grep - filter lines/items matching pattern
 */
export const grep: Processor = {
  name: 'grep',
  description: 'Filter text lines or array items matching a regex pattern.',
  configSchema: {
    pattern: { type: 'string', description: 'Regex pattern to match', required: true },
    field: { type: 'string', description: 'Field to search in (for object input)' },
    invert: { type: 'boolean', description: 'Invert match (like grep -v)', default: false },
    ignoreCase: { type: 'boolean', description: 'Case-insensitive matching', default: false },
    mode: {
      type: 'string',
      enum: ['filter', 'match', 'extract'],
      description: 'filter=keep matching, match=return boolean, extract=return matched groups',
      default: 'filter',
    },
  },
  create: (config) => {
    const pattern = config.config?.pattern as string;
    const field = config.config?.field as string | undefined;
    const invert = config.config?.invert === true;
    const ignoreCase = config.config?.ignoreCase === true;
    const mode = (config.config?.mode as string) ?? 'filter';

    if (!pattern) {
      throw new Error('grep processor requires "pattern" config');
    }

    const flags = ignoreCase ? 'gi' : 'g';
    const regex = new RegExp(pattern, flags);

    const getText = (input: unknown): string | null => {
      if (typeof input === 'string') return input;
      if (field && input && typeof input === 'object') {
        const val = (input as Record<string, unknown>)[field];
        return typeof val === 'string' ? val : null;
      }
      return null;
    };

    return {
      process: async (input) => {
        // Handle array input - filter each item
        if (Array.isArray(input)) {
          const results = input.filter((item) => {
            const text = getText(item);
            if (text === null) return invert;
            regex.lastIndex = 0;
            const matches = regex.test(text);
            return invert ? !matches : matches;
          });
          return results.length > 0 ? results : undefined;
        }

        // Handle string input - filter lines or whole string
        const text = getText(input);
        if (text === null) {
          return invert ? input : undefined;
        }

        // Check if multi-line
        if (text.includes('\n') && mode === 'filter') {
          const lines = text.split('\n');
          const filtered = lines.filter((line) => {
            regex.lastIndex = 0;
            const matches = regex.test(line);
            return invert ? !matches : matches;
          });
          return filtered.length > 0 ? filtered.join('\n') : undefined;
        }

        // Single value
        regex.lastIndex = 0;

        switch (mode) {
          case 'match':
            return regex.test(text);

          case 'extract':
            regex.lastIndex = 0;
            const match = regex.exec(text);
            if (!match) return invert ? input : undefined;
            return invert ? undefined : (match.length > 1 ? match.slice(1) : match[0]);

          case 'filter':
          default:
            const matches = regex.test(text);
            const pass = invert ? !matches : matches;
            return pass ? input : undefined;
        }
      },
    };
  },
};

/**
 * Jq - simple JSON path queries (for complex jq, use shell + jq)
 */
export const jq: Processor = {
  name: 'jq',
  description: 'Simple JSON path extraction. For complex queries, use shell processor with jq command.',
  configSchema: {
    path: { type: 'string', description: 'Path expression like ".data.items[0].name" or ".data[]"', required: true },
  },
  create: (config) => {
    const pathExpr = config.config?.path as string;

    if (!pathExpr) {
      throw new Error('jq processor requires "path" config');
    }

    // Parse path expression
    const parts = pathExpr.split('.').filter(Boolean);

    return {
      process: async (input) => {
        let value: unknown = input;

        for (const part of parts) {
          if (value === undefined || value === null) return undefined;

          // Handle array access like "items[0]" or "items[]"
          const arrayMatch = part.match(/^(\w+)\[(\d*)\]$/);
          if (arrayMatch) {
            const [, key, index] = arrayMatch;
            if (typeof value === 'object' && value !== null) {
              value = (value as Record<string, unknown>)[key];
            } else {
              return undefined;
            }

            if (!Array.isArray(value)) return undefined;

            if (index === '') {
              // "[]" means return the array (will be flattened by flatten processor)
              return value;
            }
            value = value[parseInt(index, 10)];
          } else {
            // Simple field access
            if (typeof value === 'object' && value !== null) {
              value = (value as Record<string, unknown>)[part];
            } else {
              return undefined;
            }
          }
        }

        return value;
      },
    };
  },
};

/**
 * Register all external processors
 */
export function registerExternalProcessors(registry: ProcessorRegistry): void {
  registry.registerAll([
    shell,
    python,
    grep,
    jq,
  ]);
}
