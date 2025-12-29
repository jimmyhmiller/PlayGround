/**
 * Stateful Processors
 *
 * Processors that maintain state across invocations: diff, debounce, throttle, accumulate
 */

import type { Processor } from '../../../types/pipeline';
import type { ProcessorRegistry } from '../ProcessorRegistry';

/**
 * Diff - emit only when value changes
 */
export const diff: Processor = {
  name: 'diff',
  description: 'Emit only when the input (or specified field) changes from previous value. Outputs { previous, current, input }.',
  configSchema: {
    field: { type: 'string', description: 'Field to compare (compares whole value if not specified)' },
    emitFirst: { type: 'boolean', description: 'Emit the first value even with no previous', default: true },
    outputMode: {
      type: 'string',
      enum: ['diff', 'current', 'input'],
      description: 'What to output: "diff" = {previous, current}, "current" = just current value, "input" = original input',
      default: 'input',
    },
  },
  create: (config) => {
    const field = config.config?.field as string | undefined;
    const emitFirst = config.config?.emitFirst !== false;
    const outputMode = (config.config?.outputMode as string) ?? 'input';
    let previous: unknown = undefined;
    let hasPrevious = false;

    const getValue = (input: unknown): unknown => {
      if (!field) return input;
      if (input && typeof input === 'object') {
        return (input as Record<string, unknown>)[field];
      }
      return input;
    };

    return {
      process: async (input) => {
        const current = getValue(input);
        const currentJson = JSON.stringify(current);
        const previousJson = JSON.stringify(previous);

        if (!hasPrevious) {
          hasPrevious = true;
          previous = current;
          if (emitFirst) {
            return outputMode === 'diff' ? { previous: undefined, current, input } : input;
          }
          return undefined;
        }

        if (currentJson !== previousJson) {
          const prev = previous;
          previous = current;

          switch (outputMode) {
            case 'diff':
              return { previous: prev, current, input };
            case 'current':
              return current;
            case 'input':
            default:
              return input;
          }
        }

        return undefined;
      },
      getState: () => ({ previous, hasPrevious }),
    };
  },
};

/**
 * Debounce - emit only after quiet period
 */
export const debounce: Processor = {
  name: 'debounce',
  description: 'Wait for a quiet period before emitting. If new input arrives, restart the timer. Last value wins.',
  configSchema: {
    delay: { type: 'number', description: 'Quiet period in milliseconds', default: 300 },
  },
  create: (config) => {
    const delay = (config.config?.delay as number) ?? 300;
    let timeoutHandle: ReturnType<typeof setTimeout> | null = null;
    let pendingResolve: ((value: unknown) => void) | null = null;

    return {
      process: async (input) => {
        // Clear any existing timeout
        if (timeoutHandle) {
          clearTimeout(timeoutHandle);
          // Resolve previous promise with undefined (filtered out)
          if (pendingResolve) {
            pendingResolve(undefined);
            pendingResolve = null;
          }
        }

        // Return a promise that resolves after the delay
        return new Promise((resolve) => {
          pendingResolve = resolve;
          timeoutHandle = setTimeout(() => {
            timeoutHandle = null;
            pendingResolve = null;
            resolve(input);
          }, delay);
        });
      },
      destroy: () => {
        if (timeoutHandle) {
          clearTimeout(timeoutHandle);
          timeoutHandle = null;
        }
        if (pendingResolve) {
          pendingResolve(undefined);
          pendingResolve = null;
        }
      },
    };
  },
};

/**
 * Throttle - emit at most once per interval
 */
export const throttle: Processor = {
  name: 'throttle',
  description: 'Emit at most once per interval. First value in each interval is emitted, rest are dropped.',
  configSchema: {
    interval: { type: 'number', description: 'Minimum interval between emissions (ms)', default: 1000 },
  },
  create: (config) => {
    const interval = (config.config?.interval as number) ?? 1000;
    let lastEmit = 0;

    return {
      process: async (input) => {
        const now = Date.now();
        if (now - lastEmit >= interval) {
          lastEmit = now;
          return input;
        }
        return undefined;
      },
      getState: () => ({ lastEmit, interval }),
    };
  },
};

/**
 * Accumulate - running aggregation
 */
export const accumulate: Processor = {
  name: 'accumulate',
  description: 'Accumulate values with running aggregation. Outputs statistics object.',
  configSchema: {
    mode: {
      type: 'string',
      enum: ['sum', 'count', 'avg', 'min', 'max', 'all', 'list'],
      description: 'Aggregation mode',
      default: 'all',
    },
    field: { type: 'string', description: 'Field to extract numeric value from (uses input directly if not specified)' },
    maxItems: { type: 'number', description: 'Max items to keep in list mode', default: 100 },
  },
  create: (config) => {
    const mode = (config.config?.mode as string) ?? 'all';
    const field = config.config?.field as string | undefined;
    const maxItems = (config.config?.maxItems as number) ?? 100;

    let sum = 0;
    let count = 0;
    let min = Infinity;
    let max = -Infinity;
    let list: unknown[] = [];

    const getNumericValue = (input: unknown): number => {
      let value: unknown = input;
      if (field && input && typeof input === 'object') {
        value = (input as Record<string, unknown>)[field];
      }
      const num = Number(value);
      return isNaN(num) ? 0 : num;
    };

    return {
      process: async (input) => {
        const value = getNumericValue(input);
        count++;
        sum += value;
        min = Math.min(min, value);
        max = Math.max(max, value);

        if (mode === 'list') {
          list.push(input);
          if (list.length > maxItems) list.shift();
        }

        switch (mode) {
          case 'sum':
            return { sum };
          case 'count':
            return { count };
          case 'avg':
            return { avg: sum / count, count };
          case 'min':
            return { min: min === Infinity ? null : min };
          case 'max':
            return { max: max === -Infinity ? null : max };
          case 'list':
            return { items: [...list], count };
          case 'all':
          default:
            return {
              sum,
              count,
              avg: sum / count,
              min: min === Infinity ? null : min,
              max: max === -Infinity ? null : max,
            };
        }
      },
      getState: () => ({ sum, count, min, max, listSize: list.length }),
    };
  },
};

/**
 * Distinct - filter out duplicates
 */
export const distinct: Processor = {
  name: 'distinct',
  description: 'Filter out duplicate values. Uses JSON serialization for comparison.',
  configSchema: {
    field: { type: 'string', description: 'Field to use for uniqueness check (uses whole value if not specified)' },
    maxSeen: { type: 'number', description: 'Max items to remember (LRU eviction)', default: 1000 },
  },
  create: (config) => {
    const field = config.config?.field as string | undefined;
    const maxSeen = (config.config?.maxSeen as number) ?? 1000;
    const seen = new Map<string, boolean>();

    const getKey = (input: unknown): string => {
      let value = input;
      if (field && input && typeof input === 'object') {
        value = (input as Record<string, unknown>)[field];
      }
      return JSON.stringify(value);
    };

    return {
      process: async (input) => {
        const key = getKey(input);

        if (seen.has(key)) {
          return undefined;
        }

        // LRU eviction
        if (seen.size >= maxSeen) {
          const firstKey = seen.keys().next().value;
          if (firstKey !== undefined) {
            seen.delete(firstKey);
          }
        }

        seen.set(key, true);
        return input;
      },
      getState: () => ({ seenCount: seen.size }),
    };
  },
};

/**
 * Register all stateful processors
 */
export function registerStatefulProcessors(registry: ProcessorRegistry): void {
  registry.registerAll([
    diff,
    debounce,
    throttle,
    accumulate,
    distinct,
  ]);
}
