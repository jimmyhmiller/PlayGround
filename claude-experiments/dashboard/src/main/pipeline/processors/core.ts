/**
 * Core Processors
 *
 * Basic data transformation processors: identity, map, filter, flatten, collect
 */

import type { Processor } from '../../../types/pipeline';
import type { ProcessorRegistry } from '../ProcessorRegistry';

/**
 * Identity - pass through unchanged
 */
export const identity: Processor = {
  name: 'identity',
  description: 'Pass data through unchanged. Useful for debugging or as a placeholder.',
  create: () => ({
    process: async (input) => input,
  }),
};

/**
 * Map - extract a field or apply a named transform
 */
export const map: Processor = {
  name: 'map',
  description: 'Extract a field using dot notation (e.g., "data.value") or apply a named transform.',
  configSchema: {
    path: { type: 'string', description: 'Dot-separated path to extract (e.g., "payload.data.items")' },
    transform: {
      type: 'string',
      enum: ['toString', 'toNumber', 'toBoolean', 'trim', 'lowercase', 'uppercase', 'keys', 'values', 'length'],
      description: 'Named transform to apply after path extraction',
    },
  },
  create: (config) => {
    const path = config.config?.path as string | undefined;
    const transform = config.config?.transform as string | undefined;

    return {
      process: async (input) => {
        let value: unknown = input;

        // Extract path
        if (path) {
          const parts = path.split('.');
          for (const part of parts) {
            if (value === null || value === undefined) {
              return undefined;
            }
            if (typeof value === 'object') {
              value = (value as Record<string, unknown>)[part];
            } else {
              return undefined;
            }
          }
        }

        // Apply named transform
        if (transform && value !== undefined) {
          switch (transform) {
            case 'toString':
              return String(value);
            case 'toNumber':
              return Number(value);
            case 'toBoolean':
              return Boolean(value);
            case 'trim':
              return typeof value === 'string' ? value.trim() : value;
            case 'lowercase':
              return typeof value === 'string' ? value.toLowerCase() : value;
            case 'uppercase':
              return typeof value === 'string' ? value.toUpperCase() : value;
            case 'keys':
              return value && typeof value === 'object' ? Object.keys(value) : undefined;
            case 'values':
              return value && typeof value === 'object' ? Object.values(value) : undefined;
            case 'length':
              if (Array.isArray(value)) return value.length;
              if (typeof value === 'string') return value.length;
              if (value && typeof value === 'object') return Object.keys(value).length;
              return undefined;
          }
        }

        return value;
      },
    };
  },
};

/**
 * Filter - keep only items matching a condition
 */
export const filter: Processor = {
  name: 'filter',
  description: 'Filter items by field existence, equality, or pattern matching. Returns undefined to filter out.',
  configSchema: {
    exists: { type: 'string', description: 'Keep items where this field exists and is truthy' },
    equals: {
      type: 'object',
      properties: {
        field: { type: 'string' },
        value: { type: 'any' },
      },
      description: 'Keep items where field equals value',
    },
    contains: {
      type: 'object',
      properties: {
        field: { type: 'string' },
        value: { type: 'string' },
      },
      description: 'Keep items where string field contains value',
    },
    matches: {
      type: 'object',
      properties: {
        field: { type: 'string' },
        pattern: { type: 'string' },
      },
      description: 'Keep items where field matches regex pattern',
    },
    not: { type: 'boolean', description: 'Invert the filter (keep non-matching items)', default: false },
  },
  create: (config) => {
    const exists = config.config?.exists as string | undefined;
    const equals = config.config?.equals as { field: string; value: unknown } | undefined;
    const contains = config.config?.contains as { field: string; value: string } | undefined;
    const matches = config.config?.matches as { field: string; pattern: string } | undefined;
    const not = config.config?.not === true;

    const matchesRegex = matches ? new RegExp(matches.pattern) : null;

    return {
      process: async (input) => {
        if (input === null || input === undefined) {
          return not ? input : undefined;
        }

        let pass = true;

        if (typeof input === 'object') {
          const obj = input as Record<string, unknown>;

          if (exists) {
            pass = pass && Boolean(obj[exists]);
          }

          if (equals) {
            pass = pass && obj[equals.field] === equals.value;
          }

          if (contains) {
            const fieldVal = obj[contains.field];
            pass = pass && typeof fieldVal === 'string' && fieldVal.includes(contains.value);
          }

          if (matches && matchesRegex) {
            const fieldVal = obj[matches.field];
            pass = pass && typeof fieldVal === 'string' && matchesRegex.test(fieldVal);
          }
        } else if (typeof input === 'string') {
          // For string input, use the whole string for pattern matching
          if (contains) {
            pass = pass && input.includes(contains.value);
          }
          if (matches && matchesRegex) {
            pass = pass && matchesRegex.test(input);
          }
        }

        // Apply inversion
        if (not) {
          pass = !pass;
        }

        return pass ? input : undefined;
      },
    };
  },
};

/**
 * Flatten - expand arrays into individual items
 */
export const flatten: Processor = {
  name: 'flatten',
  description: 'If input is an array, emit each element separately. Nested arrays are flattened one level.',
  configSchema: {
    depth: { type: 'number', description: 'Depth to flatten (default: 1)', default: 1 },
  },
  create: (config) => {
    const depth = (config.config?.depth as number) ?? 1;

    return {
      process: async (input) => {
        if (!Array.isArray(input)) {
          return input;
        }

        if (depth <= 0) {
          return input;
        }

        // Flatten to specified depth
        let result = input;
        for (let i = 0; i < depth; i++) {
          result = result.flat();
        }

        return result; // Returning array emits each element
      },
    };
  },
};

/**
 * Collect - accumulate items into batches
 */
export const collect: Processor = {
  name: 'collect',
  description: 'Collect items into batches of specified size. Emits when batch is full.',
  configSchema: {
    size: { type: 'number', description: 'Batch size (default: 10)', default: 10 },
  },
  create: (config) => {
    const size = (config.config?.size as number) ?? 10;
    let buffer: unknown[] = [];

    return {
      process: async (input) => {
        buffer.push(input);

        if (buffer.length >= size) {
          const batch = buffer;
          buffer = [];
          return batch;
        }

        return undefined; // Don't emit until batch is full
      },
      destroy: () => {
        buffer = [];
      },
      getState: () => ({ buffered: buffer.length, size }),
    };
  },
};

/**
 * Take - take first N items then stop
 */
export const take: Processor = {
  name: 'take',
  description: 'Pass through only the first N items, then filter all subsequent items.',
  configSchema: {
    count: { type: 'number', description: 'Number of items to take', required: true },
  },
  create: (config) => {
    const count = (config.config?.count as number) ?? 1;
    let taken = 0;

    return {
      process: async (input) => {
        if (taken >= count) {
          return undefined;
        }
        taken++;
        return input;
      },
      getState: () => ({ taken, count }),
    };
  },
};

/**
 * Skip - skip first N items
 */
export const skip: Processor = {
  name: 'skip',
  description: 'Skip the first N items, then pass through all subsequent items.',
  configSchema: {
    count: { type: 'number', description: 'Number of items to skip', required: true },
  },
  create: (config) => {
    const count = (config.config?.count as number) ?? 0;
    let skipped = 0;

    return {
      process: async (input) => {
        if (skipped < count) {
          skipped++;
          return undefined;
        }
        return input;
      },
      getState: () => ({ skipped, count }),
    };
  },
};

/**
 * Register all core processors
 */
export function registerCoreProcessors(registry: ProcessorRegistry): void {
  registry.registerAll([
    identity,
    map,
    filter,
    flatten,
    collect,
    take,
    skip,
  ]);
}
