/**
 * Data Processors
 *
 * Data parsing and formatting processors: csv, json, template, split
 */

import type { Processor } from '../../../types/pipeline';
import type { ProcessorRegistry } from '../ProcessorRegistry';

/**
 * CSV - parse CSV text into array of objects
 */
export const csv: Processor = {
  name: 'csv',
  description: 'Parse CSV text into an array of objects. First row is used as headers by default.',
  configSchema: {
    delimiter: { type: 'string', description: 'Field delimiter', default: ',' },
    headers: { type: 'boolean', description: 'First row contains headers', default: true },
    columns: {
      type: 'array',
      items: { type: 'string' },
      description: 'Column names (required if headers=false)',
    },
    skipEmpty: { type: 'boolean', description: 'Skip empty lines', default: true },
  },
  create: (config) => {
    const delimiter = (config.config?.delimiter as string) ?? ',';
    const hasHeaders = config.config?.headers !== false;
    const columns = config.config?.columns as string[] | undefined;
    const skipEmpty = config.config?.skipEmpty !== false;

    return {
      process: async (input) => {
        if (typeof input !== 'string') {
          return undefined;
        }

        let lines = input.split('\n');
        if (skipEmpty) {
          lines = lines.filter((l) => l.trim());
        }

        if (lines.length === 0) return [];

        let headers: string[];
        let dataLines: string[];

        if (hasHeaders) {
          headers = parseCSVLine(lines[0], delimiter);
          dataLines = lines.slice(1);
        } else if (columns) {
          headers = columns;
          dataLines = lines;
        } else {
          // Auto-generate column names
          const firstRow = parseCSVLine(lines[0], delimiter);
          headers = firstRow.map((_, i) => `col${i}`);
          dataLines = lines;
        }

        const rows = dataLines.map((line) => {
          const values = parseCSVLine(line, delimiter);
          const row: Record<string, string> = {};
          headers.forEach((h, i) => {
            row[h] = values[i] ?? '';
          });
          return row;
        });

        return rows;
      },
    };
  },
};

/**
 * Parse a single CSV line, handling quoted values
 */
function parseCSVLine(line: string, delimiter: string): string[] {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];

    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        // Escaped quote
        current += '"';
        i++;
      } else {
        // Toggle quotes
        inQuotes = !inQuotes;
      }
    } else if (char === delimiter && !inQuotes) {
      result.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }

  result.push(current.trim());
  return result;
}

/**
 * JSON - parse or stringify JSON
 */
export const json: Processor = {
  name: 'json',
  description: 'Parse JSON string to object, or stringify object to JSON string.',
  configSchema: {
    mode: {
      type: 'string',
      enum: ['parse', 'stringify'],
      description: 'Parse JSON string or stringify to JSON',
      default: 'parse',
    },
    pretty: { type: 'boolean', description: 'Pretty print when stringifying', default: false },
    field: { type: 'string', description: 'Field to parse/stringify (uses whole input if not specified)' },
  },
  create: (config) => {
    const mode = (config.config?.mode as string) ?? 'parse';
    const pretty = config.config?.pretty === true;
    const field = config.config?.field as string | undefined;

    return {
      process: async (input) => {
        let value: unknown = input;

        // Extract field if specified
        if (field && input && typeof input === 'object') {
          value = (input as Record<string, unknown>)[field];
        }

        if (mode === 'parse') {
          if (typeof value !== 'string') {
            return value; // Already parsed
          }
          try {
            return JSON.parse(value);
          } catch {
            return { error: true, message: 'Invalid JSON', input: value };
          }
        } else {
          // stringify
          try {
            return pretty ? JSON.stringify(value, null, 2) : JSON.stringify(value);
          } catch {
            return { error: true, message: 'Cannot stringify', input: value };
          }
        }
      },
    };
  },
};

/**
 * Template - format output using a template string
 */
export const template: Processor = {
  name: 'template',
  description: 'Format data using a template with ${field} or ${field.nested} placeholders.',
  configSchema: {
    template: { type: 'string', description: 'Template string with ${field} placeholders', required: true },
    defaultValue: { type: 'string', description: 'Default value for missing fields', default: '' },
  },
  create: (config) => {
    const tmpl = config.config?.template as string;
    const defaultValue = (config.config?.defaultValue as string) ?? '';

    if (!tmpl) {
      throw new Error('template processor requires "template" config');
    }

    return {
      process: async (input) => {
        if (!input || typeof input !== 'object') {
          // For non-objects, just replace ${value} with the input
          return tmpl.replace(/\$\{value\}/g, String(input ?? defaultValue));
        }

        return tmpl.replace(/\$\{([\w.]+)\}/g, (_, path: string) => {
          let value: unknown = input;
          for (const part of path.split('.')) {
            if (value && typeof value === 'object') {
              value = (value as Record<string, unknown>)[part];
            } else {
              return defaultValue;
            }
          }
          return value !== undefined && value !== null ? String(value) : defaultValue;
        });
      },
    };
  },
};

/**
 * Split - split string into parts
 */
export const split: Processor = {
  name: 'split',
  description: 'Split a string into an array of parts.',
  configSchema: {
    separator: { type: 'string', description: 'Separator string or regex pattern', default: '\\n' },
    regex: { type: 'boolean', description: 'Treat separator as regex', default: false },
    field: { type: 'string', description: 'Field to split from object input' },
    trim: { type: 'boolean', description: 'Trim whitespace from each part', default: true },
    filter: { type: 'boolean', description: 'Remove empty strings', default: true },
    limit: { type: 'number', description: 'Maximum number of parts' },
  },
  create: (config) => {
    const separator = (config.config?.separator as string) ?? '\n';
    const isRegex = config.config?.regex === true;
    const field = config.config?.field as string | undefined;
    const trim = config.config?.trim !== false;
    const filter = config.config?.filter !== false;
    const limit = config.config?.limit as number | undefined;

    const splitPattern = isRegex ? new RegExp(separator) : separator;

    return {
      process: async (input) => {
        let text: string;

        if (field && input && typeof input === 'object') {
          const val = (input as Record<string, unknown>)[field];
          text = typeof val === 'string' ? val : String(val ?? '');
        } else if (typeof input === 'string') {
          text = input;
        } else {
          return undefined;
        }

        let parts = text.split(splitPattern);

        if (limit !== undefined) {
          parts = parts.slice(0, limit);
        }

        if (trim) {
          parts = parts.map((p) => p.trim());
        }

        if (filter) {
          parts = parts.filter(Boolean);
        }

        return parts;
      },
    };
  },
};

/**
 * Join - join array into string
 */
export const join: Processor = {
  name: 'join',
  description: 'Join an array of values into a single string.',
  configSchema: {
    separator: { type: 'string', description: 'Separator to use between items', default: '\n' },
    field: { type: 'string', description: 'Field to extract from each item before joining' },
  },
  create: (config) => {
    const separator = (config.config?.separator as string) ?? '\n';
    const field = config.config?.field as string | undefined;

    return {
      process: async (input) => {
        if (!Array.isArray(input)) {
          return String(input ?? '');
        }

        const values = input.map((item) => {
          if (field && item && typeof item === 'object') {
            return String((item as Record<string, unknown>)[field] ?? '');
          }
          return String(item ?? '');
        });

        return values.join(separator);
      },
    };
  },
};

/**
 * Pick - select specific fields from objects
 */
export const pick: Processor = {
  name: 'pick',
  description: 'Select specific fields from an object, discarding all others.',
  configSchema: {
    fields: {
      type: 'array',
      items: { type: 'string' },
      description: 'Fields to keep',
      required: true,
    },
  },
  create: (config) => {
    const fields = config.config?.fields as string[];

    if (!fields || !Array.isArray(fields)) {
      throw new Error('pick processor requires "fields" config as an array');
    }

    return {
      process: async (input) => {
        if (!input || typeof input !== 'object') {
          return undefined;
        }

        const obj = input as Record<string, unknown>;
        const result: Record<string, unknown> = {};

        for (const field of fields) {
          if (field in obj) {
            result[field] = obj[field];
          }
        }

        return result;
      },
    };
  },
};

/**
 * Omit - remove specific fields from objects
 */
export const omit: Processor = {
  name: 'omit',
  description: 'Remove specific fields from an object, keeping all others.',
  configSchema: {
    fields: {
      type: 'array',
      items: { type: 'string' },
      description: 'Fields to remove',
      required: true,
    },
  },
  create: (config) => {
    const fields = new Set(config.config?.fields as string[]);

    if (fields.size === 0) {
      throw new Error('omit processor requires "fields" config as a non-empty array');
    }

    return {
      process: async (input) => {
        if (!input || typeof input !== 'object') {
          return input;
        }

        const obj = input as Record<string, unknown>;
        const result: Record<string, unknown> = {};

        for (const [key, value] of Object.entries(obj)) {
          if (!fields.has(key)) {
            result[key] = value;
          }
        }

        return result;
      },
    };
  },
};

/**
 * Rename - rename fields in objects
 */
export const rename: Processor = {
  name: 'rename',
  description: 'Rename fields in an object.',
  configSchema: {
    mapping: {
      type: 'object',
      description: 'Object mapping old field names to new names, e.g., {"oldName": "newName"}',
      required: true,
    },
  },
  create: (config) => {
    const mapping = config.config?.mapping as Record<string, string>;

    if (!mapping || typeof mapping !== 'object') {
      throw new Error('rename processor requires "mapping" config as an object');
    }

    return {
      process: async (input) => {
        if (!input || typeof input !== 'object') {
          return input;
        }

        const obj = input as Record<string, unknown>;
        const result: Record<string, unknown> = {};

        for (const [key, value] of Object.entries(obj)) {
          const newKey = mapping[key] ?? key;
          result[newKey] = value;
        }

        return result;
      },
    };
  },
};

/**
 * Register all data processors
 */
export function registerDataProcessors(registry: ProcessorRegistry): void {
  registry.registerAll([
    csv,
    json,
    template,
    split,
    join,
    pick,
    omit,
    rename,
  ]);
}
