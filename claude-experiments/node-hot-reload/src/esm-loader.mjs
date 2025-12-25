/**
 * ESM Loader hooks for hot-reload
 *
 * This loader intercepts ESM imports and transforms TypeScript/JavaScript files
 * to enable hot-reloading support.
 *
 * This file must be ESM (.mjs) for Node's module.register() API.
 */

import * as path from 'node:path';
import * as fs from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

// Create require to import CJS modules
const require = createRequire(import.meta.url);

// Import the transform function from compiled CJS
const { transform } = require('./transform.js');

const sourceRoot = process.env.HOT_SOURCE_ROOT || process.cwd();

// Extensions we handle
const SUPPORTED_EXTENSIONS = ['.ts', '.tsx', '.js', '.jsx', '.mts', '.mjs'];

/**
 * Check if a path is within node_modules
 */
function isNodeModules(filePath) {
  return filePath.includes('node_modules');
}

/**
 * Resolve hook - handles TypeScript extension resolution
 */
export async function resolve(specifier, context, nextResolve) {
  // Skip data: URLs and node: built-ins
  if (specifier.startsWith('data:') || specifier.startsWith('node:')) {
    return nextResolve(specifier, context);
  }

  // For relative/absolute imports, try to resolve TypeScript extensions
  if (specifier.startsWith('.') || specifier.startsWith('/') || specifier.startsWith('file://')) {
    const parentPath = context.parentURL ? fileURLToPath(context.parentURL) : sourceRoot;
    const parentDir = path.dirname(parentPath);

    // If specifier already has an extension, try it directly
    const ext = path.extname(specifier);
    if (ext && SUPPORTED_EXTENSIONS.includes(ext)) {
      const resolved = path.resolve(parentDir, specifier);
      if (fs.existsSync(resolved)) {
        return {
          url: pathToFileURL(resolved).href,
          shortCircuit: true,
          format: 'module',
        };
      }
    }

    // Try adding TypeScript extensions
    if (!ext || ext === '.js') {
      const basePath = ext === '.js'
        ? path.resolve(parentDir, specifier.slice(0, -3))
        : path.resolve(parentDir, specifier);

      // Try .ts, .tsx, .js, .jsx in order
      for (const tryExt of ['.ts', '.tsx', '.js', '.jsx']) {
        const tryPath = basePath + tryExt;
        if (fs.existsSync(tryPath)) {
          return {
            url: pathToFileURL(tryPath).href,
            shortCircuit: true,
            format: 'module',
          };
        }
      }

      // Try index files
      for (const tryExt of ['.ts', '.tsx', '.js', '.jsx']) {
        const tryPath = path.join(basePath, 'index' + tryExt);
        if (fs.existsSync(tryPath)) {
          return {
            url: pathToFileURL(tryPath).href,
            shortCircuit: true,
            format: 'module',
          };
        }
      }
    }
  }

  // Defer to default resolution for everything else
  return nextResolve(specifier, context);
}

/**
 * Load hook - transforms source code for hot-reload
 */
export async function load(url, context, nextLoad) {
  // Only handle file:// URLs
  if (!url.startsWith('file://')) {
    return nextLoad(url, context);
  }

  const filePath = fileURLToPath(url);
  const ext = path.extname(filePath);

  // Skip node_modules and non-supported extensions
  if (isNodeModules(filePath) || !SUPPORTED_EXTENSIONS.includes(ext)) {
    return nextLoad(url, context);
  }

  // Read the source file
  const source = await fs.promises.readFile(filePath, 'utf-8');

  try {
    // Transform for hot-reload with ESM output
    const transformed = transform(source, {
      filename: filePath,
      sourceRoot,
      esm: true,
    });

    return {
      format: 'module',
      shortCircuit: true,
      source: transformed,
    };
  } catch (error) {
    console.error(`[hot] Failed to transform ${filePath}:`, error.message);
    // On transform error, try to load as regular JS (for already-transpiled files)
    return nextLoad(url, context);
  }
}
