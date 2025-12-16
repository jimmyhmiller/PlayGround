/**
 * Hot-reload register hook
 *
 * Usage:
 *   node -r hot-reload/register app.js
 *
 * Or in code:
 *   require('hot-reload/register');
 */

import { addHook } from 'pirates';
import { transform } from './transform';
import { createRuntime } from './runtime';
import { startServer } from './server';

const HOT_PORT = parseInt(process.env.HOT_PORT || '3456');
const sourceRoot = process.cwd();

// Create global runtime
const runtime = createRuntime();
(global as any).__hot = runtime;

// Hook into require() to transform .js and .ts files
addHook(
  (code, filename) => {
    // Skip node_modules
    if (filename.includes('node_modules')) return code;

    try {
      return transform(code, { filename, sourceRoot });
    } catch (e) {
      console.error(`[hot] Failed to transform ${filename}:`, (e as Error).message);
      return code; // Return original on error
    }
  },
  {
    exts: ['.js', '.ts', '.tsx', '.jsx'],
    ignoreNodeModules: true,
  }
);

// Start server and connect
const server = startServer({ sourceDir: sourceRoot, port: HOT_PORT });
runtime.connect(HOT_PORT);

console.log(`[hot] Hot-reload enabled on port ${HOT_PORT}`);
console.log(`[hot] Watching ${sourceRoot} for changes`);

// Clean shutdown
process.on('SIGINT', () => {
  server.close();
  process.exit(0);
});
