/**
 * ESM Loader Registration for hot-reload
 *
 * This file is imported via --import flag to set up hot-reload for ESM/TypeScript.
 *
 * Usage:
 *   node --import hot-reload/loader app.ts
 *
 * Environment variables:
 *   HOT_PORT - WebSocket server port (default: 3456)
 *   HOT_SOURCE_ROOT - Source root directory (default: cwd)
 */

import { register } from 'node:module';
import { createRequire } from 'node:module';

// Create require to import CJS modules
const require = createRequire(import.meta.url);

// Import from compiled CJS
const { createRuntime } = require('./runtime.js');
const { startServer } = require('./server.js');

const HOT_PORT = parseInt(process.env.HOT_PORT || '3456');
const sourceRoot = process.env.HOT_SOURCE_ROOT || process.cwd();

// Create global runtime - must be set up BEFORE modules are loaded
const runtime = createRuntime();
runtime.setSourceRoot(sourceRoot);
globalThis.__hot = runtime;

// Register the ESM loader hooks
// The loader runs in a separate thread and transforms source code
register('./esm-loader.mjs', import.meta.url);

// Start the hot-reload server (file watcher + WebSocket)
const server = startServer({ sourceDir: sourceRoot, port: HOT_PORT });

// Connect runtime to server
runtime.connect(HOT_PORT);

console.log(`[hot] Hot-reload enabled (ESM mode) on port ${HOT_PORT}`);
console.log(`[hot] Watching ${sourceRoot} for changes`);

// Clean shutdown
process.on('SIGINT', () => {
  server.close();
  process.exit(0);
});

process.on('SIGTERM', () => {
  server.close();
  process.exit(0);
});
