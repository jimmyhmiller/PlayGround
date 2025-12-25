/**
 * Electron-compatible hot-reload loader
 *
 * Usage: electron -r hot-reload/electron-loader .
 *
 * Uses pirates to hook require() for TypeScript transformation.
 * This works with Electron's Node.js 18.x runtime.
 */

const { addHook } = require('pirates');
const { transform } = require('./transform.js');
const { createRuntime } = require('./runtime.js');
const { startServer } = require('./server.js');

const HOT_PORT = parseInt(process.env.HOT_PORT || '3456');
const sourceRoot = process.env.HOT_SOURCE_ROOT || process.cwd();

// Create global runtime
const runtime = createRuntime();
runtime.setSourceRoot(sourceRoot);
global.__hot = runtime;

// Hook require() to transform .js and .ts files
addHook(
  (code, filename) => {
    // Skip node_modules
    if (filename.includes('node_modules')) return code;

    try {
      return transform(code, { filename, sourceRoot });
    } catch (e) {
      console.error(`[hot] Failed to transform ${filename}:`, e.message);
      return code;
    }
  },
  {
    exts: ['.js', '.ts', '.tsx', '.jsx'],
    ignoreNodeModules: true,
  }
);

// Start hot-reload server
const server = startServer({ sourceDir: sourceRoot, port: HOT_PORT });

// Connect runtime to server
runtime.connect(HOT_PORT);

console.log(`[hot] Hot-reload enabled (Electron mode) on port ${HOT_PORT}`);
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
