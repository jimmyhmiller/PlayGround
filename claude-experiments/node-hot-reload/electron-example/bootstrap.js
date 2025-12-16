/**
 * Bootstrap script for Electron with hot-reload
 * This sets up the runtime and loads the transformed main.js
 */

const path = require('path');
const fs = require('fs');

// Import from our hot-reload package (parent directory)
const { transform } = require('../dist/transform');
const { createRuntime } = require('../dist/runtime');
const { startServer } = require('../dist/server');

const HOT_PORT = 3457;
const SOURCE_DIR = __dirname;

// Create runtime and make it global
const runtime = createRuntime();
global.__hot = runtime;

// Collect and transform all modules
function collectModules(entryFile) {
  const modules = new Map();
  const queue = [entryFile];
  const seen = new Set();

  while (queue.length > 0) {
    const file = queue.shift();
    const absolute = path.isAbsolute(file) ? file : path.resolve(SOURCE_DIR, file);

    if (seen.has(absolute)) continue;
    seen.add(absolute);

    if (!absolute.endsWith('.js')) continue;
    if (!fs.existsSync(absolute)) continue;
    // Skip this bootstrap file and node_modules
    if (absolute.includes('bootstrap.js')) continue;
    if (absolute.includes('node_modules')) continue;

    const code = fs.readFileSync(absolute, 'utf-8');

    try {
      const transformed = transform(code, {
        filename: absolute,
        sourceRoot: SOURCE_DIR,
      });

      const moduleId = path.relative(SOURCE_DIR, absolute);

      // Extract dependencies
      const deps = [];
      const getRegex = /__hot\.get\("([^"]+)"\)/g;
      let match;
      while ((match = getRegex.exec(transformed)) !== null) {
        const dep = match[1];
        if (dep.endsWith('.js')) {
          deps.push(dep);
          const depAbsolute = path.resolve(SOURCE_DIR, dep);
          queue.push(depAbsolute);
        }
      }

      modules.set(moduleId, { id: moduleId, code: transformed, deps });
    } catch (e) {
      console.error(`[hot] Failed to transform ${absolute}:`, e.message);
    }
  }

  return modules;
}

// Topological sort
function topologicalSort(modules) {
  const sorted = [];
  const visited = new Set();
  const visiting = new Set();

  function visit(id) {
    if (visited.has(id)) return;
    if (visiting.has(id)) return;
    visiting.add(id);

    const mod = modules.get(id);
    if (mod) {
      for (const dep of mod.deps) {
        const depId = dep.endsWith('.js') ? dep : `${dep}.js`;
        if (modules.has(depId)) visit(depId);
      }
    }

    visiting.delete(id);
    visited.add(id);
    sorted.push(id);
  }

  for (const id of modules.keys()) {
    visit(id);
  }

  return sorted;
}

// Main
console.log('[hot] Starting Electron with hot-reload...');

const modules = collectModules(path.join(SOURCE_DIR, 'main.js'));
console.log(`[hot] Found ${modules.size} module(s)`);

const loadOrder = topologicalSort(modules);
console.log(`[hot] Load order: ${loadOrder.join(' -> ')}`);

// Load all modules
for (const id of loadOrder) {
  const mod = modules.get(id);
  if (mod) {
    try {
      const fn = new Function('__hot', 'require', '__dirname', '__filename', mod.code);
      fn(runtime, require, SOURCE_DIR, path.join(SOURCE_DIR, id));
      console.log(`[hot] Loaded ${id}`);
    } catch (e) {
      console.error(`[hot] Failed to load ${id}:`, e);
      process.exit(1);
    }
  }
}

// Start dev server
const server = startServer({
  sourceDir: SOURCE_DIR,
  port: HOT_PORT,
});

// Connect runtime to dev server
runtime.connect(HOT_PORT);

console.log('[hot] Hot-reload ready! Edit main.js to see changes.');
