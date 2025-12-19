#!/usr/bin/env node

import * as fs from "fs";
import * as path from "path";
import { transform } from "./transform";
import { startServer } from "./server";
import { createRuntime } from "./runtime";

const HOT_PORT = 3456;

interface ModuleInfo {
  id: string;
  code: string;
  deps: string[];
}

function collectModules(
  entryFile: string,
  sourceRoot: string
): Map<string, ModuleInfo> {
  const modules = new Map<string, ModuleInfo>();
  const queue = [entryFile];
  const seen = new Set<string>();

  while (queue.length > 0) {
    const file = queue.shift()!;
    const absolute = path.isAbsolute(file)
      ? file
      : path.resolve(sourceRoot, file);

    if (seen.has(absolute)) continue;
    seen.add(absolute);

    // Skip if not a .js file or doesn't exist
    if (!absolute.endsWith(".js")) continue;
    if (!fs.existsSync(absolute)) continue;

    const code = fs.readFileSync(absolute, "utf-8");
    const transformed = transform(code, {
      filename: absolute,
      sourceRoot,
    });

    const moduleId = path.relative(sourceRoot, absolute);

    // Extract dependencies from transformed code (look for __hot.get calls)
    const deps: string[] = [];
    const getRegex = /__hot\.get\("([^"]+)"\)/g;
    let match;
    while ((match = getRegex.exec(transformed)) !== null) {
      const dep = match[1];
      // Skip node_modules - they don't end in .js typically
      if (!dep.endsWith(".js")) continue;

      deps.push(dep);

      // Queue the dependency for processing
      const depAbsolute = path.resolve(sourceRoot, dep);
      const depWithExt = depAbsolute.endsWith(".js")
        ? depAbsolute
        : `${depAbsolute}.js`;
      queue.push(depWithExt);
    }

    modules.set(moduleId, { id: moduleId, code: transformed, deps });
  }

  return modules;
}

function topologicalSort(modules: Map<string, ModuleInfo>): string[] {
  const sorted: string[] = [];
  const visited = new Set<string>();
  const visiting = new Set<string>();

  function visit(id: string) {
    if (visited.has(id)) return;
    if (visiting.has(id)) {
      // Circular dependency - just continue
      return;
    }

    visiting.add(id);

    const mod = modules.get(id);
    if (mod) {
      for (const dep of mod.deps) {
        // Normalize dep to match module id format
        const depId = dep.endsWith(".js") ? dep : `${dep}.js`;
        if (modules.has(depId)) {
          visit(depId);
        }
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

function run(entry: string) {
  const entryAbsolute = path.resolve(entry);
  const sourceRoot = path.dirname(entryAbsolute);

  console.log(`[hot] Starting with entry: ${entry}`);
  console.log(`[hot] Source root: ${sourceRoot}`);

  // Collect and transform all modules
  const modules = collectModules(entryAbsolute, sourceRoot);
  console.log(`[hot] Found ${modules.size} module(s)`);

  // Sort by dependencies
  const loadOrder = topologicalSort(modules);
  console.log(`[hot] Load order: ${loadOrder.join(" -> ")}`);

  // Create runtime and make it global
  const runtime = createRuntime();
  (global as any).__hot = runtime;

  // Load all modules
  for (const id of loadOrder) {
    const mod = modules.get(id);
    if (mod) {
      try {
        const fn = new Function("__hot", mod.code);
        fn(runtime);
        console.log(`[hot] Loaded ${id}`);
      } catch (e) {
        console.error(`[hot] Failed to load ${id}:`, e);
        process.exit(1);
      }
    }
  }

  // Start dev server
  const server = startServer({
    sourceDir: sourceRoot,
    port: HOT_PORT,
  });

  // Connect runtime to dev server
  runtime.connect(HOT_PORT);

  // Handle shutdown
  process.on("SIGINT", () => {
    console.log("\n[hot] Shutting down...");
    server.close();
    process.exit(0);
  });
}

// Simple CLI
const args = process.argv.slice(2);
const command = args[0];

if (command === "run" && args[1]) {
  run(args[1]);
} else {
  console.log("Usage: hot run <entry.js>");
  console.log("");
  console.log("Example:");
  console.log("  hot run ./example/index.js");
  process.exit(1);
}
