#!/usr/bin/env node

import * as fs from "fs";
import * as path from "path";
import * as net from "net";
import { transform } from "./transform";
import { startServer } from "./server";
import { createRuntime } from "./runtime";
import { strip } from "./strip";

const DEFAULT_PORTS = [3456, 3457, 3458, 3459, 3460];

function findAvailablePort(): Promise<number> {
  return new Promise((resolve) => {
    let index = 0;

    function tryPort() {
      if (index >= DEFAULT_PORTS.length) {
        resolve(DEFAULT_PORTS[0]); // Fall back to first port
        return;
      }

      const port = DEFAULT_PORTS[index];
      const server = net.createServer();

      server.once("error", () => {
        index++;
        tryPort();
      });

      server.once("listening", () => {
        server.close(() => {
          resolve(port);
        });
      });

      server.listen(port, "127.0.0.1");
    }

    tryPort();
  });
}

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

async function run(entry: string) {
  const entryAbsolute = path.resolve(entry);
  const sourceRoot = path.dirname(entryAbsolute);

  console.log(`[hot] Starting with entry: ${entry}`);
  console.log(`[hot] Source root: ${sourceRoot}`);

  // Find available port
  const port = await findAvailablePort();
  console.log(`[hot] Using port: ${port}`);

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
    port,
  });

  // Connect runtime to dev server
  runtime.connect(port);

  // Handle shutdown
  process.on("SIGINT", () => {
    console.log("\n[hot] Shutting down...");
    server.close();
    process.exit(0);
  });
}

function stripFile(inputFile: string, outputFile?: string) {
  const code = fs.readFileSync(inputFile, "utf-8");
  const stripped = strip(code);

  if (outputFile) {
    const outDir = path.dirname(outputFile);
    if (!fs.existsSync(outDir)) {
      fs.mkdirSync(outDir, { recursive: true });
    }
    fs.writeFileSync(outputFile, stripped);
    console.log(`[hot] Stripped: ${inputFile} -> ${outputFile}`);
  } else {
    // Output to stdout
    process.stdout.write(stripped);
  }
}

function stripDir(inputDir: string, outputDir: string, extensions: string[]) {
  const files = walkDir(inputDir, extensions);
  let count = 0;

  for (const file of files) {
    const relative = path.relative(inputDir, file);
    const outFile = path.join(outputDir, relative);

    const code = fs.readFileSync(file, "utf-8");
    const stripped = strip(code);

    const outDir = path.dirname(outFile);
    if (!fs.existsSync(outDir)) {
      fs.mkdirSync(outDir, { recursive: true });
    }

    fs.writeFileSync(outFile, stripped);
    count++;
  }

  console.log(`[hot] Stripped ${count} file(s) from ${inputDir} to ${outputDir}`);
}

function walkDir(dir: string, extensions: string[]): string[] {
  const results: string[] = [];

  function walk(currentDir: string) {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(currentDir, entry.name);

      if (entry.isDirectory()) {
        // Skip node_modules and hidden directories
        if (entry.name !== "node_modules" && !entry.name.startsWith(".")) {
          walk(fullPath);
        }
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name);
        if (extensions.includes(ext)) {
          results.push(fullPath);
        }
      }
    }
  }

  walk(dir);
  return results;
}

function printUsage() {
  console.log("Usage:");
  console.log("  hot run <entry.js>              Run with hot reloading");
  console.log("  hot strip <file.js>             Strip once/defonce from file (stdout)");
  console.log("  hot strip <file.js> -o <out.js> Strip once/defonce to output file");
  console.log("  hot strip -d <src> -o <dist>    Strip directory recursively");
  console.log("");
  console.log("Options:");
  console.log("  -o, --output <path>   Output file or directory");
  console.log("  -d, --dir <path>      Input directory (recursive)");
  console.log("  -e, --ext <exts>      File extensions (default: .js,.ts,.jsx,.tsx)");
  console.log("");
  console.log("Examples:");
  console.log("  hot run ./src/index.js");
  console.log("  hot strip ./src/app.js -o ./dist/app.js");
  console.log("  hot strip -d ./src -o ./dist");
  console.log("  hot strip ./src/app.js | prettier --stdin-filepath app.js");
}

// Simple CLI
const args = process.argv.slice(2);
const command = args[0];

if (command === "run" && args[1]) {
  run(args[1]);
} else if (command === "strip") {
  // Parse strip arguments
  let inputFile: string | undefined;
  let inputDir: string | undefined;
  let output: string | undefined;
  let extensions = [".js", ".ts", ".jsx", ".tsx"];

  for (let i = 1; i < args.length; i++) {
    const arg = args[i];
    if (arg === "-o" || arg === "--output") {
      output = args[++i];
    } else if (arg === "-d" || arg === "--dir") {
      inputDir = args[++i];
    } else if (arg === "-e" || arg === "--ext") {
      extensions = args[++i].split(",").map((e) => (e.startsWith(".") ? e : `.${e}`));
    } else if (!arg.startsWith("-")) {
      inputFile = arg;
    }
  }

  if (inputDir) {
    if (!output) {
      console.error("[hot] Error: --output is required when using --dir");
      process.exit(1);
    }
    stripDir(inputDir, output, extensions);
  } else if (inputFile) {
    stripFile(inputFile, output);
  } else {
    console.error("[hot] Error: No input file or directory specified");
    printUsage();
    process.exit(1);
  }
} else {
  printUsage();
  process.exit(1);
}
