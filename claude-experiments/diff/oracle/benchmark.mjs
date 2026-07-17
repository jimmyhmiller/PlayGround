import { execFileSync } from "node:child_process";
import { closeSync, mkdtempSync, openSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { performance } from "node:perf_hooks";
import { rolldown, watch } from "rolldown";

const oracleRoot = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = dirname(oracleRoot);
const moduleCount = positiveInteger(process.argv[2] ?? "10000", "module count");
const importsPerModule = positiveInteger(process.argv[3] ?? "4", "imports per module");
const iterations = positiveInteger(process.argv[4] ?? "3", "iterations");
const workspace = mkdtempSync(join(tmpdir(), "diffpack-rolldown-bench-"));
const entry = join(workspace, moduleName(0));
const rolldownOutput = join(workspace, "rolldown.cjs");

try {
  generateCorpus();
  execFileSync("cargo", ["build", "--release", "--quiet"], {
    cwd: repositoryRoot,
    stdio: "inherit",
  });

  const rolldownCold = [];
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const started = performance.now();
    const bundle = await rolldown({ input: entry, treeshake: false });
    await bundle.write(singleChunkOutput(rolldownOutput));
    await bundle.close();
    rolldownCold.push(performance.now() - started);
  }

  const rolldownWatch = await benchmarkRolldownWatch();
  const diffpackContentRuns = Array.from({ length: iterations }, () =>
    runDiffpack("bundle-scale-direct"));
  const diffpackDependencyRuns = Array.from({ length: iterations }, () =>
    runDiffpack("bundle-scale-direct-deps"));

  const diffpackCold = diffpackContentRuns.map((run) =>
    run.discover_transform_resolve_ms
      + run.initial_reachability_ms
      + run.initial_emit_ms);
  const diffpackContent = diffpackContentRuns.map((run) =>
    run.edit_transform_resolve_ms
      + run.edit_reachability_ms
      + run.edit_emit_ms);
  const diffpackDependency = diffpackDependencyRuns.map((run) =>
    run.edit_transform_resolve_ms
      + run.edit_reachability_ms
      + run.edit_emit_ms);

  console.log(`modules=${moduleCount} imports=${importsPerModule} iterations=${iterations}`);
  console.log("method,cold_ms,content_edit_ms,dependency_edit_ms,final_output_mb");
  console.log(
    `diffpack,${median(diffpackCold).toFixed(3)},${median(diffpackContent).toFixed(3)},${median(diffpackDependency).toFixed(3)},${median(diffpackDependencyRuns.map((run) => run.bundle_mb)).toFixed(3)}`,
  );
  console.log(
    `rolldown,${median(rolldownCold).toFixed(3)},${rolldownWatch.content.toFixed(3)},${rolldownWatch.dependency.toFixed(3)},${(statSync(rolldownOutput).size / 1_000_000).toFixed(3)}`,
  );
  console.log("note: corpus generation and filesystem-watch detection latency are excluded");
} finally {
  rmSync(workspace, { recursive: true, force: true });
}

async function benchmarkRolldownWatch() {
  const builds = buildQueue();
  const watcher = watch({
    input: entry,
    treeshake: false,
    incrementalBuild: true,
    output: singleChunkOutput(rolldownOutput),
  });
  watcher.on("event", async (event) => {
    if (event.code === "BUNDLE_END") {
      await event.result.close();
      builds.push(event.duration);
    } else if (event.code === "ERROR") {
      builds.fail(event.error);
    }
  });

  await builds.next();
  const editedIndex = Math.floor(moduleCount / 2);
  writeFileSync(
    join(workspace, moduleName(editedIndex)),
    moduleSource(editedIndex, Number.MAX_SAFE_INTEGER),
  );
  const content = await builds.next();

  writeFileSync(entry, moduleSourceAfterDependencyRemoval(0));
  const dependency = await builds.next();
  await watcher.close();
  return { content, dependency };
}

function buildQueue() {
  const values = [];
  const waiters = [];
  let failure;
  return {
    push(value) {
      const waiter = waiters.shift();
      if (waiter) waiter.resolve(value);
      else values.push(value);
    },
    fail(error) {
      failure = error;
      for (const waiter of waiters.splice(0)) waiter.reject(error);
    },
    next() {
      if (failure) return Promise.reject(failure);
      if (values.length > 0) return Promise.resolve(values.shift());
      return new Promise((resolve, reject) => waiters.push({ resolve, reject }));
    },
  };
}

function runDiffpack(mode) {
  const binary = join(repositoryRoot, "target", "release", "diffpack");
  const output = execFileSync(
    binary,
    [mode, String(moduleCount), String(importsPerModule)],
    { cwd: repositoryRoot, encoding: "utf8" },
  ).trim().split("\n");
  const headings = output.at(-2).split(",");
  const values = output.at(-1).split(",");
  return Object.fromEntries(headings.map((heading, index) => [heading, Number(values[index])]));
}

function generateCorpus() {
  for (let index = 0; index < moduleCount; index += 1) {
    writeFileSync(join(workspace, moduleName(index)), moduleSource(index, index));
  }
  // Ensure the directory metadata has been published before timing readers.
  const descriptor = openSync(workspace, "r");
  closeSync(descriptor);
}

function moduleSource(index, value) {
  let source = "";
  for (const dependency of dependencyIndices(index)) {
    source += `import \"./${moduleName(dependency)}\";\n`;
  }
  return `${source}export const value_${index}: number = ${value};\n`;
}

function moduleSourceAfterDependencyRemoval(index) {
  const dependencies = dependencyIndices(index);
  dependencies.shift();
  let source = "";
  for (const dependency of dependencies) {
    source += `import \"./${moduleName(dependency)}\";\n`;
  }
  return `${source}export const value_${index}: number = ${Number.MAX_SAFE_INTEGER};\n`;
}

function dependencyIndices(index) {
  const dependencies = [];
  const fanout = 8;
  for (let offset = 1; offset <= fanout; offset += 1) {
    const child = index * fanout + offset;
    if (child < moduleCount) dependencies.push(child);
  }
  for (let salt = 1; dependencies.length < importsPerModule && salt <= index; salt += 1) {
    const target = index - salt;
    if (!dependencies.includes(target)) dependencies.push(target);
  }
  return dependencies.slice(0, importsPerModule);
}

function moduleName(index) {
  return `module-${String(index).padStart(8, "0")}.ts`;
}

function singleChunkOutput(file) {
  return { file, format: "cjs", codeSplitting: false, minify: false, sourcemap: false };
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

function positiveInteger(value, description) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`${description} must be a positive integer`);
  }
  return parsed;
}
