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
const treeShake = process.argv.includes("--treeshake");
const liveOutput = process.argv.includes("--live");
const minifyRolldown = process.argv.includes("--minify");
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
    const bundle = await rolldown({ input: entry, treeshake: treeShake });
    await bundle.write(singleChunkOutput(rolldownOutput));
    await bundle.close();
    rolldownCold.push(performance.now() - started);
  }

  const rolldownWatchRuns = [];
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    rolldownWatchRuns.push(await benchmarkRolldownWatch());
  }
  const diffpackContentRuns = Array.from({ length: iterations }, () =>
    runDiffpack(
      liveOutput
        ? (minifyRolldown ? "bundle-scale-direct-live-minify" : "bundle-scale-direct-live")
        : "bundle-scale-direct",
    ));
  const diffpackDependencyRuns = Array.from({ length: iterations }, () =>
    runDiffpack(
      liveOutput
        ? (minifyRolldown ? "bundle-scale-direct-live-minify-deps" : "bundle-scale-direct-live-deps")
        : "bundle-scale-direct-deps",
    ));

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

  if (liveOutput) {
    const expectedContentValue = moduleCount * (moduleCount - 1) / 2 + moduleCount;
    const expectedDependencyValue = liveValueAfterDependencyRemoval();
    const diffpackContentValues = new Set(diffpackContentRuns.map((run) => run.runtime_value));
    const diffpackValues = new Set(diffpackDependencyRuns.map((run) => run.runtime_value));
    const rolldownValue = executeNumericBundle(rolldownOutput);
    if (
      diffpackContentValues.size !== 1
      || !diffpackContentValues.has(expectedContentValue)
      || rolldownWatchRuns.some((run) => run.content.runtimeValue !== expectedContentValue)
    ) {
      throw new Error(
        `runtime mismatch after content edit: expected=${expectedContentValue} Diffpack=${[...diffpackContentValues].join(",")} Rolldown=${rolldownWatchRuns.map((run) => run.content.runtimeValue).join(",")}`,
      );
    }
    if (
      diffpackValues.size !== 1
      || !diffpackValues.has(expectedDependencyValue)
      || rolldownValue !== expectedDependencyValue
      || rolldownWatchRuns.some((run) => run.dependency.runtimeValue !== expectedDependencyValue)
    ) {
      throw new Error(
        `runtime mismatch after dependency edit: expected=${expectedDependencyValue} Diffpack=${[...diffpackValues].join(",")} Rolldown=${rolldownValue}`,
      );
    }
  }
  console.log(`modules=${moduleCount} imports=${importsPerModule} iterations=${iterations} treeshake=${treeShake} live=${liveOutput} rolldown_minify=${minifyRolldown}`);
  const diffpackBytes = median(diffpackDependencyRuns.map((run) => run.bundle_bytes));
  const rolldownBytes = statSync(rolldownOutput).size;
  if (treeShake && diffpackBytes > rolldownBytes) {
    throw new Error(`Diffpack output is larger than Rolldown: ${diffpackBytes} > ${rolldownBytes}`);
  }
  console.log("method,cold_ms,content_edit_ms,dependency_edit_ms,final_output_mb,final_output_bytes");
  console.log(
    `diffpack,${median(diffpackCold).toFixed(3)},${median(diffpackContent).toFixed(3)},${median(diffpackDependency).toFixed(3)},${median(diffpackDependencyRuns.map((run) => run.bundle_mb)).toFixed(3)},${diffpackBytes}`,
  );
  console.log(
    `rolldown,${median(rolldownCold).toFixed(3)},${median(rolldownWatchRuns.map((run) => run.content.duration)).toFixed(3)},${median(rolldownWatchRuns.map((run) => run.dependency.duration)).toFixed(3)},${(rolldownBytes / 1_000_000).toFixed(3)},${rolldownBytes}`,
  );
  console.log("note: corpus generation and filesystem-watch detection latency are excluded");
} finally {
  rmSync(workspace, { recursive: true, force: true });
}

async function benchmarkRolldownWatch() {
  writeFileSync(entry, moduleSource(0, 0));
  const editedIndex = Math.floor(moduleCount / 2);
  const content = await benchmarkRolldownEdit(() => {
    writeFileSync(
      join(workspace, moduleName(editedIndex)),
      moduleSource(editedIndex, liveOutput ? moduleCount + editedIndex : Number.MAX_SAFE_INTEGER),
    );
  });
  writeFileSync(join(workspace, moduleName(editedIndex)), moduleSource(editedIndex, editedIndex));
  const dependency = await benchmarkRolldownEdit(() => {
    writeFileSync(entry, moduleSourceAfterDependencyRemoval(0));
  });
  writeFileSync(entry, moduleSource(0, 0));
  return { content, dependency };
}

async function benchmarkRolldownEdit(applyEdit) {
  const builds = buildQueue();
  const watcher = watch({
    input: entry,
    treeshake: treeShake,
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
  applyEdit();
  const duration = await builds.next();
  const runtimeValue = liveOutput ? executeNumericBundle(rolldownOutput) : undefined;
  await watcher.close();
  return { duration, runtimeValue };
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

function executeNumericBundle(file) {
  const stdout = execFileSync("node", [file], { encoding: "utf8" }).trim();
  const value = Number(stdout);
  if (!Number.isSafeInteger(value)) {
    throw new Error(`bundle output is not a safe integer: ${JSON.stringify(stdout)}`);
  }
  return value;
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
  if (liveOutput) return liveModuleSource(index, value, false);
  let source = "";
  for (const dependency of dependencyIndices(index)) {
    source += `import \"./${moduleName(dependency)}\";\n`;
  }
  return `${source}export const value_${index}: number = ${value};\n`;
}

function moduleSourceAfterDependencyRemoval(index) {
  if (liveOutput) return liveModuleSource(index, moduleCount + index, true);
  const dependencies = dependencyIndices(index);
  dependencies.shift();
  let source = "";
  for (const dependency of dependencies) {
    source += `import \"./${moduleName(dependency)}\";\n`;
  }
  return `${source}export const value_${index}: number = ${Number.MAX_SAFE_INTEGER};\n`;
}

function liveModuleSource(index, value, removeFirstDependency) {
  const dependencies = liveDependencyIndices(index);
  if (removeFirstDependency) dependencies.shift();
  let source = "";
  for (const dependency of dependencies) {
    source += `import { value_${dependency} } from "./${moduleName(dependency)}";\n`;
  }
  const expression = [String(value), ...dependencies.map((dependency) => `value_${dependency}`)].join(" + ");
  source += `export const value_${index}: number = ${expression};\n`;
  if (index === 0) source += "console.log(value_0);\n";
  return source;
}

function liveDependencyIndices(index) {
  const dependencies = [];
  for (let offset = 1; offset <= importsPerModule; offset += 1) {
    const child = index * importsPerModule + offset;
    if (child < moduleCount) dependencies.push(child);
  }
  return dependencies;
}

function liveValueAfterDependencyRemoval() {
  const removed = new Set();
  const firstDependency = liveDependencyIndices(0)[0];
  if (firstDependency !== undefined) {
    const pending = [firstDependency];
    while (pending.length > 0) {
      const index = pending.pop();
      if (removed.has(index)) continue;
      removed.add(index);
      pending.push(...liveDependencyIndices(index));
    }
  }
  let value = moduleCount;
  for (let index = 1; index < moduleCount; index += 1) {
    if (!removed.has(index)) value += index;
  }
  return value;
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
  return { file, format: "cjs", codeSplitting: false, minify: minifyRolldown, sourcemap: false };
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
