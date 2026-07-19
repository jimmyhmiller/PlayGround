import { execFileSync, spawnSync } from "node:child_process";
import {
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";

const oracleRoot = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = dirname(oracleRoot);
const iterations = positiveInteger(process.argv[2] ?? "15");
const workspace = mkdtempSync(join(tmpdir(), "diffpack-capability-bench-"));
const diffpack = join(repositoryRoot, "target", "release", "diffpack");
const rolldown = join(oracleRoot, "node_modules", ".bin", "rolldown");

const cases = [
  {
    name: "tree-shaking",
    fixture: "tree-shaking/entry.js",
    expected: "used\n",
    sentinel: "UNUSED_EXPORT_SENTINEL_73B4",
  },
  {
    name: "package-sideEffects",
    fixture: "package-side-effects/entry.js",
    expected: "entry\n",
  },
  {
    name: "source-maps",
    fixture: "tree-shaking/entry.js",
    expected: "used\n",
    sourceMap: true,
  },
  {
    name: "code-splitting",
    fixture: "code-splitting/entry.js",
    expected: "alpha beta\n",
    codeSplitting: true,
  },
];

execFileSync("cargo", ["build", "--release", "--quiet"], {
  cwd: repositoryRoot,
  stdio: "inherit",
});

console.log(`iterations=${iterations} mode=end-to-end-release-cli`);
console.log("case,method,median_ms,output_kb,files");
try {
  for (const benchmarkCase of cases) {
    const entry = resolve(oracleRoot, "fixtures", benchmarkCase.fixture);
    const verification = Object.fromEntries(
      ["diffpack", "rolldown"].map((method) => {
        const result = build(
          method,
          benchmarkCase,
          entry,
          join(workspace, `${benchmarkCase.name}-${method}-verification`),
        );
        return [method, inspect(benchmarkCase, result)];
      }),
    );
    assertPairEquivalent(benchmarkCase, verification.diffpack, verification.rolldown);
    for (const method of ["diffpack", "rolldown"]) {
      // One unreported run pays lazy loader and filesystem warmup costs.
      build(method, benchmarkCase, entry, join(workspace, `${benchmarkCase.name}-${method}-warmup`));
      const times = [];
      let result;
      for (let iteration = 0; iteration < iterations; iteration += 1) {
        const outputRoot = join(workspace, `${benchmarkCase.name}-${method}-${iteration}`);
        const started = performance.now();
        result = build(method, benchmarkCase, entry, outputRoot);
        times.push(performance.now() - started);
      }
      inspect(benchmarkCase, result);
      console.log([
        benchmarkCase.name,
        method,
        median(times).toFixed(3),
        (directoryBytes(result.root) / 1_000).toFixed(3),
        files(result.root).length,
      ].join(","));
    }
  }
  console.log("note: timings include process startup, discovery, transform, link, render, mkdir, and writes");
} finally {
  rmSync(workspace, { recursive: true, force: true });
}

function build(method, benchmarkCase, entry, root) {
  rmSync(root, { recursive: true, force: true });
  const output = method === "diffpack"
    ? join(root, "bundle.cjs")
    : benchmarkCase.codeSplitting
      ? root
      : join(root, "bundle.cjs");
  let command;
  let commandArguments;
  if (method === "diffpack") {
    command = diffpack;
    commandArguments = ["bundle", entry, output];
    if (benchmarkCase.sourceMap) commandArguments.push("--sourcemap");
  } else {
    command = rolldown;
    commandArguments = [entry, "--format", "cjs", "--logLevel", "silent"];
    if (benchmarkCase.codeSplitting) commandArguments.push("--dir", output);
    else commandArguments.push("--file", output, "--no-codeSplitting");
    if (benchmarkCase.sourceMap) commandArguments.push("--sourcemap");
  }
  const run = spawnSync(command, commandArguments, { cwd: repositoryRoot, encoding: "utf8" });
  if (run.status !== 0) {
    throw new Error(`${method} ${benchmarkCase.name} failed: ${run.stderr || run.stdout}`);
  }
  const entryOutput = benchmarkCase.codeSplitting && method === "rolldown"
    ? join(root, readdirSync(root).find((file) => file.startsWith("entry") && file.endsWith(".js")))
    : output;
  return { root, entryOutput };
}

function inspect(benchmarkCase, result) {
  const run = spawnSync(process.execPath, [result.entryOutput], { encoding: "utf8" });
  const stdout = String(run.stdout).replaceAll("\r\n", "\n");
  if (run.status !== 0 || stdout !== benchmarkCase.expected) {
    throw new Error(`${benchmarkCase.name} runtime mismatch: ${JSON.stringify(stdout)}`);
  }
  if (benchmarkCase.sentinel) {
    const output = files(result.root)
      .filter((path) => !path.endsWith(".map"))
      .map((path) => readFileSync(path, "utf8"))
      .join("\n");
    if (output.includes(benchmarkCase.sentinel)) {
      throw new Error(`${benchmarkCase.name} retained unused sentinel`);
    }
  }
  if (benchmarkCase.sourceMap) {
    const maps = files(result.root).filter((path) => path.endsWith(".map"));
    if (maps.length === 0) throw new Error(`${benchmarkCase.name} emitted no source map`);
  }
  if (benchmarkCase.codeSplitting && files(result.root).filter(isJavaScript).length < 2) {
    throw new Error(`${benchmarkCase.name} emitted no lazy chunks`);
  }
  return {
    status: run.status,
    stdout,
    maps: files(result.root).filter((path) => path.endsWith(".map")).length,
    chunks: files(result.root).filter(isJavaScript).length,
    bytes: directoryBytes(result.root),
  };
}

function assertPairEquivalent(benchmarkCase, diffpackResult, rolldownResult) {
  if (diffpackResult.status !== rolldownResult.status
    || diffpackResult.stdout !== rolldownResult.stdout) {
    throw new Error(`${benchmarkCase.name} Diffpack and Rolldown runtime outputs differ`);
  }
  if (benchmarkCase.sourceMap && diffpackResult.maps !== rolldownResult.maps) {
    throw new Error(`${benchmarkCase.name} source-map artifact counts differ`);
  }
  if (benchmarkCase.codeSplitting && diffpackResult.chunks !== rolldownResult.chunks) {
    throw new Error(`${benchmarkCase.name} chunk counts differ`);
  }
  if (diffpackResult.bytes > rolldownResult.bytes) {
    throw new Error(
      `${benchmarkCase.name} Diffpack output is larger: ${diffpackResult.bytes} > ${rolldownResult.bytes}`,
    );
  }
}

function files(root) {
  const entries = [];
  const pending = [root];
  while (pending.length > 0) {
    const path = pending.pop();
    const stat = statSync(path);
    if (stat.isDirectory()) {
      for (const entry of readdirSync(path)) pending.push(join(path, entry));
    } else {
      entries.push(path);
    }
  }
  return entries;
}

function directoryBytes(root) {
  return files(root).reduce((total, path) => total + statSync(path).size, 0);
}

function isJavaScript(path) {
  return path.endsWith(".js") || path.endsWith(".cjs");
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

function positiveInteger(value) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) throw new Error("iterations must be positive");
  return parsed;
}
