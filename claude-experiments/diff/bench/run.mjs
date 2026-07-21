// Competitive benchmark harness: diffpack vs esbuild, rolldown, rspack, vite.
// See bench/README.md for usage and docs/COMPETITIVE_BENCHMARKS.md for the
// methodology and measured results.
//
// Hard rules implemented here:
//  - Every (bundler, corpus) pair is runtime-verified: the emitted bundle is
//    executed under node and must print the independently computed value.
//    A pair that fails verification is reported as a named gap, never timed.
//  - Cold runs are fresh processes; per-tool cache/output paths are deleted
//    before every run (the exact deletion list is in toolSpecs below and in
//    the methodology doc).
import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { generateCorpus, expectedFreshValue, moduleName, editedModuleIndex, moduleSource } from "./gen.mjs";
import { median, round, timeProcess, peakRss, outputBytes, removePaths, verifyBundleValue } from "./util.mjs";

const benchRoot = dirname(fileURLToPath(import.meta.url));
const repoRoot = dirname(benchRoot);
const diffpackBinary = join(repoRoot, "target", "release", "diffpack");
const esbuildBin = join(benchRoot, "node_modules", ".bin", "esbuild");
const resultsDir = join(benchRoot, "results");
const resultsFile = join(resultsDir, "results.json");

const args = parseArgs(process.argv.slice(2));
const profiles = args.profiles ?? ["tiny", "realistic"];
const sizes = args.sizes ?? [1000, 10000];
const tools = args.tools ?? ["diffpack", "esbuild", "rolldown", "rspack", "vite"];
const coldRuns = args.coldRuns ?? 5;
const timedEdits = args.edits ?? 5;
const warmupEdits = args.warmupEdits ?? 2;
const appRuns = args.appRuns ?? 5;
const importsPerModule = args.imports ?? 4;

if (!existsSync(diffpackBinary)) {
  throw new Error(`missing ${diffpackBinary}; run \`cargo build --release\` first`);
}

mkdirSync(resultsDir, { recursive: true });
const results = existsSync(resultsFile)
  ? JSON.parse(readFileSync(resultsFile, "utf8"))
  : { meta: {}, corpora: {}, app: {} };
results.meta = collectMeta();
saveResults();

const workRoot = mkdtempSync(join(tmpdir(), "diffpack-competitive-bench-"));
console.log(`work directory: ${workRoot}`);

try {
  if (!args.onlyApp) {
    for (const profile of profiles) {
      for (const moduleCount of sizes) {
        await benchCorpus(profile, moduleCount);
      }
    }
  }
  if (!args.skipApp) {
    benchApp();
  }
  printSummary();
} finally {
  if (!args.keepWork) rmSync(workRoot, { recursive: true, force: true });
  else console.log(`kept work directory: ${workRoot}`);
}

async function benchCorpus(profile, moduleCount) {
  const key = `${profile}-${moduleCount}`;
  const corpusDir = join(workRoot, `corpus-${key}`);
  const outRoot = join(workRoot, `out-${key}`);
  mkdirSync(outRoot, { recursive: true });
  console.log(`\n=== corpus ${key}: generating ${moduleCount} modules (${importsPerModule} imports/module) ===`);
  const { sourceBytes } = generateCorpus(corpusDir, { profile, moduleCount, importsPerModule });
  const entry = join(corpusDir, moduleName(0));
  const expected = expectedFreshValue(moduleCount);
  console.log(`source: ${(sourceBytes / 1e6).toFixed(2)} MB, expected runtime value: ${expected}`);

  const cell = (results.corpora[key] ??= {
    profile,
    moduleCount,
    importsPerModule,
    sourceBytes,
    tools: {},
  });
  cell.sourceBytes = sourceBytes;

  const specs = toolSpecs({ corpusDir, entry, outRoot });
  for (const tool of tools) {
    const spec = specs[tool];
    if (!spec) throw new Error(`unknown tool: ${tool}`);
    console.log(`\n--- ${key} / ${tool} ---`);
    const record = { verified: false };
    cell.tools[tool] = record;
    try {
      // Warmup run: populates the OS page cache for tool binaries/scripts and
      // acts as the initial verification gate. Not counted.
      removePaths(spec.cleanup);
      timeProcess(...spec.command);
      verifyBundleValue(spec.verifyFile, expected, `${tool} warmup ${key}`);
      record.verified = true;

      const cold = [];
      for (let run = 0; run < coldRuns; run += 1) {
        removePaths(spec.cleanup);
        const { elapsedMs } = timeProcess(...spec.command);
        verifyBundleValue(spec.verifyFile, expected, `${tool} cold run ${run} ${key}`);
        cold.push(elapsedMs);
      }
      record.coldMs = cold.map((v) => round(v, 1));
      record.coldMedianMs = round(median(cold), 1);

      const sizesOut = outputBytes(spec.outputPath);
      record.outputBytes = sizesOut.raw;
      record.outputGzipBytes = sizesOut.gzip;
      record.outputFiles = sizesOut.files;

      removePaths(spec.cleanup);
      record.peakRssBytes = peakRss(...spec.command);
      verifyBundleValue(spec.verifyFile, expected, `${tool} rss run ${key}`);
      console.log(
        `cold median ${record.coldMedianMs} ms (${record.coldMs.join(", ")}), peak RSS ${(record.peakRssBytes / 1e6).toFixed(0)} MB, output ${record.outputBytes} B (${record.outputGzipBytes} B gzip)`,
      );
    } catch (error) {
      record.error = String(error.message ?? error);
      console.error(`EXCLUDED from timing: ${tool} on ${key}: ${record.error}`);
      saveResults();
      continue;
    }

    if (spec.incremental === null) {
      record.incremental = { skipped: spec.incrementalSkipReason };
      console.log(`incremental: skipped (${spec.incrementalSkipReason})`);
    } else {
      try {
        removePaths(spec.cleanup);
        const driverArgs = JSON.stringify({
          corpusDir,
          entry,
          profile,
          moduleCount,
          importsPerModule,
          warmupEdits,
          timedEdits,
          diffpackBinary,
          ...spec.incremental,
        });
        const result = spawnSync("node", [join(benchRoot, "tools", spec.incrementalDriver), driverArgs], {
          encoding: "utf8",
          cwd: benchRoot,
          timeout: 30 * 60 * 1000,
        });
        if (result.status !== 0) {
          throw new Error(
            `incremental driver exited ${result.status}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`,
          );
        }
        const line = result.stdout.split("\n").find((candidate) => candidate.startsWith("RESULT "));
        if (!line) throw new Error(`incremental driver produced no RESULT line:\n${result.stdout}`);
        const { rebuildMs } = JSON.parse(line.slice("RESULT ".length));
        record.incremental = {
          rebuildMs: rebuildMs.map((v) => round(v, 2)),
          medianMs: round(median(rebuildMs), 2),
        };
        console.log(
          `incremental median ${record.incremental.medianMs} ms (${record.incremental.rebuildMs.join(", ")})`,
        );
        verifyCorpusRestored(corpusDir, profile, moduleCount);
      } catch (error) {
        record.incremental = { error: String(error.message ?? error) };
        console.error(`incremental EXCLUDED: ${tool} on ${key}: ${record.incremental.error}`);
        // The incremental drivers may have left the edited module in a
        // non-original state; regenerate it so later tools see a clean corpus.
        const index = editedModuleIndex(moduleCount);
        writeFileSync(
          join(corpusDir, moduleName(index)),
          moduleSource(profile, index, moduleCount, importsPerModule, index),
        );
      }
    }
    saveResults();
  }
}

function verifyCorpusRestored(corpusDir, profile, moduleCount) {
  const index = editedModuleIndex(moduleCount);
  const expected = moduleSource(profile, index, moduleCount, importsPerModule, index);
  const actual = readFileSync(join(corpusDir, moduleName(index)), "utf8");
  if (actual !== expected) {
    throw new Error(`incremental driver did not restore ${moduleName(index)} to its original content`);
  }
}

function toolSpecs({ corpusDir, entry, outRoot }) {
  const diffpackOut = join(outRoot, "diffpack.mjs");
  const esbuildOut = join(outRoot, "esbuild.mjs");
  const rolldownOut = join(outRoot, "rolldown.mjs");
  const rspackOutDir = join(outRoot, "rspack");
  const viteOutDir = join(outRoot, "vite");
  return {
    diffpack: {
      command: [diffpackBinary, ["bundle", entry, diffpackOut]],
      cleanup: [diffpackOut],
      verifyFile: diffpackOut,
      outputPath: diffpackOut,
      incrementalDriver: "incr-diffpack.mjs",
      incremental: { outfile: join(outRoot, "diffpack-watch.mjs") },
    },
    esbuild: {
      command: [
        esbuildBin,
        [entry, "--bundle", "--format=esm", "--target=esnext", `--outfile=${esbuildOut}`, "--log-level=silent"],
      ],
      cleanup: [esbuildOut],
      verifyFile: esbuildOut,
      outputPath: esbuildOut,
      incrementalDriver: "incr-esbuild.mjs",
      incremental: { outfile: join(outRoot, "esbuild-incr.mjs") },
    },
    rolldown: {
      command: ["node", [join(benchRoot, "tools", "cold-rolldown.mjs"), entry, rolldownOut]],
      cleanup: [rolldownOut],
      verifyFile: rolldownOut,
      outputPath: rolldownOut,
      incrementalDriver: "incr-rolldown.mjs",
      incremental: { outfile: join(outRoot, "rolldown-incr.mjs") },
    },
    rspack: {
      command: ["node", [join(benchRoot, "tools", "cold-rspack.mjs"), entry, rspackOutDir]],
      cleanup: [rspackOutDir, join(corpusDir, "node_modules")],
      verifyFile: join(rspackOutDir, "bundle.mjs"),
      outputPath: rspackOutDir,
      incrementalDriver: "incr-rspack.mjs",
      incremental: { outdir: join(outRoot, "rspack-incr") },
    },
    vite: {
      command: ["node", [join(benchRoot, "tools", "cold-vite.mjs"), corpusDir, entry, viteOutDir]],
      cleanup: [viteOutDir, join(corpusDir, ".vite-cache"), join(corpusDir, "node_modules")],
      verifyFile: join(viteOutDir, "bundle.mjs"),
      outputPath: viteOutDir,
      incremental: null,
      incrementalSkipReason:
        "vite's incremental story is its dev-server HMR (unbundled/on-demand transform), a different axis than batch rebuild time; vite build has no in-process rebuild API",
    },
  };
}

function benchApp() {
  const fixture = join(repoRoot, "integration", "tanstack-start-reference");
  if (!existsSync(join(fixture, "node_modules"))) {
    throw new Error(
      `${fixture}/node_modules is missing; run npm install there first (npm ci currently fails: the pinned lockfile is out of sync)`,
    );
  }
  console.log("\n=== real app: integration/tanstack-start-reference ===");
  const viteBin = join(fixture, "node_modules", ".bin", "vite");
  const diffpackOut = join(fixture, ".diffpack-output");
  const viteCleanup = [
    join(fixture, ".output"),
    join(fixture, ".tanstack"),
    join(fixture, "node_modules", ".vite"),
    join(fixture, "node_modules", ".tmp"),
    join(fixture, "node_modules", ".cache"),
  ];

  const cases = [
    {
      name: "diffpack-client",
      command: [diffpackBinary, ["build-app", ".", "client"], { cwd: fixture }],
      cleanup: () => removePaths([diffpackOut]),
      outputPath: () => join(diffpackOut, "public"),
    },
    {
      name: "diffpack-ssr",
      // The ssr build consumes the client build's route manifest, so a client
      // build must exist; only the server output is removed per run.
      prepare: () => {
        removePaths([diffpackOut]);
        timeProcess(diffpackBinary, ["build-app", ".", "client"], { cwd: fixture });
      },
      command: [diffpackBinary, ["build-app", ".", "ssr"], { cwd: fixture }],
      cleanup: () => removePaths([join(diffpackOut, "server")]),
      outputPath: () => join(diffpackOut, "server"),
    },
    {
      name: "vite-build",
      command: [viteBin, ["build"], { cwd: fixture }],
      cleanup: () => removePaths(viteCleanup),
      outputPath: () => join(fixture, ".output"),
    },
    {
      name: "npm-run-build (vite build && tsc --noEmit)",
      command: ["npm", ["run", "build"], { cwd: fixture }],
      cleanup: () => removePaths(viteCleanup),
      outputPath: null,
    },
  ];

  for (const benchCase of cases) {
    console.log(`\n--- app / ${benchCase.name} ---`);
    const record = {};
    results.app[benchCase.name] = record;
    try {
      benchCase.prepare?.();
      benchCase.cleanup();
      timeProcess(...benchCase.command); // warmup, not counted
      const wall = [];
      for (let run = 0; run < appRuns; run += 1) {
        benchCase.cleanup();
        const { elapsedMs } = timeProcess(...benchCase.command);
        wall.push(elapsedMs);
      }
      record.wallMs = wall.map((v) => round(v, 0));
      record.wallMedianMs = round(median(wall), 0);
      benchCase.cleanup();
      record.peakRssBytes = peakRss(...benchCase.command);
      if (benchCase.outputPath) {
        const sizesOut = outputBytes(benchCase.outputPath());
        record.outputBytes = sizesOut.raw;
        record.outputGzipBytes = sizesOut.gzip;
        record.outputFiles = sizesOut.files;
      }
      console.log(
        `wall median ${record.wallMedianMs} ms (${record.wallMs.join(", ")}), peak RSS ${(record.peakRssBytes / 1e6).toFixed(0)} MB`,
      );
    } catch (error) {
      record.error = String(error.message ?? error);
      console.error(`app case FAILED: ${benchCase.name}: ${record.error}`);
    }
    saveResults();
  }

  // vite's TanStack plugin rewrites src/routeTree.gen.ts; restore the pinned
  // fixture to its committed state so the working tree stays clean.
  execFileSync("git", ["checkout", "--", "src/routeTree.gen.ts"], { cwd: fixture });
}

function collectMeta() {
  const cpu = readFileSync("/proc/cpuinfo", "utf8")
    .split("\n")
    .find((line) => line.startsWith("model name"))
    ?.split(": ")[1]
    ?.trim();
  const versions = {};
  for (const name of ["esbuild", "rolldown", "@rspack/core", "vite"]) {
    try {
      versions[name] = JSON.parse(
        readFileSync(join(benchRoot, "node_modules", name, "package.json"), "utf8"),
      ).version;
    } catch {
      versions[name] = "not installed";
    }
  }
  let commit = "unknown";
  try {
    commit = execFileSync("git", ["rev-parse", "--short", "HEAD"], { cwd: repoRoot, encoding: "utf8" }).trim();
  } catch {}
  return {
    date: new Date().toISOString(),
    cpu,
    node: process.version,
    diffpackCommit: commit,
    versions,
  };
}

function printSummary() {
  console.log("\n================ SUMMARY ================");
  console.log(JSON.stringify(results.meta, null, 2));
  for (const [key, cell] of Object.entries(results.corpora)) {
    console.log(`\n## ${key} (${(cell.sourceBytes / 1e6).toFixed(2)} MB source)`);
    console.log("| tool | cold median ms | incremental median ms | peak RSS MB | output bytes | gzip bytes |");
    console.log("| --- | ---: | ---: | ---: | ---: | ---: |");
    for (const [tool, record] of Object.entries(cell.tools)) {
      if (record.error) {
        console.log(`| ${tool} | EXCLUDED: ${record.error.split("\n")[0].slice(0, 80)} | | | | |`);
        continue;
      }
      const incremental = record.incremental?.medianMs ?? (record.incremental?.skipped ? "skipped" : "n/a");
      console.log(
        `| ${tool} | ${record.coldMedianMs} | ${incremental} | ${(record.peakRssBytes / 1e6).toFixed(0)} | ${record.outputBytes} | ${record.outputGzipBytes} |`,
      );
    }
  }
  if (Object.keys(results.app).length > 0) {
    console.log("\n## real app (integration/tanstack-start-reference)");
    console.log("| case | wall median ms | peak RSS MB | output bytes |");
    console.log("| --- | ---: | ---: | ---: |");
    for (const [name, record] of Object.entries(results.app)) {
      if (record.error) {
        console.log(`| ${name} | FAILED: ${record.error.split("\n")[0].slice(0, 80)} | | |`);
        continue;
      }
      console.log(
        `| ${name} | ${record.wallMedianMs} | ${(record.peakRssBytes / 1e6).toFixed(0)} | ${record.outputBytes ?? ""} |`,
      );
    }
  }
  console.log(`\nresults saved to ${resultsFile}`);
}

function saveResults() {
  writeFileSync(resultsFile, `${JSON.stringify(results, null, 2)}\n`);
}

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    const next = () => argv[++index];
    if (argument === "--profiles") parsed.profiles = next().split(",");
    else if (argument === "--sizes") parsed.sizes = next().split(",").map(Number);
    else if (argument === "--tools") parsed.tools = next().split(",");
    else if (argument === "--cold-runs") parsed.coldRuns = Number(next());
    else if (argument === "--edits") parsed.edits = Number(next());
    else if (argument === "--warmup-edits") parsed.warmupEdits = Number(next());
    else if (argument === "--app-runs") parsed.appRuns = Number(next());
    else if (argument === "--imports") parsed.imports = Number(next());
    else if (argument === "--skip-app") parsed.skipApp = true;
    else if (argument === "--only-app") parsed.onlyApp = true;
    else if (argument === "--keep-work") parsed.keepWork = true;
    else throw new Error(`unknown argument: ${argument}`);
  }
  return parsed;
}
