// Execution-differential conformance runner.
//
// Ground truth: each fixture's entry executed directly by Node (unbundled).
// The recorded stdout lives in <fixture>/expected.txt and the expected exit
// class in <fixture>/expected-exit.txt (absent means exit 0). Regenerate with
// `node run.mjs --update-expected` — expectations are never hand-written.
//
// Each fixture is then bundled with diffpack, rolldown, and esbuild, each
// bundle is executed by Node, and the result is classified per bundler:
//   PASS          stdout and exit class match Node's unbundled run
//   WRONG-OUTPUT  bundle ran but produced different stdout (silent wrongness)
//   BUILD-ERROR   the bundler refused / hard-errored
//   RUNTIME-ERROR bundle crashed where Node's unbundled run succeeded
//
// Exit code is nonzero only if diffpack has any WRONG-OUTPUT result;
// BUILD-ERROR is diffpack's documented honest-hard-error policy and is
// recorded as a gap, not a runner failure.

import { spawnSync } from "node:child_process";
import {
  existsSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { rolldown } from "rolldown";
import * as esbuild from "esbuild";

const conformanceRoot = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = dirname(conformanceRoot);
const fixturesRoot = join(conformanceRoot, "fixtures");
const diffpackBinary = join(repositoryRoot, "target", "release", "diffpack");

const argv = process.argv.slice(2);
const updateExpected = argv.includes("--update-expected");
const filters = argv.filter((a) => !a.startsWith("--"));

const ENTRY_CANDIDATES = ["entry.js", "entry.mjs", "entry.cjs", "entry.ts"];

const allFixtures = readdirSync(fixturesRoot, { withFileTypes: true })
  .filter((d) => d.isDirectory())
  .map((d) => d.name)
  .sort();
const fixtures = filters.length === 0
  ? allFixtures
  : allFixtures.filter((name) => filters.some((f) => name.includes(f)));
if (fixtures.length === 0) {
  console.error(`no fixtures matched: ${filters.join(", ")}`);
  process.exit(2);
}

function entryFor(fixture) {
  for (const candidate of ENTRY_CANDIDATES) {
    const path = join(fixturesRoot, fixture, candidate);
    if (existsSync(path)) return path;
  }
  throw new Error(`fixture ${fixture} has no entry file`);
}

function runNode(script) {
  const result = spawnSync(process.execPath, [script], {
    encoding: "utf8",
    timeout: 15_000,
    cwd: dirname(script),
  });
  return {
    status: result.status ?? 1,
    stdout: String(result.stdout ?? "").replaceAll("\r\n", "\n"),
    stderr: String(result.stderr ?? result.error?.message ?? "").replaceAll("\r\n", "\n"),
  };
}

async function referenceGroundTruth(fixture) {
  const dir = mkdtempSync(join(tmpdir(), "diffpack-conformance-gt-"));
  try {
    const entry = entryFor(fixture);
    const rolldownBuilt = await buildRolldown(entry, dir);
    const esbuildBuilt = await buildEsbuild(entry, dir);
    if (!rolldownBuilt.ok || !esbuildBuilt.ok) {
      throw new Error(
        `${fixture}: reference ground truth requires both reference builds to succeed ` +
        `(rolldown: ${rolldownBuilt.ok}, esbuild: ${esbuildBuilt.ok})`,
      );
    }
    const rolldownRun = runNode(rolldownBuilt.output);
    const esbuildRun = runNode(esbuildBuilt.output);
    const sameClass = (rolldownRun.status === 0) === (esbuildRun.status === 0);
    if (!sameClass || rolldownRun.stdout !== esbuildRun.stdout) {
      throw new Error(
        `${fixture}: rolldown and esbuild disagree, cannot establish reference ground truth\n` +
        `  rolldown: exit ${rolldownRun.status} stdout ${JSON.stringify(rolldownRun.stdout)}\n` +
        `  esbuild:  exit ${esbuildRun.status} stdout ${JSON.stringify(esbuildRun.stdout)}`,
      );
    }
    return rolldownRun;
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

// ---------------------------------------------------------------------------
// --update-expected: record Node's unbundled behavior as ground truth.
//
// Exception: a fixture containing `node-cannot-run.md` cannot be executed
// unbundled by Node (the file documents why). For those, ground truth is the
// agreement of the two reference bundlers (rolldown + esbuild): both are
// built and run, and they must produce identical stdout and exit class.
// ---------------------------------------------------------------------------
if (updateExpected) {
  for (const fixture of fixtures) {
    const dir = join(fixturesRoot, fixture);
    let run;
    if (existsSync(join(dir, "node-cannot-run.md"))) {
      run = await referenceGroundTruth(fixture);
      console.log(`  (${fixture}: ground truth from rolldown+esbuild agreement, see node-cannot-run.md)`);
    } else {
      run = runNode(entryFor(fixture));
    }
    writeFileSync(join(dir, "expected.txt"), run.stdout);
    const exitPath = join(dir, "expected-exit.txt");
    if (run.status !== 0) {
      writeFileSync(exitPath, `${run.status}\n`);
    } else if (existsSync(exitPath)) {
      rmSync(exitPath);
    }
    console.log(`recorded ${fixture} (exit ${run.status}, ${run.stdout.split("\n").length - 1} stdout lines)`);
  }
  process.exit(0);
}

// ---------------------------------------------------------------------------
// Bundler drivers. All three produce a single self-contained file, all in the
// SAME output format: diffpack `--format esm` executed as `.mjs`, matching
// rolldown and esbuild's format=esm, platform=node, single chunk.
// ---------------------------------------------------------------------------
if (!existsSync(diffpackBinary)) {
  console.error(`diffpack binary missing at ${diffpackBinary}; run: cargo build --release`);
  process.exit(2);
}

async function buildDiffpack(entry, outDir) {
  const output = join(outDir, "diffpack.mjs");
  const result = spawnSync(diffpackBinary, ["bundle", entry, output, "--format", "esm"], {
    encoding: "utf8",
    timeout: 60_000,
  });
  if (result.status !== 0) {
    const message = stripAnsi(`${result.stderr ?? ""}${result.stdout ?? ""}`.trim()) || "nonzero exit";
    return { ok: false, message };
  }
  return { ok: true, output };
}

async function buildRolldown(entry, outDir) {
  const output = join(outDir, "rolldown.mjs");
  try {
    const bundle = await rolldown({
      input: entry,
      treeshake: false,
      logLevel: "silent",
    });
    await bundle.write({ file: output, format: "esm", codeSplitting: false });
    await bundle.close();
    return { ok: true, output };
  } catch (error) {
    return { ok: false, message: stripAnsi(String(error?.message ?? error)) };
  }
}

async function buildEsbuild(entry, outDir) {
  const output = join(outDir, "esbuild.mjs");
  try {
    await esbuild.build({
      entryPoints: [entry],
      bundle: true,
      format: "esm",
      platform: "node",
      outfile: output,
      logLevel: "silent",
    });
    return { ok: true, output };
  } catch (error) {
    const message = error?.errors?.length
      ? error.errors.map((e) => e.text).join("; ")
      : String(error?.message ?? error);
    return { ok: false, message };
  }
}

const bundlers = [
  ["diffpack", buildDiffpack],
  ["rolldown", buildRolldown],
  ["esbuild", buildEsbuild],
];

function classify(expected, run) {
  const stdoutMatches = run.stdout === expected.stdout;
  if (expected.exit === 0) {
    if (run.status !== 0) {
      return { class: "RUNTIME-ERROR", detail: firstLine(run.stderr) };
    }
    return stdoutMatches
      ? { class: "PASS" }
      : { class: "WRONG-OUTPUT", detail: diffSummary(expected.stdout, run.stdout) };
  }
  // Ground truth is a nonzero exit (e.g. an uncaught TDZ error): the bundle
  // must also fail, and the stdout emitted before the failure must match.
  if (run.status === 0) {
    return { class: "WRONG-OUTPUT", detail: "expected a runtime failure but the bundle exited 0" };
  }
  return stdoutMatches
    ? { class: "PASS" }
    : { class: "WRONG-OUTPUT", detail: diffSummary(expected.stdout, run.stdout) };
}

function firstLine(text) {
  const lines = String(text ?? "").split("\n");
  // Prefer the actual error line over Node warnings and stack frames.
  const error = lines.find((l) => /(error|cannot|unexpected)/i.test(l) && !l.startsWith("(node:") && !l.trim().startsWith("at "));
  return (error ?? lines.find((l) => l.trim() !== "") ?? "").trim();
}

function stripAnsi(text) {
  return String(text ?? "").replaceAll(/\x1b\[[0-9;]*m/g, "");
}

function diffSummary(expected, actual) {
  return `expected ${JSON.stringify(expected)} got ${JSON.stringify(actual)}`;
}

// ---------------------------------------------------------------------------
// Main loop.
// ---------------------------------------------------------------------------
const workspace = mkdtempSync(join(tmpdir(), "diffpack-conformance-"));
const results = {};

try {
  for (const fixture of fixtures) {
    const entry = entryFor(fixture);
    const expectedPath = join(fixturesRoot, fixture, "expected.txt");
    if (!existsSync(expectedPath)) {
      console.error(`fixture ${fixture} has no expected.txt; run --update-expected`);
      process.exit(2);
    }
    const exitPath = join(fixturesRoot, fixture, "expected-exit.txt");
    const expected = {
      stdout: readFileSync(expectedPath, "utf8"),
      exit: existsSync(exitPath) ? Number(readFileSync(exitPath, "utf8").trim()) : 0,
    };

    const outDir = join(workspace, fixture);
    const fixtureResults = {};
    for (const [name, build] of bundlers) {
      const outCase = join(outDir, name);
      spawnSync("mkdir", ["-p", outCase]);
      const built = await build(entry, outCase);
      if (!built.ok) {
        fixtureResults[name] = { class: "BUILD-ERROR", message: built.message };
        continue;
      }
      const run = runNode(built.output);
      const verdict = classify(expected, run);
      fixtureResults[name] = {
        class: verdict.class,
        ...(verdict.detail ? { detail: verdict.detail } : {}),
        exit: run.status,
        stdout: run.stdout,
      };
    }
    results[fixture] = { expected, bundlers: fixtureResults };
  }
} finally {
  rmSync(workspace, { recursive: true, force: true });
}

// ---------------------------------------------------------------------------
// Matrix + results.json + exit code.
// ---------------------------------------------------------------------------
const CLASS_SHORT = {
  "PASS": "pass",
  "WRONG-OUTPUT": "WRONG",
  "BUILD-ERROR": "build-err",
  "RUNTIME-ERROR": "run-err",
};

const nameWidth = Math.max(...fixtures.map((f) => f.length), 7);
const col = 11;
console.log(
  "fixture".padEnd(nameWidth),
  ...bundlers.map(([n]) => n.padEnd(col)),
);
for (const fixture of fixtures) {
  console.log(
    fixture.padEnd(nameWidth),
    ...bundlers.map(([n]) => CLASS_SHORT[results[fixture].bundlers[n].class].padEnd(col)),
  );
}

const counts = {};
for (const [name] of bundlers) {
  counts[name] = { "PASS": 0, "WRONG-OUTPUT": 0, "BUILD-ERROR": 0, "RUNTIME-ERROR": 0 };
  for (const fixture of fixtures) {
    counts[name][results[fixture].bundlers[name].class] += 1;
  }
}
console.log("");
for (const [name] of bundlers) {
  const c = counts[name];
  console.log(
    `${name}: ${c["PASS"]} pass, ${c["WRONG-OUTPUT"]} wrong-output, ` +
    `${c["BUILD-ERROR"]} build-error, ${c["RUNTIME-ERROR"]} runtime-error ` +
    `(of ${fixtures.length})`,
  );
}

for (const [name] of bundlers) {
  for (const cls of ["WRONG-OUTPUT", "RUNTIME-ERROR", "BUILD-ERROR"]) {
    const bad = fixtures.filter((f) => results[f].bundlers[name].class === cls);
    if (bad.length > 0) {
      console.log(`\n${name} ${cls}:`);
      for (const fixture of bad) {
        const r = results[fixture].bundlers[name];
        console.log(`  ${fixture}: ${firstLine(r.detail ?? r.message ?? "")}`);
      }
    }
  }
}

writeFileSync(
  join(conformanceRoot, "results.json"),
  JSON.stringify(
    {
      generated: new Date().toISOString(),
      node: process.version,
      bundlers: {
        diffpack: "target/release/diffpack (local build)",
        rolldown: "1.2.0",
        esbuild: JSON.parse(readFileSync(join(conformanceRoot, "node_modules/esbuild/package.json"), "utf8")).version,
      },
      counts,
      fixtures: results,
    },
    null,
    2,
  ) + "\n",
);
console.log(`\nwrote results.json (${fixtures.length} fixtures)`);

process.exitCode = counts.diffpack["WRONG-OUTPUT"] > 0 ? 1 : 0;
