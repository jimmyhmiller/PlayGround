// Next.js build benchmark: diffpack vs `next build --turbopack` vs `next build --webpack`,
// on the SAME real Next app-router fixture (integration/next-app-router).
//
// HONEST SCOPE — read before trusting any number this prints:
//   These three commands DO NOT do the same work, so the wall/RSS gaps are NOT a
//   pure bundler-vs-bundler result. What each actually does:
//     • next build (turbopack | webpack): the FULL Next framework pipeline —
//       route/type generation (.next/types), TypeScript type-checking, ESLint,
//       app-router compilation of the client + server + RSC graphs, minification,
//       PRERENDERING every static route to HTML+RSC (.next/server/app/*.html),
//       image/prerender/routes manifests, and build traces. Output → .next/.
//     • diffpack build-app (client + react-server + ssr): bundles the three RSC
//       graphs + the app-router adapter glue and emits a Node orchestrator. It does
//       NOT type-check, lint, generate route types, or prerender — routes render
//       per-request via scripts/rsc/next-server.mjs. Output → .diffpack-output/.
//   So diffpack is measuring "produce the shippable bundles"; next is measuring
//   "produce the whole framework build incl. typecheck + SSG". The table labels
//   this on every row. It is a legitimate END-TO-END "time to a deployable build"
//   comparison; it is NOT a like-for-like "who bundles faster" microbenchmark.
//
// Methodology mirrors bench/run.mjs: fresh process per run, per-case output dir
// deleted before EVERY run (true cold — next's .next/cache persistent cache is
// wiped too), one uncounted warmup, then N timed cold runs (median reported),
// peak RSS measured out-of-process via `vtime -m` / GNU time, output bytes via
// gzip. Every build is verified (a known emitted artifact must exist) or the case
// is EXCLUDED with its reason, never silently timed.
//
// Usage:
//   node scripts/bench-next.mjs                 # all three cases, 5 cold runs
//   node scripts/bench-next.mjs --runs 3
//   node scripts/bench-next.mjs --cases diffpack-next,next-turbopack
//   node scripts/bench-next.mjs --fixture integration/next-app-router

import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync, cpSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import os from "node:os";
import { median, round, timeProcess, peakRss, outputBytes, removePaths } from "../bench/util.mjs";

const scriptsDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = dirname(scriptsDir);
const diffpack = join(repoRoot, "target", "release", "diffpack");

const args = parseArgs(process.argv.slice(2));
const fixture = join(repoRoot, args.fixture ?? "integration/next-app-router");
const coldRuns = args.runs ?? 5;
const allCases = ["diffpack-next", "next-turbopack", "next-webpack"];
const cases = args.cases ?? allCases;

const nextBin = join(fixture, "node_modules", ".bin", "next");
const nextOut = join(fixture, ".next");
const dpOut = join(fixture, ".diffpack-output");
const nextEnv = { ...process.env, NEXT_TELEMETRY_DISABLED: "1" };

if (!existsSync(diffpack)) throw new Error(`missing ${diffpack}; run \`cargo build --release\` first`);
if (!existsSync(join(fixture, "node_modules"))) {
  throw new Error(`${fixture}/node_modules is missing; run \`npm install\` in the fixture first (next + turbopack ship together)`);
}
if (!existsSync(nextBin)) throw new Error(`missing ${nextBin}; the fixture's node_modules has no next binary`);

const resultsDir = join(repoRoot, "bench", "results");
mkdirSync(resultsDir, { recursive: true });
const resultsFile = join(resultsDir, "next-results.json");
const results = { meta: collectMeta(), fixture: args.fixture ?? "integration/next-app-router", cases: {} };

// One diffpack "next build" = the three-graph sequence from scripts/rsc/next-check.sh.
function diffpackSequence() {
  const opts = { cwd: fixture };
  const wallSteps = [];
  let rss = 0;
  const steps = [
    () => timeProcess(diffpack, ["build-app", ".", "client", "--no-minify"], opts),
    () => timeProcess(diffpack, ["build-app", ".", "react-server", "--no-minify"], opts),
    () => {
      // Snapshot the react-server output aside before the ssr build overwrites server/.
      removePaths([join(dpOut, "rsc-render")]);
      const t0 = process.hrtime.bigint();
      cpSync(join(dpOut, "server"), join(dpOut, "rsc-render"), { recursive: true });
      return { elapsedMs: Number(process.hrtime.bigint() - t0) / 1e6 };
    },
    () => timeProcess(diffpack, ["build-app", ".", "ssr", "--no-minify"], opts),
  ];
  for (const step of steps) wallSteps.push(step().elapsedMs);
  return { elapsedMs: wallSteps.reduce((a, b) => a + b, 0), rss };
}

const specs = {
  "diffpack-next": {
    label: "produces bundles only — NO typecheck/lint/SSG (renders per-request)",
    cleanup: () => removePaths([dpOut]),
    run: () => diffpackSequence(),
    // RSS = max over the three build invocations (they run sequentially; peak is the max).
    rss: () => {
      removePaths([dpOut]);
      const client = peakRss(diffpack, ["build-app", ".", "client", "--no-minify"], { cwd: fixture });
      const rs = peakRss(diffpack, ["build-app", ".", "react-server", "--no-minify"], { cwd: fixture });
      removePaths([join(dpOut, "rsc-render")]);
      cpSync(join(dpOut, "server"), join(dpOut, "rsc-render"), { recursive: true });
      const ssr = peakRss(diffpack, ["build-app", ".", "ssr", "--no-minify"], { cwd: fixture });
      return Math.max(client, rs, ssr);
    },
    verify: () => {
      if (!existsSync(join(dpOut, "public", "client.js"))) throw new Error("no client.js in .diffpack-output/public");
      if (!existsSync(join(dpOut, "server", "index.mjs"))) throw new Error("no server/index.mjs in .diffpack-output");
    },
    outputs: () => ({
      "client (public/)": outputBytes(join(dpOut, "public")),
      "server (server/)": outputBytes(join(dpOut, "server")),
    }),
  },
  "next-turbopack": {
    label: "FULL framework build (typecheck+lint+SSG) via Turbopack",
    cleanup: () => removePaths([nextOut]),
    command: [nextBin, ["build", "--turbopack"], { cwd: fixture, env: nextEnv }],
    verify: () => verifyNext(),
    outputs: () => nextOutputs(),
  },
  "next-webpack": {
    label: "FULL framework build (typecheck+lint+SSG) via webpack",
    cleanup: () => removePaths([nextOut]),
    command: [nextBin, ["build", "--webpack"], { cwd: fixture, env: nextEnv }],
    verify: () => verifyNext(),
    outputs: () => nextOutputs(),
  },
};

function verifyNext() {
  const html = join(nextOut, "server", "app", "index.html");
  if (!existsSync(html)) throw new Error(`next build did not prerender ${html}`);
}

// Report the SHIPPABLE next output split, and total .next excluding the build cache
// (.next/cache is a persistent incremental cache, never deployed).
function nextOutputs() {
  return {
    "static (.next/static, client)": outputBytes(join(nextOut, "static")),
    "server (.next/server)": outputBytes(join(nextOut, "server")),
  };
}

try {
  for (const name of cases) {
    const spec = specs[name];
    if (!spec) throw new Error(`unknown case: ${name} (known: ${allCases.join(", ")})`);
    console.log(`\n--- ${name}: ${spec.label} ---`);
    const record = { label: spec.label };
    results.cases[name] = record;
    try {
      spec.cleanup();
      (spec.command ? timeProcess(...spec.command) : spec.run()); // warmup, uncounted
      spec.verify();

      const wall = [];
      for (let i = 0; i < coldRuns; i += 1) {
        spec.cleanup();
        const { elapsedMs } = spec.command ? timeProcess(...spec.command) : spec.run();
        spec.verify();
        wall.push(elapsedMs);
      }
      record.wallMs = wall.map((v) => round(v, 0));
      record.wallMedianMs = round(median(wall), 0);

      spec.cleanup();
      record.peakRssBytes = spec.command ? peakRss(...spec.command) : spec.rss();
      spec.verify();

      record.outputs = {};
      for (const [k, v] of Object.entries(spec.outputs())) {
        record.outputs[k] = { bytes: v.raw, gzipBytes: v.gzip, files: v.files };
      }
      const outStr = Object.entries(record.outputs)
        .map(([k, v]) => `${k}=${(v.bytes / 1e3).toFixed(0)}KB/${v.files}f`)
        .join(", ");
      console.log(
        `wall median ${record.wallMedianMs} ms (${record.wallMs.join(", ")}), peak RSS ${(record.peakRssBytes / 1e6).toFixed(0)} MB, output ${outStr}`,
      );
    } catch (error) {
      record.error = String(error.message ?? error).split("\n")[0];
      console.error(`EXCLUDED ${name}: ${record.error}`);
    }
    writeFileSync(resultsFile, `${JSON.stringify(results, null, 2)}\n`);
  }
  printTable();
} finally {
  // next build touches next-env.d.ts / tsconfig.json; keep the working tree clean.
  for (const f of ["next-env.d.ts", "tsconfig.json"]) {
    spawnSync("git", ["checkout", "--", f], { cwd: fixture });
  }
}

function printTable() {
  console.log("\n================ diffpack vs Next build (SAME app, DIFFERENT work — see labels) ================");
  console.log(JSON.stringify(results.meta, null, 2));
  console.log("\n| case | measures | wall median | peak RSS | client out | server out |");
  console.log("| --- | --- | ---: | ---: | ---: | ---: |");
  for (const name of cases) {
    const r = results.cases[name];
    if (!r) continue;
    if (r.error) {
      console.log(`| ${name} | ${r.label} | EXCLUDED: ${r.error.slice(0, 50)} | | | |`);
      continue;
    }
    const outs = Object.values(r.outputs);
    const clientOut = outs[0] ? `${(outs[0].bytes / 1e3).toFixed(0)} KB (${outs[0].files}f)` : "";
    const serverOut = outs[1] ? `${(outs[1].bytes / 1e3).toFixed(0)} KB (${outs[1].files}f)` : "";
    console.log(
      `| ${name} | ${r.label} | ${r.wallMedianMs} ms | ${(r.peakRssBytes / 1e6).toFixed(0)} MB | ${clientOut} | ${serverOut} |`,
    );
  }
  console.log(`\nresults saved to ${resultsFile}`);
  console.log(
    "\nCAVEAT: next build also type-checks, lints, generates route types, and PRERENDERS every\n" +
      "static route to HTML+RSC; diffpack does none of those (it renders per-request). Read the\n" +
      "'measures' column as the scope of each number. This is an end-to-end 'time to a deployable\n" +
      "build', not a like-for-like bundler microbenchmark.",
  );
}

function collectMeta() {
  let cpu;
  try {
    cpu = execFileSync("sysctl", ["-n", "machdep.cpu.brand_string"], { encoding: "utf8" }).trim();
  } catch {
    cpu = os.cpus()?.[0]?.model?.trim();
  }
  let nextVersion = "unknown";
  try {
    nextVersion = JSON.parse(readFileSync(join(fixture, "node_modules", "next", "package.json"), "utf8")).version;
  } catch {}
  let commit = "unknown";
  try {
    commit = execFileSync("git", ["rev-parse", "--short", "HEAD"], { cwd: repoRoot, encoding: "utf8" }).trim();
  } catch {}
  return { date: new Date().toISOString(), cpu, node: process.version, diffpackCommit: commit, next: nextVersion };
}

function parseArgs(argv) {
  const p = {};
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === "--runs") p.runs = Number(next());
    else if (a === "--cases") p.cases = next().split(",");
    else if (a === "--fixture") p.fixture = next();
    else throw new Error(`unknown argument: ${a}`);
  }
  return p;
}
