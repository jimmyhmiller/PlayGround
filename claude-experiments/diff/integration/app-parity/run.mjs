// Runs the full app-parity suite: every app config in apps/, prints a per-app
// per-step PASS/DIFF table, writes report.json, exits nonzero on any DIFF or
// ERROR (invariant failures included).
//   node integration/app-parity/run.mjs [appName ...]
import { mkdirSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { runApp } from "./harness.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ARTIFACTS = join(HERE, "artifacts");
mkdirSync(ARTIFACTS, { recursive: true });

const APP_MODULES = [
  "redux-essentials.mjs",
  "markpad.mjs",
  "chebyshev-calculator.mjs",
  "the-last-pawn.mjs",
  "wall-go.mjs",
];

const only = process.argv.slice(2);
const suiteStart = Date.now();
const report = { startedAt: new Date().toISOString(), apps: [] };

for (const mod of APP_MODULES) {
  const config = (await import(join(HERE, "apps", mod))).default;
  if (only.length && !only.includes(config.name)) continue;
  process.stdout.write(`\n=== ${config.name} ===\n`);
  const t0 = Date.now();
  let result;
  try {
    result = await runApp(config, { artifactsDir: ARTIFACTS });
  } catch (e) {
    result = {
      app: config.name,
      steps: [],
      fatal: String(e && e.stack ? e.stack : e),
    };
  }
  result.ms = Date.now() - t0;
  report.apps.push(result);
  printApp(result);
}

report.totalMs = Date.now() - suiteStart;
writeFileSync(join(HERE, "report.json"), JSON.stringify(report, null, 2));

// ---- summary table ----
console.log("\n================ SUMMARY ================");
const rows = [["app", "step", "mode", "status", "detail"]];
for (const app of report.apps) {
  if (app.fatal) rows.push([app.app, "-", "-", "ERROR", firstLine(app.fatal)]);
  for (const s of app.steps) {
    rows.push([app.app, s.name, s.mode, s.status, stepDetail(s)]);
  }
}
printTable(rows);
console.log(
  `\ntotal runtime: ${(report.totalMs / 1000).toFixed(1)}s  |  report: ${join(HERE, "report.json")}  |  artifacts: ${ARTIFACTS}`
);

const bad = report.apps.some(
  (a) => a.fatal || a.steps.some((s) => s.status !== "PASS")
);
process.exit(bad ? 1 : 0);

// ---- helpers ----
function stepDetail(s) {
  if (s.status === "ERROR") return firstLine(s.error);
  if (s.mode === "invariant") {
    const failed = (s.checks || []).filter((c) => !c.pass);
    return failed.length
      ? `failed: ${failed.map((c) => c.name).join("; ")}`
      : `${(s.checks || []).length} invariants hold`;
  }
  const diffs = Object.entries(s.channels || {})
    .filter(([, c]) => c.status !== "PASS")
    .map(([name, c]) => name + (c.detail ? `(${c.detail})` : ""));
  return diffs.length ? `DIFF in: ${diffs.join(", ")}` : "all channels match";
}

function firstLine(s) {
  return String(s).split("\n")[0].slice(0, 100);
}

function printApp(app) {
  if (app.fatal) {
    console.log(`  FATAL: ${app.fatal}`);
    return;
  }
  for (const s of app.steps) {
    console.log(
      `  ${s.status.padEnd(5)} ${s.name} [${s.mode}] (${s.ms}ms) ${stepDetail(s)}`
    );
  }
}

function printTable(rows) {
  const widths = rows[0].map((_, i) =>
    Math.min(60, Math.max(...rows.map((r) => String(r[i]).length)))
  );
  for (const [ri, r] of rows.entries()) {
    console.log(
      r
        .map((c, i) => String(c).slice(0, widths[i]).padEnd(widths[i]))
        .join("  ")
    );
    if (ri === 0) console.log(widths.map((w) => "-".repeat(w)).join("  "));
  }
}
