import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { rolldown } from "rolldown";

const oracleRoot = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = dirname(oracleRoot);
const strict = process.argv.includes("--strict");
const workspace = mkdtempSync(join(tmpdir(), "diffpack-production-parity-"));

execFileSync("cargo", ["build", "--quiet"], { cwd: repositoryRoot, stdio: "inherit" });
const diffpack = join(repositoryRoot, "target", "debug", "diffpack");

const results = [];
try {
  await checkTreeShaking();
  await checkPackageSideEffects();
  await checkSourceMaps();
  await checkCodeSplitting();
} finally {
  rmSync(workspace, { recursive: true, force: true });
}

for (const result of results) {
  console.log(`${result.ok ? "PASS" : "GAP "} ${result.name}: ${result.detail}`);
}
const gaps = results.filter((result) => !result.ok).length;
console.log(`\n${results.length - gaps}/${results.length} production parity gates passed`);
if (strict && gaps > 0) process.exitCode = 1;

async function checkTreeShaking() {
  const fixture = resolve(oracleRoot, "fixtures/tree-shaking/entry.js");
  const { diffpackCode, rolldownCode } = await buildPair("tree-shaking", fixture, {
    treeshake: true,
  });
  const sentinel = "UNUSED_EXPORT_SENTINEL_73B4";
  results.push({
    name: "symbol-inclusion",
    ok: !diffpackCode.includes(sentinel) && !rolldownCode.includes(sentinel),
    detail: diffpackCode.includes(sentinel)
      ? "Diffpack retains an unused export that Rolldown removes"
      : "unused export removed",
  });
}

async function checkPackageSideEffects() {
  const fixture = resolve(oracleRoot, "fixtures/package-side-effects/entry.js");
  const pair = await buildPair("package-side-effects", fixture, { treeshake: true });
  const diffpackRun = execute(pair.diffpackOutput);
  const rolldownRun = execute(pair.rolldownOutput);
  results.push({
    name: "package-sideEffects",
    ok: diffpackRun === rolldownRun && diffpackRun === "entry\n",
    detail: diffpackRun === rolldownRun
      ? "sideEffects metadata respected"
      : `Diffpack=${JSON.stringify(diffpackRun)} Rolldown=${JSON.stringify(rolldownRun)}`,
  });
}

async function checkSourceMaps() {
  const fixture = resolve(oracleRoot, "fixtures/tree-shaking/entry.js");
  const pair = await buildPair("source-maps", fixture, { treeshake: true, sourcemap: true });
  const diffpackMap = `${pair.diffpackOutput}.map`;
  const rolldownMap = `${pair.rolldownOutput}.map`;
  const map = existsSync(diffpackMap)
    ? JSON.parse(readFileSync(diffpackMap, "utf8"))
    : null;
  const valid = map?.version === 3
    && map.sources?.some((source) => source.endsWith("values.js"))
    && typeof map.mappings === "string"
    && map.mappings.length > 0
    && pair.diffpackCode.includes(`sourceMappingURL=${diffpackMap.split("/").at(-1)}`);
  results.push({
    name: "source-maps",
    ok: valid && existsSync(rolldownMap),
    detail: valid ? "mapped sources and source contents emitted" : "Diffpack source map missing or invalid",
  });
}

async function checkCodeSplitting() {
  const fixture = resolve(oracleRoot, "fixtures/code-splitting/entry.js");
  const caseRoot = join(workspace, "code-splitting");
  const diffpackOutput = join(caseRoot, "diffpack.cjs");
  const diffpackBuild = spawnSync(diffpack, ["bundle", fixture, diffpackOutput], { encoding: "utf8" });
  if (diffpackBuild.status !== 0) throw new Error(diffpackBuild.stderr);
  const rolldownDirectory = join(caseRoot, "rolldown");
  const bundle = await rolldown({ input: fixture, treeshake: true });
  await bundle.write({ dir: rolldownDirectory, format: "cjs", codeSplitting: true });
  await bundle.close();
  const diffpackChunks = readdirSync(caseRoot).filter((file) => file.endsWith(".cjs")).length;
  const rolldownChunks = readdirSync(rolldownDirectory).filter((file) => file.endsWith(".js")).length;
  const diffpackRun = execute(diffpackOutput);
  const rolldownEntry = join(
    rolldownDirectory,
    readdirSync(rolldownDirectory).find((file) => file.startsWith("entry") && file.endsWith(".js")),
  );
  const rolldownRun = execute(rolldownEntry);
  results.push({
    name: "code-splitting",
    ok: diffpackChunks > 1
      && rolldownChunks > 1
      && diffpackRun === "alpha beta\n"
      && diffpackRun === rolldownRun,
    detail: `Diffpack chunks=${diffpackChunks}, Rolldown chunks=${rolldownChunks}, output=${JSON.stringify(diffpackRun)}`,
  });
}

async function buildPair(name, entry, options) {
  const caseRoot = join(workspace, name);
  const diffpackOutput = join(caseRoot, "diffpack.cjs");
  const rolldownOutput = join(caseRoot, "rolldown.cjs");
  const diffpackArguments = ["bundle", entry, diffpackOutput];
  if (options.sourcemap) diffpackArguments.push("--sourcemap");
  const diffpackBuild = spawnSync(diffpack, diffpackArguments, {
    encoding: "utf8",
  });
  if (diffpackBuild.status !== 0) throw new Error(diffpackBuild.stderr);
  const bundle = await rolldown({ input: entry, treeshake: options.treeshake });
  await bundle.write({
    file: rolldownOutput,
    format: "cjs",
    codeSplitting: false,
    sourcemap: options.sourcemap ?? false,
  });
  await bundle.close();
  return {
    diffpackOutput,
    rolldownOutput,
    diffpackCode: readFileSync(diffpackOutput, "utf8"),
    rolldownCode: readFileSync(rolldownOutput, "utf8"),
  };
}

function execute(path) {
  const result = spawnSync(process.execPath, [path], { encoding: "utf8" });
  return result.stdout.replaceAll("\r\n", "\n");
}
