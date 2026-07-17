import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { rolldown } from "rolldown";

const oracleRoot = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = dirname(oracleRoot);
const cases = JSON.parse(readFileSync(join(oracleRoot, "cases.json"), "utf8"));
const filters = process.argv.slice(2);

if (filters.includes("--list")) {
  for (const testCase of cases) {
    console.log(`${testCase.name}\t${testCase.tags.join(",")}`);
  }
  process.exit(0);
}

const requested = filters.filter((argument) => !argument.startsWith("--"));
const selected = requested.length === 0
  ? cases
  : cases.filter((testCase) => requested.some(
      (filter) => testCase.name.includes(filter) || testCase.tags.includes(filter),
    ));

if (selected.length === 0) {
  throw new Error(`no oracle cases matched: ${requested.join(", ")}`);
}

execFileSync("cargo", ["build", "--quiet"], {
  cwd: repositoryRoot,
  stdio: "inherit",
});

const diffpack = join(repositoryRoot, "target", "debug", "diffpack");
const workspace = mkdtempSync(join(tmpdir(), "diffpack-oracle-"));
let failures = 0;

try {
  for (const testCase of selected) {
    const entry = resolve(oracleRoot, testCase.entry);
    const caseOutput = join(workspace, testCase.name);
    const diffpackOutput = join(caseOutput, "diffpack.cjs");
    const rolldownOutput = join(caseOutput, "rolldown.cjs");

    const diffpackBuild = spawnSync(diffpack, ["bundle", entry, diffpackOutput], {
      cwd: repositoryRoot,
      encoding: "utf8",
      timeout: 30_000,
    });
    const referenceBuild = await buildWithRolldown(entry, rolldownOutput);
    const diffpackRun = diffpackBuild.status === 0
      ? executeBundle(diffpackOutput)
      : processResult(diffpackBuild);
    const referenceRun = referenceBuild.ok
      ? executeBundle(rolldownOutput)
      : referenceBuild.result;

    const expected = testCase.failure
      ? { failure: true }
      : { status: 0, stdout: testCase.stdout, stderr: "" };
    const diffpackCorrect = matchesExpected(diffpackRun, expected);
    const referenceCorrect = matchesExpected(referenceRun, expected);
    const implementationsAgree = testCase.failure
      ? diffpackRun.status !== 0 && referenceRun.status !== 0
      : sameResult(diffpackRun, referenceRun);
    const passed = diffpackCorrect && referenceCorrect && implementationsAgree;

    console.log(`${passed ? "PASS" : "FAIL"} ${testCase.name}`);
    if (!passed) {
      failures += 1;
      printMismatch("expected", expected);
      printMismatch("diffpack", diffpackRun);
      printMismatch("rolldown", referenceRun);
      if (diffpackBuild.status !== 0) {
        console.log(indent("diffpack build", processResult(diffpackBuild)));
      }
    }
  }
} finally {
  rmSync(workspace, { recursive: true, force: true });
}

console.log(`\n${selected.length - failures}/${selected.length} behavioral cases passed`);
process.exitCode = failures === 0 ? 0 : 1;

async function buildWithRolldown(entry, output) {
  try {
    const bundle = await rolldown({ input: entry, treeshake: false });
    await bundle.write({
      file: output,
      format: "cjs",
      codeSplitting: false,
    });
    await bundle.close();
    return { ok: true };
  } catch (error) {
    return {
      ok: false,
      result: { status: 1, stdout: "", stderr: String(error) },
    };
  }
}

function executeBundle(path) {
  return processResult(spawnSync(process.execPath, [path], {
    encoding: "utf8",
    timeout: 10_000,
  }));
}

function processResult(result) {
  return {
    status: result.status ?? 1,
    stdout: normalize(result.stdout),
    stderr: normalize(result.stderr || result.error?.message || ""),
  };
}

function normalize(value) {
  return String(value ?? "").replaceAll("\r\n", "\n");
}

function sameResult(left, right) {
  return left.status === right.status
    && left.stdout === right.stdout
    && left.stderr === right.stderr;
}

function matchesExpected(result, expected) {
  return expected.failure ? result.status !== 0 : sameResult(result, expected);
}

function printMismatch(label, result) {
  console.log(indent(label, result));
}

function indent(label, result) {
  if (result.failure) {
    return `  ${label}: non-zero build or runtime status`;
  }
  return `  ${label}: status=${result.status} stdout=${JSON.stringify(result.stdout)} stderr=${JSON.stringify(result.stderr)}`;
}
