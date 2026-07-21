import { execFileSync, spawnSync } from "node:child_process";
import { readFileSync, readdirSync, statSync, rmSync } from "node:fs";
import { join } from "node:path";
import { gzipSync } from "node:zlib";

export function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

export function round(value, digits = 1) {
  return Number(value.toFixed(digits));
}

// Execute a bundle under node and return the printed integer. Throws unless
// stdout is exactly one safe integer.
export function executeNumericBundle(file) {
  const stdout = execFileSync("node", [file], { encoding: "utf8" }).trim();
  const value = Number(stdout);
  if (!Number.isSafeInteger(value)) {
    throw new Error(`bundle output of ${file} is not a safe integer: ${JSON.stringify(stdout)}`);
  }
  return value;
}

export function verifyBundleValue(file, expected, label) {
  const actual = executeNumericBundle(file);
  if (actual !== expected) {
    throw new Error(
      `RUNTIME DISAGREEMENT (${label}): ${file} printed ${actual}, expected ${expected}. Refusing to report timings for this pair.`,
    );
  }
  return actual;
}

// Wall time of a fresh process, in milliseconds.
export function timeProcess(command, args, options = {}) {
  const started = process.hrtime.bigint();
  const result = spawnSync(command, args, { encoding: "utf8", ...options });
  const elapsedMs = Number(process.hrtime.bigint() - started) / 1e6;
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(
      `${command} ${args.join(" ")} exited ${result.status}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`,
    );
  }
  return { elapsedMs, stdout: result.stdout, stderr: result.stderr };
}

// Peak RSS (bytes) of a standalone process via /usr/bin/time -v.
export function peakRss(command, args, options = {}) {
  const result = spawnSync("/usr/bin/time", ["-v", command, ...args], {
    encoding: "utf8",
    ...options,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(
      `/usr/bin/time -v ${command} exited ${result.status}\nstderr:\n${result.stderr}`,
    );
  }
  const match = result.stderr.match(/Maximum resident set size \(kbytes\): (\d+)/);
  if (!match) throw new Error(`could not parse /usr/bin/time -v output:\n${result.stderr}`);
  return Number(match[1]) * 1024;
}

// Total and gzipped bytes of every file in a directory tree (or single file).
export function outputBytes(path) {
  const files = [];
  const stack = [path];
  while (stack.length > 0) {
    const current = stack.pop();
    const info = statSync(current);
    if (info.isDirectory()) {
      for (const entry of readdirSync(current)) stack.push(join(current, entry));
    } else {
      files.push(current);
    }
  }
  let raw = 0;
  let gzip = 0;
  for (const file of files) {
    const content = readFileSync(file);
    raw += content.length;
    gzip += gzipSync(content, { level: 6 }).length;
  }
  return { raw, gzip, files: files.length };
}

export function removePaths(paths) {
  for (const path of paths) rmSync(path, { recursive: true, force: true });
}
