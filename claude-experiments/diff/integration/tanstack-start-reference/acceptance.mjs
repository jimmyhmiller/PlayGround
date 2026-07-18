import { spawn } from "node:child_process";
import { readFile, readdir } from "node:fs/promises";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const fixtureRoot = dirname(fileURLToPath(import.meta.url));
const contract = JSON.parse(
  await readFile(join(fixtureRoot, "acceptance-contract.json"), "utf8"),
);
const mode = process.argv[2] ?? "diffpack";
const strict = process.argv.includes("--strict");
if (!new Set(["reference", "diffpack"]).has(mode)) {
  throw new Error("usage: node acceptance.mjs <reference|diffpack> [--strict]");
}

const outputRoot = join(
  fixtureRoot,
  mode === "reference" ? contract.referenceOutput : contract.diffpackOutput,
);
const checks = [];
const record = (name, ok, detail) => checks.push({ name, ok, detail });
const files = await listFiles(outputRoot).catch(() => []);
const publicFiles = files.filter((file) => file.startsWith("public/"));
const serverFiles = files.filter((file) => file.startsWith("server/"));

record("output directory", files.length > 0, `${files.length} files under ${relative(fixtureRoot, outputRoot)}`);
record(
  "browser chunks",
  publicFiles.filter((file) => file.endsWith(".js")).length >= contract.minimumPublicJavaScriptFiles,
  `${publicFiles.filter((file) => file.endsWith(".js")).length} JavaScript files`,
);
record(
  "extracted CSS",
  publicFiles.filter((file) => file.endsWith(".css")).length >= contract.minimumPublicCssFiles,
  `${publicFiles.filter((file) => file.endsWith(".css")).length} CSS files`,
);
record(
  "server graph",
  serverFiles.filter((file) => file.endsWith(".mjs")).length >= contract.minimumServerModuleFiles,
  `${serverFiles.filter((file) => file.endsWith(".mjs")).length} server modules`,
);
for (const file of contract.requiredPublicFiles) {
  record(`public/${file}`, publicFiles.includes(`public/${file}`), "required static asset");
}
for (const file of contract.requiredServerFiles) {
  record(`server/${file}`, serverFiles.includes(`server/${file}`), "required server entry");
}
for (const fragment of contract.requiredServerFilenameFragments) {
  record(
    `server artifact: ${fragment}`,
    serverFiles.some((file) => file.includes(fragment)),
    "required environment/manifest artifact",
  );
}

const serverEntry = join(outputRoot, "server/index.mjs");
if (serverFiles.includes("server/index.mjs")) {
  const port = 43127;
  const server = spawn(process.execPath, [serverEntry], {
    cwd: fixtureRoot,
    env: { ...process.env, PORT: String(port), HOST: "127.0.0.1" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let serverLog = "";
  server.stdout.on("data", (chunk) => { serverLog += chunk; });
  server.stderr.on("data", (chunk) => { serverLog += chunk; });
  try {
    await waitForServer(`http://127.0.0.1:${port}/`, server, () => serverLog);
    for (const expectation of contract.http) {
      const response = await fetch(`http://127.0.0.1:${port}${expectation.path}`);
      const body = await response.text();
      const contentType = response.headers.get("content-type") ?? "";
      record(
        `HTTP ${expectation.path}`,
        response.status === expectation.status
          && contentType.includes(expectation.contentType)
          && expectation.includes.every((value) => body.includes(value)),
        `status=${response.status} content-type=${contentType}`,
      );
    }
  } catch (error) {
    record("production server", false, error.message);
  } finally {
    server.kill("SIGTERM");
    await new Promise((done) => server.once("exit", done));
  }
} else {
  for (const expectation of contract.http) {
    record(`HTTP ${expectation.path}`, false, "server entry is missing");
  }
}

for (const check of checks) {
  console.log(`${check.ok ? "PASS" : "MISS"} ${check.name}: ${check.detail}`);
}
const passed = checks.filter((check) => check.ok).length;
console.log(`\n${mode}: ${passed}/${checks.length} TanStack production gates passed`);
if (strict && passed !== checks.length) process.exitCode = 1;

async function listFiles(root) {
  const files = [];
  async function visit(directory) {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) await visit(path);
      else files.push(relative(root, path).replaceAll("\\", "/"));
    }
  }
  await visit(root);
  return files.sort();
}

async function waitForServer(url, server, getLog) {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    if (server.exitCode !== null) {
      throw new Error(`server exited with ${server.exitCode}: ${getLog().trim()}`);
    }
    try {
      await fetch(url);
      return;
    } catch {
      await new Promise((done) => setTimeout(done, 50));
    }
  }
  throw new Error(`server did not listen in time: ${getLog().trim()}`);
}
