// RSC Slice B / R3 oracle — resolve a client reference through diffpack's
// `__webpack_*` seam over its module registry.
//
// Given a `"use client"` module's client reference `$$id`
// ("<moduleId>#<export>"), diffpack's emitted client bundle, and the
// client-references manifest (bundlerConfig) it emitted, this replicates exactly
// what `react-server-dom-webpack` does in the browser:
//
//   1. Loading the client bundle installs diffpack's runtime AND the webpack seam
//      globals (`__webpack_require__`, `__webpack_require__.u`,
//      `__webpack_chunk_load__`) on `globalThis`.
//   2. The SERVER render side splits `$$id` at the last `#` and looks the moduleId
//      up in the manifest to produce the flight metadata `{ id, chunks, name }`
//      (react's `resolveClientReferenceMetadata`).
//   3. The BROWSER side chunk-loads each `chunks` pair via `__webpack_chunk_load__`,
//      then `__webpack_require__(id)` and reads the export off the module.
//
// The oracle asserts this resolves to the REAL exported component (the actual
// function, invoked). It fails loudly (never skips) on any missing piece.

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { join, dirname } from "node:path";

function fail(message) {
  console.error(`FAIL: ${message}`);
  process.exit(1);
}

const outputDir = process.argv[2];
if (!outputDir) fail("usage: node seam-resolve.mjs <.diffpack-output dir>");

const publicDir = join(outputDir, "public");
const clientEntry = join(publicDir, "client.js");
const manifestPath = join(outputDir, "client-references-manifest.json");

for (const [label, p] of [["client bundle", clientEntry], ["manifest", manifestPath]]) {
  if (!existsSync(p)) fail(`${label} not found at ${p} — run the client build first`);
}

// Node treats a sibling `.js` as CommonJS unless the directory declares ESM. The
// emitted client bundle is browser ESM (`export default`, dynamic `import()`), so
// mark the emitted `public/` as an ES module package for the oracle to import it.
const pkg = join(publicDir, "package.json");
if (!existsSync(pkg)) writeFileSync(pkg, JSON.stringify({ type: "module" }));

// ---- 1. Load the client bundle: installs the runtime + the webpack seam. ----
await import(pathToFileURL(clientEntry).href);

for (const g of ["__webpack_require__", "__webpack_chunk_load__"]) {
  if (typeof globalThis[g] !== "function") {
    fail(`seam global ${g} was not installed by the client bundle`);
  }
}
if (typeof globalThis.__webpack_require__.u !== "function") {
  fail("__webpack_require__.u (chunk filename resolver) was not installed");
}

// ---- 2. Read the manifest and pick the `"use client"` module. ----
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
const moduleIds = Object.keys(manifest);
const moduleId = moduleIds.find((id) => id.endsWith("Counter.js"));
if (!moduleId) fail(`no Counter client reference in manifest; keys: ${moduleIds}`);

const exportName = "Counter";
const $$id = `${moduleId}#${exportName}`;

// ---- 3. SERVER render side: resolveClientReferenceMetadata(bundlerConfig, {$$id}). ----
// Split at the LAST `#`, key = prefix, export = suffix (react's algorithm).
const hash = $$id.lastIndexOf("#");
const key = $$id.slice(0, hash);
const wantExport = $$id.slice(hash + 1);
if (key !== moduleId) fail(`$$id base ${key} !== manifest key ${moduleId}`);

const entry = manifest[key];
if (!entry) fail(`manifest has no entry for ${key}`);
if (typeof entry.id !== "number") fail(`manifest entry.id is not a numeric runtime id: ${entry.id}`);
if (!Array.isArray(entry.chunks)) fail(`manifest entry.chunks is not an array`);
if (entry.chunks.length % 2 !== 0) fail(`manifest chunks must be a flat even-length [id,file,...]`);

// ---- 4. BROWSER side: chunk-load prerequisites, then require the module. ----
for (let i = 0; i < entry.chunks.length; i += 2) {
  const chunkId = entry.chunks[i];
  // Faithful to react's client: __webpack_chunk_load__(chunkId) before require.
  await globalThis.__webpack_chunk_load__(chunkId);
}

const mod = globalThis.__webpack_require__(entry.id);
if (!mod) fail(`__webpack_require__(${entry.id}) returned nothing`);
const resolved = wantExport === "default" ? mod.default : mod[wantExport];

if (typeof resolved !== "function") {
  fail(`resolved reference is not the real component function (got ${typeof resolved})`);
}
const rendered = resolved();
if (rendered !== "REAL-COUNTER") {
  fail(`resolved component produced ${JSON.stringify(rendered)}, expected "REAL-COUNTER"`);
}

// ---- 5. The seam hard-errors (never silently falls back) on an unknown chunk. ----
let threw = false;
try {
  globalThis.__webpack_chunk_load__("no-such-chunk-id");
} catch (e) {
  threw = /unknown chunk id/.test(String(e && e.message));
}
if (!threw) fail("__webpack_chunk_load__ did not hard-error on an unknown chunk id");

console.log(`PASS: client reference ${$$id}`);
console.log(`  -> resolveClientReferenceMetadata => { id: ${entry.id}, chunks: ${JSON.stringify(entry.chunks)}, name: ${JSON.stringify(entry.name)} }`);
console.log(`  -> __webpack_require__(${entry.id})[${JSON.stringify(wantExport)}]() === ${JSON.stringify(rendered)}`);
