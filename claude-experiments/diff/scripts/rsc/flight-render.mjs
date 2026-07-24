// RSC flight oracle (node-only, no browser) — prove diffpack's flight runtime
// renders the Server Component <Page/> (which embeds a hook-bearing `"use client"`
// island and a `"use server"` action) to correct SSR HTML, using the REAL
// react-server-dom-webpack across the condition boundary and diffpack's own three
// graphs. This is the fast subset of the Slice E browser gate (scripts/rsc/rsc-check.sh):
// same render + SSR-of-flight path, asserted in node without a browser.
//
// It threads diffpack's real outputs (flight-check.sh builds all three graphs):
//   • the CLIENT build's client-references manifest (Manifest #1 = the ids the
//     flight carries) and the SSR build's own server-references manifest (the ids
//     the SSR graph resolves through) are JOINED on the shared canonical module id
//     into the divergent-id serverConsumerManifest (Manifest #2 / ssrModuleMapping);
//   • the REACT-SERVER render bundle (its own inlined react-server React, in a child
//     process) renders <Page/> to a flight stream with Manifest #1 as bundlerConfig;
//   • the SSR bundle (its own inlined React + the island as real hook-bearing code)
//     consumes that flight through Manifest #2 and renders it to HTML.
//
// Asserts the HTML carries BOTH the Server Component's own text and the client
// island's initial state (count: 5) — the client reference resolved to the real
// diffpack-bundled component under the SSR React. Fails loudly if the pinned RSC
// deps are absent; never skips.

import { spawnSync } from "node:child_process";
import { readFileSync, realpathSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = realpathSync(join(here, "..", ".."));
const fixture = process.argv[2] ? realpathSync(process.argv[2]) : join(repo, "integration", "rsc-reference");
const output = join(fixture, ".diffpack-output");

function fail(message) {
  console.error(`FAIL: ${message}`);
  process.exit(1);
}

const rscRenderEntry = join(output, "rsc-render", "server.mjs");
const ssrEntry = join(output, "server", "server.mjs");
const clientManifestPath = join(output, "client-references-manifest.json");
const ssrManifestPath = join(output, "server-references-manifest.json");
for (const [label, p] of [
  ["react-server render bundle", rscRenderEntry],
  ["SSR bundle", ssrEntry],
  ["client-references manifest", clientManifestPath],
  ["server-references manifest", ssrManifestPath],
]) {
  if (!existsSync(p)) fail(`${label} not found at ${p} — run scripts/rsc/flight-check.sh (builds all three graphs) first`);
}
if (!existsSync(join(fixture, "node_modules", "react-server-dom-webpack"))) {
  fail(`react-server-dom-webpack not installed in ${fixture}; run \`npm install\` in the fixture (never skipped)`);
}

// --- Manifest #2: the divergent-id ssrModuleMapping ------------------------------
const clientRefs = JSON.parse(readFileSync(clientManifestPath, "utf8"));
const ssrRefs = JSON.parse(readFileSync(ssrManifestPath, "utf8"));
const moduleMap = {};
for (const [moduleId, clientEntry] of Object.entries(clientRefs)) {
  const ssrEntryRef = ssrRefs[moduleId];
  if (!ssrEntryRef) fail(`no SSR reference for ${moduleId}; the SSR graph did not bundle this "use client" module`);
  moduleMap[String(clientEntry.id)] = { "*": { id: ssrEntryRef.id, chunks: ssrEntryRef.chunks, name: "*" } };
}
const serverConsumerManifest = { moduleMap, serverModuleMap: null, moduleLoading: { prefix: "", crossOrigin: null } };
console.log(`OK: divergent-id ssrModuleMapping ${JSON.stringify(moduleMap)} (flight/client id -> SSR id)`);

// --- The react-server render child produces the flight ---------------------------
const render = spawnSync(process.execPath, [rscRenderEntry, "render", clientManifestPath], {
  encoding: "buffer",
  input: "",
});
if (render.status !== 0) {
  fail(`react-server render child failed (exit ${render.status}):\n${render.stderr}`);
}
const flight = render.stdout;
if (!flight || flight.length === 0) fail("react-server render produced no flight");
const flightText = flight.toString("utf8");
if (!/I\[/.test(flightText)) fail(`flight carries no client-reference import row (I[...]):\n${flightText}`);
console.log(`OK: react-server render produced ${flight.length} bytes carrying a client reference`);

// --- The SSR bundle consumes the flight -> HTML ----------------------------------
const ssrModule = await import(pathToFileURL(ssrEntry).href);
const renderFlightToHTML = ssrModule.renderFlightToHTML || (ssrModule.default && ssrModule.default.renderFlightToHTML);
if (typeof renderFlightToHTML !== "function") fail("the SSR bundle does not export renderFlightToHTML");
const html = await renderFlightToHTML(new Uint8Array(flight), serverConsumerManifest);

if (!html.includes("Server:from-server")) fail(`SSR HTML missing the Server Component text:\n${html}`);
if (!html.includes("count: 5")) fail(`SSR HTML missing the client child's rendered output (count: 5):\n${html}`);
if (!/id="counter"/.test(html)) fail(`SSR HTML missing the client child element (id="counter"):\n${html}`);
console.log(`OK: SSR HTML = ${JSON.stringify(html)}`);
console.log("PASS: <Page/> (Server Component + hook-bearing use-client island) renders to correct SSR HTML via the flight runtime; the flight's client reference resolved to the real diffpack-bundled component through the divergent-id ssrModuleMapping");
