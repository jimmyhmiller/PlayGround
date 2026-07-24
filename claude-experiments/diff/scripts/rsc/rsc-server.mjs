// The RSC app's HTTP server — the emitted Node runtime the fixture is served by.
//
// diffpack builds every graph natively (Rust); this orchestrator is plain Node
// (the app server, run as the oracle's process) that wires diffpack's three emitted
// bundles + manifests into a working RSC app. It runs NO React itself: it only
// moves flight BYTES between the graphs, each of which carries its own inlined
// React. Isolation between the react-server React and the SSR/browser React is by
// process: the react-server render/action bundle runs in a spawned child.
//
//   GET /            -> spawn the react-server render child -> flight of <Page/>;
//                       feed the flight to the SSR bundle (client refs resolved to
//                       the SSR graph's real modules via the divergent-id
//                       serverConsumerManifest) -> HTML; return an HTML document
//                       that inlines the flight and boots the browser bundle.
//   POST /_action/   -> spawn the react-server action child (decode -> dispatch ->
//                       render result to flight) -> return the flight (text/x-component).
//   GET /<asset>     -> serve the client build's public/ assets.
//
// The serverConsumerManifest (Manifest #2) is built by JOINING the client build's
// client-references manifest (the ids the flight carries) with the SSR build's own
// references manifest (the ids the SSR graph resolves through) on the shared
// canonical module id — the divergent-id ssrModuleMapping.

import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { readFileSync, existsSync, readdirSync } from "node:fs";
import { join, extname } from "node:path";
import { pathToFileURL } from "node:url";

function fail(message) {
  console.error(`rsc-server: ${message}`);
  process.exit(1);
}

const outputDir = process.argv[2];
const port = Number(process.argv[3] || "0");
if (!outputDir) fail("usage: node rsc-server.mjs <.diffpack-output dir> [port]");

const publicDir = join(outputDir, "public");
const rscRenderEntry = join(outputDir, "rsc-render", "server.mjs");
const ssrEntry = join(outputDir, "server", "server.mjs");
const clientManifestPath = join(outputDir, "client-references-manifest.json");
const ssrManifestPath = join(outputDir, "server-references-manifest.json");

for (const [label, p] of [
  ["client public/", publicDir],
  ["react-server render bundle", rscRenderEntry],
  ["SSR bundle", ssrEntry],
  ["client-references manifest", clientManifestPath],
  ["ssr-references manifest", ssrManifestPath],
]) {
  if (!existsSync(p)) fail(`${label} not found at ${p} — build all three graphs first`);
}

// --- Manifest #2: the divergent-id ssrModuleMapping ------------------------------
// moduleMap is keyed by the id the FLIGHT carries (client id) and resolves to the
// id the SSR graph requires (this build's id) for the same canonical module.
const clientRefs = JSON.parse(readFileSync(clientManifestPath, "utf8"));
const ssrRefs = JSON.parse(readFileSync(ssrManifestPath, "utf8"));
const moduleMap = {};
for (const [moduleId, clientEntry] of Object.entries(clientRefs)) {
  const ssrEntryRef = ssrRefs[moduleId];
  if (!ssrEntryRef) {
    fail(
      `no SSR reference for ${moduleId}; the SSR graph did not bundle this "use client" module`,
    );
  }
  moduleMap[String(clientEntry.id)] = {
    "*": { id: ssrEntryRef.id, chunks: ssrEntryRef.chunks, name: "*" },
  };
}
const serverConsumerManifest = {
  moduleMap,
  serverModuleMap: null,
  moduleLoading: { prefix: "", crossOrigin: null },
};

// --- The SSR bundle (in-process; its own inlined React) --------------------------
const ssrModule = await import(pathToFileURL(ssrEntry).href);
const renderFlightToHTML =
  ssrModule.renderFlightToHTML || (ssrModule.default && ssrModule.default.renderFlightToHTML);
if (typeof renderFlightToHTML !== "function") {
  fail("the SSR bundle does not export renderFlightToHTML");
}

// --- Spawn the react-server child for a flight (render or action) ----------------
function runReactServer(args, stdinBody) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [rscRenderEntry, ...args], {
      stdio: ["pipe", "pipe", "pipe"],
    });
    const out = [];
    const err = [];
    child.stdout.on("data", (chunk) => out.push(Buffer.from(chunk)));
    child.stderr.on("data", (chunk) => err.push(Buffer.from(chunk)));
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`react-server child (${args.join(" ")}) exited ${code}:\n${Buffer.concat(err)}`));
        return;
      }
      resolve(Buffer.concat(out));
    });
    if (stdinBody != null) child.stdin.write(stdinBody);
    child.stdin.end();
  });
}

function htmlDocument(ssrHtml, flightBase64) {
  return `<!doctype html>
<html>
<head><meta charset="utf-8"><title>diffpack RSC</title></head>
<body>
<div id="root">${ssrHtml}</div>
<script>window.__DIFFPACK_FLIGHT__ = ${JSON.stringify(flightBase64)};</script>
<script type="module" src="/client.js"></script>
</body>
</html>`;
}

const MIME = {
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".map": "application/json",
};

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://localhost");
    if (req.method === "GET" && url.pathname === "/") {
      const flight = await runReactServer(["render", clientManifestPath]);
      const html = await renderFlightToHTML(new Uint8Array(flight), serverConsumerManifest);
      const doc = htmlDocument(html, flight.toString("base64"));
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
      res.end(doc);
      return;
    }
    if (req.method === "POST" && url.pathname === "/_action/") {
      const id = req.headers["x-diffpack-action-id"];
      if (!id) {
        res.writeHead(400).end("missing x-diffpack-action-id");
        return;
      }
      const body = [];
      for await (const chunk of req) body.push(Buffer.from(chunk));
      const flight = await runReactServer(["action", id, clientManifestPath], Buffer.concat(body));
      res.writeHead(200, { "content-type": "text/x-component" });
      res.end(flight);
      return;
    }
    if (req.method === "GET") {
      // Serve a client asset out of public/.
      const name = url.pathname.replace(/^\//, "");
      const filePath = join(publicDir, name);
      if (name && existsSync(filePath) && filePath.startsWith(publicDir)) {
        res.writeHead(200, { "content-type": MIME[extname(filePath)] || "application/octet-stream" });
        res.end(readFileSync(filePath));
        return;
      }
    }
    res.writeHead(404).end("not found");
  } catch (error) {
    console.error(error && error.stack ? error.stack : String(error));
    res.writeHead(500, { "content-type": "text/plain" }).end(String(error && error.stack));
  }
});

server.listen(port, () => {
  const actual = server.address().port;
  // A single line the gate parses to learn the port, then the server stays up.
  console.log(`rsc-server listening on http://localhost:${actual}`);
  if (process.env.RSC_PUBLIC_ASSETS === "1") {
    console.log(`  public/: ${readdirSync(publicDir).join(", ")}`);
  }
});
