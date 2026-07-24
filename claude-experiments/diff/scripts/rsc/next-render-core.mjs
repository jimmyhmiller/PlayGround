// The shared render seam for a diffpack-built Next app-router app: manifest loading,
// the SSR-of-flight bundle import, and the react-server render child spawn. Extracted
// verbatim from scripts/rsc/next-server.mjs so the live orchestrator and the SSG
// prerenderer render IDENTICALLY (same divergent-id serverConsumerManifest, same
// onAllReady full-document SSR, same fd-3 meta channel) — build-time prerender is the
// per-request pipeline run ahead of time and written to disk.
//
// This module runs NO React itself: it only moves flight BYTES between the (bundled,
// self-contained) SSR and react-server graphs, each carrying its own inlined React.

import { spawn } from "node:child_process";
import { readFileSync, existsSync, statSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

/// Read both client-references manifests and build the divergent-id ssrModuleMapping
/// (Manifest #2) the SSR bundle resolves the flight's client references through.
export function loadManifests(outputDir) {
  const clientManifestPath = join(outputDir, "client-references-manifest.json");
  const ssrManifestPath = join(outputDir, "server-references-manifest.json");
  const clientRefs = JSON.parse(readFileSync(clientManifestPath, "utf8"));
  const ssrRefs = JSON.parse(readFileSync(ssrManifestPath, "utf8"));
  const moduleMap = {};
  for (const [moduleId, clientEntry] of Object.entries(clientRefs)) {
    const ssrEntryRef = ssrRefs[moduleId];
    if (!ssrEntryRef) {
      throw new Error(
        `next-render-core: no SSR reference for ${moduleId}; the SSR graph did not bundle this "use client" module`,
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
  return { clientRefs, ssrRefs, serverConsumerManifest, clientManifestPath };
}

function pickRender(mod) {
  const fn = mod.renderFlightToDocument || (mod.default && mod.default.renderFlightToDocument);
  if (typeof fn !== "function") {
    throw new Error("next-render-core: the SSR bundle does not export renderFlightToDocument");
  }
  return fn;
}

/// Memoized dynamic import of the SSR-of-flight bundle → its `renderFlightToDocument`.
/// In dev the bundle is re-imported (fresh `?v=<mtime>`) when it changes on disk; in
/// prod it is imported once and cached.
export function getRenderFlightToDocument(ssrEntry, options) {
  const dev = !!(options && options.dev);
  let cache = { key: null, fn: null };
  return async function render(...args) {
    if (!dev) {
      if (!cache.fn) cache.fn = pickRender(await import(pathToFileURL(ssrEntry).href));
      return cache.fn(...args);
    }
    const key = statSync(ssrEntry).mtimeMs;
    if (!cache.fn || cache.key !== key) {
      cache = { key, fn: pickRender(await import(pathToFileURL(ssrEntry).href + "?v=" + key)) };
    }
    return cache.fn(...args);
  };
}

/// A `runReactServer(args, stdinBody)` closure over the react-server render child
/// entry. Spawns `node <rscRenderEntry> <args...>`, collects the flight on stdout and
/// the `{status,params,redirect,notFound}` meta on fd 3, and rejects on a nonzero
/// exit. Identical to the orchestrator's spawn (a 404 renders its flight AND exits 0,
/// carrying its status on fd 3 — so a nonzero exit is a genuine failure).
export function makeRunReactServer(rscRenderEntry) {
  return function runReactServer(args, stdinBody) {
    return new Promise((resolve, reject) => {
      const child = spawn(process.execPath, [rscRenderEntry, ...args], {
        stdio: ["pipe", "pipe", "pipe", "pipe"],
      });
      const out = [];
      const err = [];
      const meta = [];
      child.stdout.on("data", (chunk) => out.push(Buffer.from(chunk)));
      child.stderr.on("data", (chunk) => err.push(Buffer.from(chunk)));
      if (child.stdio[3]) child.stdio[3].on("data", (chunk) => meta.push(Buffer.from(chunk)));
      child.on("error", reject);
      child.on("close", (code) => {
        if (code !== 0) {
          reject(new Error(`react-server child (${args.join(" ")}) exited ${code}:\n${Buffer.concat(err)}`));
          return;
        }
        let parsed = { status: 200, params: {} };
        const raw = Buffer.concat(meta).toString("utf8").trim();
        if (raw) {
          try {
            parsed = JSON.parse(raw);
          } catch {
            // a malformed sidechannel is non-fatal; keep the defaults.
          }
        }
        resolve({
          flight: Buffer.concat(out),
          status: parsed.status || 200,
          params: parsed.params || {},
          redirect: parsed.redirect,
          notFound: parsed.notFound,
          stderr: Buffer.concat(err).toString("utf8"),
        });
      });
      if (stdinBody != null) child.stdin.write(stdinBody);
      child.stdin.end();
    });
  };
}

/// Assert the four bundles/manifests the render seam needs exist, or throw naming
/// exactly which is missing + how to produce it (mirrors the orchestrator's checks).
export function requireBuiltBundles(outputDir) {
  const need = [
    ["react-server render bundle", join(outputDir, "rsc-render", "server.mjs")],
    ["SSR bundle", join(outputDir, "server", "server.mjs")],
    ["client-references manifest", join(outputDir, "client-references-manifest.json")],
    ["ssr-references manifest", join(outputDir, "server-references-manifest.json")],
  ];
  for (const [label, p] of need) {
    if (!existsSync(p)) {
      throw new Error(
        `next-render-core: ${label} not found at ${p} — run the client -> react-server ` +
          `(cp -> rsc-render) -> ssr builds first`,
      );
    }
  }
}
