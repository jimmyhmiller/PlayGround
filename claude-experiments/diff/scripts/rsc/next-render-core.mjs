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
import { existsSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { MANIFEST_FILES, loadServerConsumerManifest } from "./ssr-module-map.mjs";

/// Read the three client-references manifests and build the divergent-id
/// ssrModuleMapping (Manifest #2) the SSR bundle resolves the flight's client
/// references through. The join and its checks live in `./ssr-module-map.mjs` so the
/// orchestrator, the prerenderer and the fixture checks share one rule.
export function loadManifests(outputDir) {
  try {
    return loadServerConsumerManifest(outputDir);
  } catch (error) {
    throw new Error(`next-render-core: ${error.message}`, { cause: error });
  }
}

function pickRender(mod) {
  const fn = mod.renderFlightToDocument || (mod.default && mod.default.renderFlightToDocument);
  if (typeof fn !== "function") {
    throw new Error("next-render-core: the SSR bundle does not export renderFlightToDocument");
  }
  return fn;
}

/// Memoized dynamic import of the SSR-of-flight bundle → its `renderFlightToDocument`.
/// The bundle is imported ONCE and cached.
///
/// There is deliberately no dev/watch mode here. This module is the BUILD-TIME seam
/// (the SSG prerenderer): the bundle it reads is written once and never re-emitted
/// underneath it. Live-reload freshness belongs to `next-server.mjs`, which owns the
/// dev hot-update channel — and re-importing this entry could not deliver it anyway,
/// because the entry reaches its split chunks through query-less
/// `import("./server.chunk-N.mjs")` URLs that Node serves from its ESM cache.
/// Passing `{ dev: true }` is therefore a hard error, not a silently ignored option.
export function getRenderFlightToDocument(ssrEntry, options) {
  if (options && options.dev) {
    throw new Error(
      "next-render-core: `dev` is not supported — this is the build-time render seam. Dev freshness is next-server.mjs's hot-update channel (POST /__diffpack_dev/hot).",
    );
  }
  let cache = { fn: null };
  return async function render(...args) {
    if (!cache.fn) cache.fn = pickRender(await import(pathToFileURL(ssrEntry).href));
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
      // `--enable-source-maps`, not `process.setSourceMapsEnabled()`: this child's
      // entry IS an emitted chunk (`rsc-render/server.mjs`), so there is no
      // diffpack-authored line in it to make the call from, and source-map support
      // does not cross a process boundary. Without the flag the Server Component
      // render — the layer whose exceptions are hardest to place — reports
      // positions in `server.chunk-N.mjs` while its `.map` sits unread beside it.
      const child = spawn(process.execPath, ["--enable-source-maps", rscRenderEntry, ...args], {
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
          // next/cache: the cache tags this page read (captured so the prerender manifest
          // can register the pathname under them for revalidateTag).
          tags: parsed.tags || [],
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
    ...MANIFEST_FILES.map(([label, file]) => [label, join(outputDir, file)]),
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
