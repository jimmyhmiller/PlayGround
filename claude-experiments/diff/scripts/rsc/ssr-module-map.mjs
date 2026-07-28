// Manifest #2 — the divergent-id `ssrModuleMapping` (`serverConsumerManifest.moduleMap`)
// that `react-server-dom-webpack/client`'s `resolveClientReference` reads. Built ONCE
// here and imported by every seam that turns a flight into HTML (the dev/prod
// orchestrator, the SSG prerenderer, and the two fixture checks), so the rule for what
// counts as a resolvable client reference cannot drift between them.
//
// THREE manifests, three graphs, one canonical module id (the absolute module path):
//
//   client-references-manifest.json        the BROWSER graph's ids + chunks. The
//                                          `bundlerConfig` the react-server render
//                                          serializes into the flight.
//   react-server-references-manifest.json  the REACT-SERVER graph's set. AUTHORITATIVE
//                                          for which client references a flight can
//                                          carry: a `"use client"` module reaches the
//                                          wire only if that graph resolved to it.
//   server-references-manifest.json        the SSR-of-flight graph's ids + chunks —
//                                          what the flight's client ids must map TO.
//
// The browser set and the server sets legitimately DIFFER. A package whose `exports`
// map sends the `browser` and `node` conditions to different files (`@sentry/nextjs`,
// for one) contributes a `"use client"` module to the browser graph that no server
// graph ever resolves — that module cannot appear in a flight, so demanding an SSR
// twin for it would reject a correct build. The check is therefore anchored on the
// react-server set, and a browser-only module still gets a `moduleMap` entry: one that
// throws by name the instant anything tries to resolve it, so the reasoning above can
// never degrade into a silently missing mapping.

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

export const CLIENT_REFERENCES_MANIFEST = "client-references-manifest.json";
export const SSR_REFERENCES_MANIFEST = "server-references-manifest.json";
export const REACT_SERVER_REFERENCES_MANIFEST = "react-server-references-manifest.json";

/// The three manifests plus the labels used when one is missing, in the order a build
/// produces them. Exported so each seam's "did you build all three graphs?" check and
/// this module's reader name the same files.
export const MANIFEST_FILES = [
  ["client-references manifest", CLIENT_REFERENCES_MANIFEST],
  ["react-server-references manifest", REACT_SERVER_REFERENCES_MANIFEST],
  ["ssr-references manifest", SSR_REFERENCES_MANIFEST],
];

/// A `moduleMap` entry for a `"use client"` module that exists only in the BROWSER
/// graph. Reaching it means the flight carried a client reference the react-server
/// graph could not have produced, so every property read throws, naming the module.
/// `toJSON` and symbol reads are let through so logging/serializing a whole moduleMap
/// stays possible (the entry serializes to a self-describing marker).
function unresolvableClientReference(moduleId) {
  const marker = {
    toJSON: () => `<no SSR reference: ${moduleId}>`,
  };
  return new Proxy(marker, {
    get(target, property, receiver) {
      if (typeof property === "symbol" || property === "toJSON") {
        return Reflect.get(target, property, receiver);
      }
      throw new Error(
        `no SSR reference for ${moduleId}: that "use client" module is in the browser ` +
          `graph but in neither server graph (its package's \`exports\` sends the ` +
          `\`browser\` and \`node\` conditions to different files), yet the flight asked ` +
          `to render it — so the react-server graph resolved to a module the SSR graph ` +
          `did not bundle`,
      );
    },
  });
}

/// Read all three manifests out of `outputDir`. A missing one is a hard error naming
/// the file and the graph that writes it — never an empty manifest.
export function readReferenceManifests(outputDir) {
  const read = (file, label) => {
    const path = join(outputDir, file);
    if (!existsSync(path)) {
      throw new Error(
        `${label} not found at ${path} — run the client -> react-server -> ssr builds first`,
      );
    }
    return JSON.parse(readFileSync(path, "utf8"));
  };
  return {
    clientRefs: read(CLIENT_REFERENCES_MANIFEST, "client-references manifest"),
    flightRefs: read(REACT_SERVER_REFERENCES_MANIFEST, "react-server-references manifest"),
    ssrRefs: read(SSR_REFERENCES_MANIFEST, "ssr-references manifest"),
    clientManifestPath: join(outputDir, CLIENT_REFERENCES_MANIFEST),
  };
}

/// Join the three manifests on the canonical module id into `moduleMap`, keyed by the
/// id the FLIGHT carries (the browser graph's) and resolving to the id the SSR graph
/// requires. Throws — naming every offending module and which graph lacks it — when a
/// module the react-server graph CAN put on the wire is absent from either other graph.
export function buildSsrModuleMap({ clientRefs, flightRefs, ssrRefs }) {
  const missingFromClient = [];
  const missingFromSsr = [];
  for (const moduleId of Object.keys(flightRefs)) {
    if (!clientRefs[moduleId]) missingFromClient.push(moduleId);
    else if (!ssrRefs[moduleId]) missingFromSsr.push(moduleId);
  }
  if (missingFromClient.length > 0 || missingFromSsr.length > 0) {
    const parts = [];
    if (missingFromClient.length > 0) {
      parts.push(
        `the CLIENT graph did not bundle ${missingFromClient.length} "use client" ` +
          `module(s) the react-server graph can reference, so the browser could never ` +
          `hydrate them:\n  ${missingFromClient.join("\n  ")}`,
      );
    }
    if (missingFromSsr.length > 0) {
      parts.push(
        `the SSR graph did not bundle ${missingFromSsr.length} "use client" module(s) ` +
          `the react-server graph can reference, so a flight carrying them cannot be ` +
          `rendered to HTML:\n  ${missingFromSsr.join("\n  ")}`,
      );
    }
    throw new Error(`client references are unresolvable: ${parts.join("\n")}`);
  }

  const moduleMap = {};
  for (const [moduleId, clientEntry] of Object.entries(clientRefs)) {
    const ssrEntryRef = ssrRefs[moduleId];
    moduleMap[String(clientEntry.id)] = ssrEntryRef
      ? { "*": { id: ssrEntryRef.id, chunks: ssrEntryRef.chunks, name: "*" } }
      : unresolvableClientReference(moduleId);
  }
  return moduleMap;
}

/// `readReferenceManifests` + `buildSsrModuleMap`, wrapped in the full
/// `serverConsumerManifest` shape `createFromReadableStream` takes. `moduleLoading` is
/// REQUIRED — the consumer reads `.prefix` off it unconditionally.
export function loadServerConsumerManifest(outputDir) {
  const manifests = readReferenceManifests(outputDir);
  const moduleMap = buildSsrModuleMap(manifests);
  return {
    ...manifests,
    moduleMap,
    serverConsumerManifest: {
      moduleMap,
      serverModuleMap: null,
      moduleLoading: { prefix: "", crossOrigin: null },
    },
  };
}
