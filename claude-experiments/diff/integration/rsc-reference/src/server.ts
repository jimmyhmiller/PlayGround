// The SSR entry (Target::Server). diffpack bundles this graph under node
// conditions, so `react`, `react-dom/server`, and `react-server-dom-webpack/client`
// are inlined as ONE React copy — the copy that must also own the `"use client"`
// island (bundled here as REAL code, since `"use client"` ships as real code in the
// SSR graph), or the island's `useState` would call a foreign dispatcher. The
// react-server render graph has its OWN inlined React in its own child process; the
// two only exchange flight BYTES.
//
// This module exports `renderFlightToHTML`: it reconstructs the flight the
// react-server render produced (client references resolved to THIS build's real
// modules through the `__webpack_*` seam over its own registry) and renders it to
// HTML with react-dom/server. The flight carries the CLIENT build's ids; the
// `serverConsumerManifest` (Manifest #2) the orchestrator passes maps each client
// id to THIS build's id for the same module (the divergent-id `ssrModuleMapping`).
import { createFromReadableStream } from "react-server-dom-webpack/client";
import { renderToStaticMarkup } from "react-dom/server";
// Bundle + register the island as real code under this build's runtime id, so the
// flight's client reference resolves back to it (its useState runs under this
// build's React).
import { Counter } from "./Counter";

// Force a code split so the build uses the registry runtime (and therefore a
// require-able registry the seam maps `__webpack_require__` onto), rather than the
// single-chunk scope-hoisted output which has no registry.
import("./lazy").then((module) => {
  (globalThis as Record<string, unknown>).__diffpack_ssr_lazy = module.value;
});

// Keep the island export reachable so it is retained and assigned a runtime id.
(globalThis as Record<string, unknown>).__diffpack_ssr_island = Counter;

// Install the `__webpack_*` seam over THIS bundle's registry. A server build does
// not get the browser seam prelude (that is BrowserEsm-only), so the SSR pass
// installs it here over its own runtime: `__webpack_require__(id)` returns the
// module registered under `id` in THIS graph. Every module is already loaded
// in-process (static import + the forced split), so chunk loading is a resolved
// no-op and an unknown id is still a hard error, never a silent miss.
function installSeam(): void {
  const runtimeKey = Object.keys(globalThis).find((key) =>
    key.startsWith("__diffpack_runtime:"),
  );
  if (!runtimeKey) {
    throw new Error(
      "diffpack rsc ssr: no __diffpack_runtime:* registry on globalThis; the SSR bundle must use the registry runtime (a code split forces it)",
    );
  }
  const runtime = (globalThis as Record<string, { require(id: number): unknown }>)[
    runtimeKey
  ];
  const g = globalThis as Record<string, unknown>;
  g.__webpack_require__ = (id: number) => runtime.require(id);
  (g.__webpack_require__ as { u: (c: unknown) => unknown }).u = (c: unknown) => c;
  g.__webpack_chunk_load__ = (_c: unknown) => Promise.resolve();
}

export async function renderFlightToHTML(
  flightBytes: Uint8Array,
  serverConsumerManifest: unknown,
): Promise<string> {
  installSeam();
  const bytes = new Uint8Array(flightBytes);
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
  const root = await createFromReadableStream(stream, {
    serverConsumerManifest,
    // Server references (the `increment` action passed as a prop) are reconstructed
    // but never invoked during the initial server render; a call would be a bug.
    callServer() {
      throw new Error("diffpack rsc ssr: a server action was called during SSR");
    },
  });
  return renderToStaticMarkup(root as React.ReactElement);
}
