// The browser entry (Target::Client). diffpack emits it as browser ESM with the
// RSC `__webpack_*` seam installed over its registry (the client build hosts a
// `"use client"` module, so the seam prelude is emitted). It reconstructs the
// inlined flight and hydrates the server-rendered HTML.
//
// The flight carries the CLIENT build's ids (the react-server render used Manifest
// #1 = these ids as its bundlerConfig), and this bundle IS the client build, so the
// browser resolves each client reference through its own registry with NO consumer
// manifest — the identity path. The `increment` server reference the flight carries
// is reconstructed into a callable through `callServer`, which POSTs to `/_action/`.
import { createFromReadableStream } from "react-server-dom-webpack/client";
import { hydrateRoot } from "react-dom/client";
import { use } from "react";
import { callServer } from "#diffpack-call-server";
// Bundle + register the island as real code under this build's runtime id, so the
// flight's client reference resolves to it and `hydrateRoot` attaches its handlers.
import { Counter } from "./Counter";

(globalThis as Record<string, unknown>).__diffpack_client_island = Counter;

// Force a code split so the client build uses the registry runtime + the RSC seam.
import("./lazy").then((module) => {
  (globalThis as Record<string, unknown>).__diffpack_client_lazy = module.value;
});

function Root({ tree }: { tree: Promise<React.ReactNode> }): React.ReactNode {
  return use(tree);
}

function decodeFlight(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function boot(): void {
  const flightBase64 = (window as Record<string, unknown>).__DIFFPACK_FLIGHT__ as
    | string
    | undefined;
  if (!flightBase64) {
    throw new Error(
      "diffpack rsc client: window.__DIFFPACK_FLIGHT__ is missing; the server must inline the flight payload",
    );
  }
  const bytes = decodeFlight(flightBase64);
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
  const tree = createFromReadableStream(stream, { callServer }) as Promise<React.ReactNode>;
  const container = document.getElementById("root");
  if (!container) throw new Error("diffpack rsc client: #root container not found");
  hydrateRoot(container, <Root tree={tree} />);
}

boot();
