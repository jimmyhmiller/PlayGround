// Router route -> client-chunk manifest for the server runtime.
//
// Re-exposes the natively generated TanStack Start manifest so the server graph
// has a stable, named routing entrypoint. The manifest module is emitted as a
// runtime-wrapped chunk whose default export is its module namespace, so the
// `tsrStartManifest` factory is unwrapped from there (falling back through CJS
// interop). `tsrStartManifest()` returns the route table
// (`{ routes: { <routeId>: { preloads, scripts } } }`) the SSR handler consults
// to emit each route's `<script>`/`<link>` asset tags.
import manifestModule from "../_tanstack-start-manifest_v.mjs";

function unwrap(namespace) {
  let current = namespace;
  const seen = new Set();
  while (current != null && !seen.has(current)) {
    if (typeof current.tsrStartManifest === "function") return current.tsrStartManifest;
    seen.add(current);
    current = current.default;
  }
  throw new Error(
    "diffpack router: manifest module exposes no tsrStartManifest factory",
  );
}

export const tsrStartManifest = unwrap(manifestModule);
export default { tsrStartManifest };
