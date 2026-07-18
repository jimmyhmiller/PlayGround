// Diffpack native Node server entry.
//
// Boots a `node:http` server that adapts every Node request into a Web `Request`,
// runs the app's SSR fetch handler (emitted in ./server.mjs, re-exported through
// ./_ssr/ssr.mjs), and writes the Web `Response` back to Node. Hashed client
// assets from the sibling `public/` directory are served directly. The server
// listens on `process.env.PORT` / `process.env.HOST`.
//
// This file is emitted verbatim by the Diffpack server build; it depends only on
// Node built-ins and the sibling modules the build writes next to it.
import { createServer } from "node:http";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createRequestListener } from "./_ssr/node-adapter.mjs";
import { fetch as ssrFetch } from "./_ssr/ssr.mjs";
import { tsrStartManifest } from "./_ssr/router.mjs";

const serverDir = dirname(fileURLToPath(import.meta.url));
const publicDir = join(serverDir, "..", "public");

// Touch the router manifest so a broken route table fails fast at boot rather
// than on the first request, and surface the route count in the startup log.
const routeCount = Object.keys(tsrStartManifest().routes).length;
const listener = createRequestListener(ssrFetch, publicDir);

const port = Number.parseInt(process.env.PORT ?? "3000", 10);
const host = process.env.HOST || "0.0.0.0";

// A route loader may `fetch('/api/...')` a same-origin API route during SSR
// (the `/users` route does). Node's `fetch` (undici) rejects a bare path with
// ERR_INVALID_URL — a browser resolves it against the document origin, but the
// server has none. Resolve a root-relative request against this server's own
// origin so it reaches the local API route handler over the loopback interface
// (0.0.0.0 is not a connectable address, so dial 127.0.0.1). This is the
// server's real origin, not a fabricated base.
const selfOrigin = `http://${host === "0.0.0.0" ? "127.0.0.1" : host}:${port}`;
const baseFetch = globalThis.fetch;
globalThis.fetch = (input, init) => {
  if (typeof input === "string" && input.startsWith("/")) {
    return baseFetch(selfOrigin + input, init);
  }
  if (input instanceof Request && input.url.startsWith("/")) {
    return baseFetch(new Request(selfOrigin + input.url, input), init);
  }
  return baseFetch(input, init);
};

createServer(listener).listen(port, host, () => {
  console.log(
    `diffpack server listening on http://${host}:${port} (${routeCount} routes)`,
  );
});
