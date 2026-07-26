// `next/headers` shim (diffpack next app-router adapter). These read the real
// per-request context the react-server render established (an AsyncLocalStorage
// carrying the request url/headers/cookie). They are `async` (Next 16 requires
// `await cookies()`/`await headers()`). Called OUTSIDE a request (no store), each
// HARD-ERRORS naming the missing context (repo no-silent-stub rule) rather than
// returning silently-empty values. Imported only by Server Components → lands only
// in the react-server graph (with node:async_hooks under the node condition).
import { requestAls } from "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/diff/integration/next-real/mdx/.diffpack-next/request-context.ts";

function parseCookieHeader(header) {
  const map = new Map();
  (header || "").split(";").forEach(function (pair) {
    const eq = pair.indexOf("=");
    if (eq === -1) return;
    const key = pair.slice(0, eq).trim();
    const value = pair.slice(eq + 1).trim();
    if (key) map.set(key, value);
  });
  return map;
}

export async function cookies() {
  const store = requestAls.getStore();
  if (!store) {
    // Tagged so the SSG prerenderer can distinguish a classifier gap (a route it
    // treated static that actually reads request state) from a generic render failure.
    throw Object.assign(new Error("diffpack next shim: cookies() was called outside a request context (no AsyncLocalStorage store) — call it inside a Server Component during a render"), { digest: "DIFFPACK_DYNAMIC_BAILOUT" });
  }
  const map = parseCookieHeader(store.cookieHeader);
  return {
    get(name) { return map.has(name) ? { name: name, value: map.get(name) } : undefined; },
    getAll(name) {
      const all = [];
      map.forEach(function (value, key) { if (!name || key === name) all.push({ name: key, value: value }); });
      return all;
    },
    has(name) { return map.has(name); },
    size: map.size,
  };
}

export async function headers() {
  const store = requestAls.getStore();
  if (!store) {
    throw Object.assign(new Error("diffpack next shim: headers() was called outside a request context (no AsyncLocalStorage store) — call it inside a Server Component during a render"), { digest: "DIFFPACK_DYNAMIC_BAILOUT" });
  }
  return store.headers;
}

export async function draftMode() {
  // Faithful: this adapter threads no draft cookie, so draft mode is always disabled;
  // enabling it would need a mutable response cookie the adapter does not provide.
  return {
    isEnabled: false,
    enable() { throw new Error("diffpack next shim: draftMode().enable() is not supported (no response-cookie plumbing)"); },
    disable() { throw new Error("diffpack next shim: draftMode().disable() is not supported (no response-cookie plumbing)"); },
  };
}
