// SSR fetch-handler entry.
//
// Resolves the app's Web `Request -> Response` handler from the emitted server
// bundle (../server.mjs). That bundle's default export is the TanStack Start
// server entry; depending on interop the handler is exposed directly, as a
// `.fetch` method, or one namespace level down (`.default.fetch`). All shapes
// are unwrapped here so the rest of the runtime has a single `fetch` to call.
import serverEntry from "../server.mjs";

export function resolveFetch(entry) {
  const seen = new Set();
  const queue = [entry];
  while (queue.length > 0) {
    const candidate = queue.shift();
    if (candidate == null || seen.has(candidate)) continue;
    seen.add(candidate);
    if (typeof candidate === "function") return candidate;
    if (typeof candidate.fetch === "function") return candidate.fetch.bind(candidate);
    if (typeof candidate === "object") queue.push(candidate.default);
  }
  throw new Error(
    "diffpack ssr: ./server.mjs default export exposes no fetch handler",
  );
}

export const fetch = resolveFetch(serverEntry);
export default { fetch };
