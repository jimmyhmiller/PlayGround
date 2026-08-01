// Node <-> Web adapter for the Diffpack server entry.
//
// Turns a Node `IncomingMessage` into a Web `Request`, serves a matching static
// file from the emitted `public/` directory when one exists, and otherwise runs
// the app's Web fetch handler and streams the Web `Response` back to Node.
import { readFile, stat } from "node:fs/promises";
import { join, normalize, resolve, sep } from "node:path";

const MIME = {
  ".js": "application/javascript",
  ".mjs": "application/javascript",
  ".cjs": "application/javascript",
  ".json": "application/json; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".ico": "image/x-icon",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
  ".webp": "image/webp",
  ".avif": "image/avif",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".ttf": "font/ttf",
  ".webmanifest": "application/manifest+json",
  ".wasm": "application/wasm",
};

export function contentTypeFor(path) {
  const dot = path.lastIndexOf(".");
  const ext = dot === -1 ? "" : path.slice(dot).toLowerCase();
  return MIME[ext] ?? "application/octet-stream";
}

// Build a Web `Request` mirroring the Node request: method, every raw header,
// and (for methods that carry one) the request stream as the body.
export function toWebRequest(req, hostHeader) {
  const url = `http://${hostHeader}${req.url}`;
  const headers = new Headers();
  const raw = req.rawHeaders;
  for (let i = 0; i + 1 < raw.length; i += 2) {
    headers.append(raw[i], raw[i + 1]);
  }
  const method = req.method || "GET";
  const hasBody = method !== "GET" && method !== "HEAD";
  return new Request(url, {
    method,
    headers,
    body: hasBody ? req : undefined,
    duplex: hasBody ? "half" : undefined,
  });
}

// Copy a Web `Response` onto the Node response, streaming the body when present.
export async function writeWebResponse(webResponse, res, method) {
  res.statusCode = webResponse.status;
  if (webResponse.statusText) res.statusMessage = webResponse.statusText;
  for (const [key, value] of webResponse.headers) res.setHeader(key, value);
  if (method === "HEAD" || webResponse.body == null) {
    if (webResponse.body != null) await webResponse.arrayBuffer();
    res.end();
    return;
  }
  const reader = webResponse.body.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    res.end();
  }
}

// Resolve `pathname` inside `publicDir` and, if it maps to a real file, return
// its bytes and content type. Path traversal outside `publicDir` yields null.
export async function serveStatic(publicDir, pathname) {
  if (pathname.includes("\0")) return null;
  let decoded;
  try {
    decoded = decodeURIComponent(pathname);
  } catch {
    return null;
  }
  const relative = normalize(decoded).replace(/^([/\\]|\.\.([/\\]|$))+/, "");
  const filePath = resolve(publicDir, relative);
  const root = resolve(publicDir);
  if (filePath !== root && !filePath.startsWith(root + sep)) return null;
  try {
    const info = await stat(filePath);
    if (!info.isFile()) return null;
    const body = await readFile(filePath);
    return { body, type: contentTypeFor(filePath) };
  } catch {
    return null;
  }
}

// Produce a `node:http` request listener: static files first, then the app's
// Web fetch handler. Handler errors become a 500 rather than a hung socket.
export function createRequestListener(fetchHandler, publicDir) {
  return async (req, res) => {
    const method = req.method || "GET";
    try {
      const pathname = (req.url || "/").split("?")[0];
      if (method === "GET" || method === "HEAD") {
        const asset = await serveStatic(publicDir, pathname);
        if (asset) {
          res.statusCode = 200;
          res.setHeader("content-type", asset.type);
          res.setHeader("content-length", asset.body.length);
          res.end(method === "HEAD" ? undefined : asset.body);
          return;
        }
      }
      const hostHeader = req.headers.host || "localhost";
      const webResponse = await fetchHandler(toWebRequest(req, hostHeader));
      await writeWebResponse(webResponse, res, method);
    } catch (error) {
      if (!res.headersSent) {
        res.statusCode = 500;
        res.setHeader("content-type", "text/plain; charset=utf-8");
      }
      res.end(`diffpack server error: ${(error && error.stack) || error}`);
    }
  };
}
