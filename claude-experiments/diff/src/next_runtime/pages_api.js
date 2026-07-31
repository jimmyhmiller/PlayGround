// The pages-router API-route runtime, spliced verbatim into diffpack's SSR entry
// (see `ssr_entry_module`). It is real source, not a string, so the node regression
// tests import THIS file and exercise the same code the entry runs.
//
// A Next app may be HYBRID: `app/` renders the pages while `pages/api/**` serves the
// HTTP endpoints. Those endpoints use the pages-router contract — a Node
// `(req, res)` pair, not the Web `Request`/`Response` an app-router `route.ts`
// handler gets — so they need their own invocation path. cal.com is exactly this
// shape: next-auth (`pages/api/auth/[...nextauth].ts`) and every tRPC router are
// pages API routes, and with them unserved the client cannot read a session, cannot
// log in, and every data query 404s.
//
// WHICH GRAPH THIS LIVES IN IS LOAD-BEARING. Next compiles `pages/api/**` in its
// `api-node` layer — a plain Node layer WITHOUT the `react-server` export condition —
// while `app/**/route.ts` handlers are compiled in the react-server layer. diffpack
// mirrors that: this runtime and the `pages/api/**` modules are bundled into the SSR
// graph (node conditions, the ordinary React), NOT the react-server graph. Under the
// `react-server` condition `react-dom/server` resolves to React's stub whose only
// behaviour is `throw new Error("react-dom/server is not supported in React Server
// Components")` — and cal.com's `packages/emails/src/renderEmail.ts` does exactly
// `(await import("react-dom/server")).default` on every booking, so with these routes
// in the react-server graph every POST /api/book/event answered 500. Aliasing the
// module past the condition does not work either: the client-flavoured
// `react-dom/server` reads the CLIENT internals of `react`, and the react-server graph
// pins one React per environment (see `rsc_runtime_resolve::react_aliases`). The layer
// has to be right, not the alias.
//
// The `req`/`res` pair here are REAL `http.IncomingMessage` / `http.ServerResponse`
// objects driven over an in-memory socket, not hand-written look-alikes. That is what
// makes the surface complete: `res.writeHead`, repeated `Set-Cookie` headers,
// `res.getHeaders()`, the EventEmitter methods, and streaming reads of an unparsed
// body all work because they are Node's own implementations. The response is read
// back by parsing the bytes Node serialized, which is also why `shouldKeepAlive` is
// off: without keep-alive and without a content-length Node delimits the body by
// closing the connection instead of chunk-encoding it, so the captured bytes are the
// body exactly as written.
import { IncomingMessage, ServerResponse } from "node:http";
import { Duplex } from "node:stream";

/// Parse a `Cookie` header into Next's `req.cookies` map (values percent-decoded).
export function parseCookieHeader(raw) {
  const cookies = {};
  if (!raw) return cookies;
  for (const pair of String(raw).split(";")) {
    const eq = pair.indexOf("=");
    if (eq < 0) continue;
    const name = pair.slice(0, eq).trim();
    if (!name) continue;
    let value = pair.slice(eq + 1).trim();
    if (value.length >= 2 && value.startsWith('"') && value.endsWith('"')) {
      value = value.slice(1, -1);
    }
    try {
      value = decodeURIComponent(value);
    } catch {
      // A malformed percent-sequence is left as written rather than throwing.
    }
    cookies[name] = value;
  }
  return cookies;
}

/// Next's pages-router body parser: JSON and urlencoded bodies become objects, `text/*`
/// becomes a string, anything else stays a Buffer. Returns `undefined` for an empty
/// body, which is what a handler sees for a GET.
export function parseApiBody(contentType, raw) {
  if (raw == null || raw.length === 0) return undefined;
  const type = String(contentType || "");
  if (type.includes("application/json")) {
    const text = raw.toString("utf8");
    try {
      return JSON.parse(text);
    } catch {
      // Next answers a malformed JSON body with a 400 before the handler runs; the
      // caller turns this marker into that response rather than handing the handler a
      // body that silently disagrees with its Content-Type.
      return { __diffpackInvalidJson: text };
    }
  }
  if (type.includes("application/x-www-form-urlencoded")) {
    const params = new URLSearchParams(raw.toString("utf8"));
    const out = {};
    for (const key of new Set(params.keys())) {
      const all = params.getAll(key);
      out[key] = all.length > 1 ? all : all[0];
    }
    return out;
  }
  if (type.startsWith("text/")) return raw.toString("utf8");
  return raw;
}

/// The query object a pages API handler sees: the URL's search params merged with the
/// route's own dynamic params (a catch-all arrives as an array, which is exactly what
/// `req.query.nextauth` must be for next-auth to dispatch).
export function apiQuery(search, params) {
  const query = {};
  for (const key of new Set(search.keys())) {
    const all = search.getAll(key);
    query[key] = all.length > 1 ? all : all[0];
  }
  for (const [key, value] of Object.entries(params || {})) query[key] = value;
  return query;
}

/// Split the bytes Node serialized for a response back into `{ status, headers, body }`.
/// Header lines keep their order, so repeated `Set-Cookie` headers survive as separate
/// entries instead of being comma-joined into one broken cookie.
export function parseRawHttpResponse(buffer) {
  const split = buffer.indexOf("\r\n\r\n");
  if (split < 0) {
    throw new Error("diffpack pages api: the response Node serialized has no header terminator");
  }
  const lines = buffer.subarray(0, split).toString("latin1").split("\r\n");
  const statusLine = lines[0] || "";
  const status = Number(statusLine.split(" ")[1]) || 200;
  const headers = [];
  for (const line of lines.slice(1)) {
    const colon = line.indexOf(":");
    if (colon < 0) continue;
    headers.push([line.slice(0, colon).trim().toLowerCase(), line.slice(colon + 1).trim()]);
  }
  return { status, headers, body: buffer.subarray(split + 4) };
}

/// Attach the response helpers Next adds on top of `ServerResponse` for a pages API
/// route (`res.status().json()`, `res.send()`, `res.redirect()`), with Next's own
/// content-type defaulting.
function attachResponseHelpers(res) {
  res.status = (code) => {
    res.statusCode = code;
    return res;
  };
  res.send = (data) => {
    if (data === null || data === undefined) {
      res.end();
      return res;
    }
    if (Buffer.isBuffer(data)) {
      if (!res.getHeader("Content-Type")) res.setHeader("Content-Type", "application/octet-stream");
      res.setHeader("Content-Length", String(data.length));
      res.end(data);
      return res;
    }
    if (typeof data === "string") {
      if (!res.getHeader("Content-Type")) res.setHeader("Content-Type", "text/html; charset=utf-8");
      res.setHeader("Content-Length", String(Buffer.byteLength(data)));
      res.end(data);
      return res;
    }
    return res.json(data);
  };
  res.json = (body) => {
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    const text = JSON.stringify(body);
    res.setHeader("Content-Length", String(Buffer.byteLength(text)));
    res.end(text);
    return res;
  };
  res.redirect = (statusOrUrl, maybeUrl) => {
    const url = typeof statusOrUrl === "string" ? statusOrUrl : maybeUrl;
    const status = typeof statusOrUrl === "number" ? statusOrUrl : 307;
    if (typeof url !== "string") {
      throw new Error("diffpack pages api: res.redirect() requires a destination url");
    }
    res.writeHead(status, { Location: url });
    res.end();
    return res;
  };
  return res;
}

/// Invoke one pages-router API handler and serialize what it wrote.
///
/// `routeLabel` names the route in every diagnostic; `handler` is the module's default
/// export; `config` is its optional `export const config` (only `api.bodyParser === false`
/// changes behaviour — the raw bytes then reach the handler unparsed, which is how a
/// signature-verifying webhook endpoint has to receive them).
///
/// Returns the same `{ status, headers, body(base64), setCookies }` shape an app-router
/// `route.ts` handler returns, so the orchestrator serves both through one path.
export async function runPagesApiHandler({ routeLabel, handler, config, pathname, method, reqCtx, params }) {
  if (typeof handler !== "function") {
    throw new Error(
      `pages api route ${routeLabel} has no default-exported handler function; ` +
        "a `pages/api/**` module must `export default function handler(req, res)`",
    );
  }
  const href = reqCtx.url || "http://localhost" + pathname;
  const url = new URL(href, "http://localhost");
  const headers = {};
  for (const [key, value] of reqCtx.headers || []) {
    const name = String(key).toLowerCase();
    headers[name] = name in headers ? headers[name] + ", " + value : String(value);
  }
  if (reqCtx.cookie && !headers.cookie) headers.cookie = reqCtx.cookie;
  const raw =
    reqCtx.body == null
      ? Buffer.alloc(0)
      : reqCtx.bodyIsBase64
        ? Buffer.from(reqCtx.body, "base64")
        : Buffer.from(String(reqCtx.body));

  // TWO in-memory sockets, deliberately not one. `IncomingMessage` resumes its socket
  // when a handler reads the request as a stream, and a single shared socket then closes
  // under the `ServerResponse` attached to it — the response is destroyed mid-write, no
  // bytes come out and `finish` never fires. Separate sockets keep the directions
  // independent: request bytes are pushed onto the IncomingMessage, response bytes land
  // in `written`.
  const written = [];
  const requestSocket = new Duplex({
    read() {},
    write(_chunk, _encoding, callback) {
      callback();
    },
  });
  const responseSocket = new Duplex({
    read() {},
    write(chunk, _encoding, callback) {
      written.push(Buffer.from(chunk));
      callback();
    },
  });

  const req = new IncomingMessage(requestSocket);
  req.method = method || "GET";
  req.url = url.pathname + url.search;
  req.headers = headers;
  req.httpVersion = "1.1";
  req.httpVersionMajor = 1;
  req.httpVersionMinor = 1;
  // Next's pages-router request extensions.
  req.query = apiQuery(url.searchParams, params);
  req.cookies = parseCookieHeader(headers.cookie || "");
  const bodyParser = !(config && config.api && config.api.bodyParser === false);
  if (bodyParser) {
    const parsed = parseApiBody(headers["content-type"], raw);
    if (parsed && parsed.__diffpackInvalidJson !== undefined) {
      return {
        status: 400,
        headers: [["content-type", "text/plain; charset=utf-8"]],
        body: Buffer.from("Invalid JSON").toString("base64"),
        bodyIsBase64: true,
        setCookies: [],
      };
    }
    req.body = parsed;
  }
  // The raw bytes are ALWAYS readable off the stream, parsed or not: a handler that
  // opted out of the body parser (a signature-verifying webhook) reads them, and one
  // that did not simply never reads.
  if (raw.length) req.push(raw);
  req.push(null);

  const res = new ServerResponse(req);
  // No keep-alive and no implicit chunking: Node then delimits the body by closing the
  // connection, so `written` holds the header block followed by the body verbatim.
  res.shouldKeepAlive = false;
  res.useChunkedEncodingByDefault = false;
  attachResponseHelpers(res);
  const finished = new Promise((resolve) => res.on("finish", resolve));
  res.assignSocket(responseSocket);

  await handler(req, res);
  if (!res.writableEnded) {
    // Next warns here and leaves the socket open. Diffpack cannot: the react-server
    // worker answers requests over one pipe, so a response that never ends would wedge
    // every later request behind it. End it, and say so — loudly, naming the route.
    console.error(
      `[diffpack] next: the pages api route ${routeLabel} resolved without sending a response. ` +
        "Diffpack closed it with an empty 200 so the server keeps answering; the handler must " +
        "call res.end()/res.json()/res.send().",
    );
    res.end();
  }
  await finished;
  res.detachSocket(responseSocket);

  const { status, headers: rawHeaders, body } = parseRawHttpResponse(Buffer.concat(written));
  const setCookies = [];
  const plain = [];
  for (const [key, value] of rawHeaders) {
    if (key === "set-cookie") setCookies.push(value);
    else plain.push([key, value]);
  }
  return {
    status,
    headers: plain,
    body: body.toString("base64"),
    bodyIsBase64: true,
    setCookies,
  };
}

/// Match one `pages/api/**` route's segment pattern against the request's path parts,
/// capturing dynamic params. Same four segment kinds (and the same semantics) the
/// app-router matcher in the react-server entry uses: Static matches one part exactly,
/// Dynamic one part, CatchAll the (>=1) tail, OptionalCatchAll the (>=0) tail. Returns
/// the params object or null.
export function matchApiSegments(segments, parts) {
  const params = {};
  let i = 0;
  for (const seg of segments) {
    if (seg.k === "static") {
      if (parts[i] !== seg.v) return null;
      i += 1;
    } else if (seg.k === "dynamic") {
      if (i >= parts.length) return null;
      params[seg.v] = decodeURIComponent(parts[i]);
      i += 1;
    } else if (seg.k === "catchall") {
      if (i >= parts.length) return null;
      params[seg.v] = parts.slice(i).map(decodeURIComponent);
      i = parts.length;
    } else if (seg.k === "optcatchall") {
      params[seg.v] = parts.slice(i).map(decodeURIComponent);
      i = parts.length;
    } else {
      return null;
    }
  }
  return i === parts.length ? params : null;
}

/// Dispatch a request to the first matching entry of a `pages/api/**` route table.
/// Each entry is `{ path, segments, load }` where `load()` imports the route module
/// (its own chunk, so a route costs nothing until a request reaches it). Returns the
/// `{ status, headers, body(base64), setCookies }` shape `runPagesApiHandler` produces,
/// or `null` when no entry matches.
export async function dispatchPagesApi(table, pathname, method, reqCtx) {
  const parts = pathname.split("/").filter(Boolean);
  for (const entry of table) {
    const params = matchApiSegments(entry.segments, parts);
    if (!params) continue;
    const ns = await entry.load();
    const mod = ns && ns.default !== undefined ? ns : { default: ns };
    return runPagesApiHandler({
      routeLabel: entry.path,
      handler: mod.default,
      config: ns && ns.config,
      pathname,
      method: method || "GET",
      reqCtx: reqCtx || {},
      params,
    });
  }
  return null;
}
