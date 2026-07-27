// diffpack pages-router SERVER render entry (bundled to `server/server.mjs`).
//
// Exports `handleRequest(method, pathname, query, headers, body)` returning
// `{ status, headers, body }`. It performs classic (non-RSC) React SSR:
//   * matches the request path against the generated page table;
//   * runs the page's `getServerSideProps` / `getStaticProps` (redirect/notFound
//     honored) to produce `pageProps`;
//   * renders `<App Component={Page} pageProps={pageProps}/>` with `renderToString`
//     and the collected `next/head` elements, then wraps it in the custom
//     `_document` via `renderToStaticMarkup`, injecting `__NEXT_DATA__` and the
//     client bundle so the browser hydrates `#__next`;
//   * serves `pages/api/*` handlers and, for client navigations
//     (`?__nextDataReq=1`), returns the page props as JSON instead of HTML.
//
// The orchestrator (`pages-server.mjs`) imports this and wires it to Node's http
// server; this module runs the app's own bundled React.

import { renderToStaticMarkup, renderToString } from "react-dom/server";
import { RouterContext, createRouterEvents } from "./next-router.jsx";
import { HeadManagerContext } from "./pages-head-manager.jsx";
import { DocumentContext, matchPath } from "./pages-runtime.jsx";
import {
  App,
  Document,
  ErrorPage,
  apiRoutes,
  pages,
} from "./pages-manifest.server.js";

const CLIENT_ENTRY = "/client.js";

// --- Static generation (SSG) + Incremental Static Regeneration (ISR) cache --------
//
// `getStaticProps` pages are prerendered at BUILD time (`prerender()` below, driven
// by `pages-prerender.mjs`) and their props seeded here (`seedPrerender`) so the
// live server answers them from cache with ZERO per-request data fetch — the "static"
// behaviour. When a page also sets `revalidate: N`, the cached entry expires after N
// seconds and the NEXT request regenerates it in place (ISR): the value served stays
// stable inside the window and changes once past it.
//
// Keyed by the concrete pathname (e.g. `/`, `/blog/a`). Entry shape:
//   { pageProps, expires: number|Infinity, generation: number, prebuilt: boolean }
const isrCache = new Map();

// getStaticPaths tables, memoized per route pattern:
//   { known: Set<paramsKey>, fallback: false | true | "blocking" }
const staticPathsByPattern = new Map();

// Seed the ISR cache from the build-time prerender manifest (`prerender.json`). Called
// once by the orchestrator at startup. Revalidation windows start now (deploy time),
// matching Next's "regenerate at most every N seconds after serving" contract.
export function seedPrerender(data) {
  const now = Date.now();
  for (const entry of (data && data.entries) || []) {
    const expires = typeof entry.revalidate === "number" ? now + entry.revalidate * 1000 : Infinity;
    isrCache.set(entry.url, {
      pageProps: entry.pageProps || {},
      expires,
      generation: 0,
      prebuilt: true,
    });
  }
}

// A canonical key for a param set, in the route's declared key order (so `/blog/a`
// and `/blog/b` collide with neither each other nor a wrong route).
function paramsKey(page, params) {
  return page.keys
    .map((key) => {
      const value = params[key.name];
      return Array.isArray(value) ? value.join("/") : String(value);
    })
    .join("|");
}

// Reconstruct the concrete URL for a dynamic route + params (`/blog/[slug]` + {slug:"a"}
// -> `/blog/a`). Catch-all arrays join on "/".
function urlFor(page, params) {
  if (!page.keys.length) return page.pattern;
  let url = page.pattern;
  for (const key of page.keys) {
    const value = params[key.name];
    const encoded = Array.isArray(value)
      ? value.map(encodeURIComponent).join("/")
      : encodeURIComponent(String(value));
    const token = url.includes(`[[...${key.name}]]`)
      ? `[[...${key.name}]]`
      : url.includes(`[...${key.name}]`)
        ? `[...${key.name}]`
        : `[${key.name}]`;
    url = url.replace(token, encoded);
  }
  return url;
}

// Extract params from a getStaticPaths string path (`"/blog/a"`) via the route regex.
function paramsFromPath(page, pathString) {
  const params = {};
  const match = page.regex.exec(pathString);
  if (!match) return params;
  page.keys.forEach((key, index) => {
    const raw = match[index + 1];
    if (raw === undefined || raw === "") return;
    params[key.name] = key.catchall ? raw.split("/") : raw;
  });
  return params;
}

// Load (memoized) the getStaticPaths table for a route: the set of prerendered param
// keys plus the fallback mode. A dynamic `getStaticProps` route with no getStaticPaths
// is a hard error (Next requires one) — never a silent empty table.
async function getStaticPathsTable(page, mod) {
  if (staticPathsByPattern.has(page.pattern)) return staticPathsByPattern.get(page.pattern);
  const table = { known: new Set(), fallback: false };
  if (typeof mod.getStaticPaths === "function") {
    const result = await mod.getStaticPaths({});
    table.fallback = result && "fallback" in result ? result.fallback : false;
    for (const entry of (result && result.paths) || []) {
      const params = typeof entry === "string" ? paramsFromPath(page, entry) : entry.params || {};
      table.known.add(paramsKey(page, params));
    }
  } else if (page.keys.length) {
    throw new Error(
      `pages-router: dynamic route ${page.pattern} exports getStaticProps but no getStaticPaths — ` +
        "Next requires getStaticPaths to enumerate (or fall back for) its static paths",
    );
  }
  staticPathsByPattern.set(page.pattern, table);
  return table;
}

// Resolve pageProps for a `getStaticProps` route with SSG/ISR semantics. Returns
// `{ pageProps, state }` where state is one of "static" (served from the build-time
// prerender), "hit" (served from a runtime-cached generation), "miss" (first runtime
// generation), "stale" (revalidated after expiry) — or `{ notFound }` / `{ redirect }`.
async function resolveStatic(page, mod, params, pathname) {
  const now = Date.now();
  const cached = isrCache.get(pathname);
  if (cached && cached.expires > now) {
    return { pageProps: cached.pageProps, state: cached.prebuilt ? "static" : "hit" };
  }
  // Cache miss on a dynamic route: consult getStaticPaths for the fallback policy.
  if (!cached && page.keys.length) {
    const table = await getStaticPathsTable(page, mod);
    if (!table.known.has(paramsKey(page, params)) && table.fallback === false) {
      return { notFound: true };
    }
    // fallback true / "blocking": generate on demand below.
  }
  const result = await mod.getStaticProps({ params });
  if (result && result.notFound) return { notFound: true };
  if (result && result.redirect) return { redirect: result.redirect };
  const pageProps = (result && result.props) || {};
  const revalidate = result && typeof result.revalidate === "number" ? result.revalidate : null;
  const expires = revalidate != null ? now + revalidate * 1000 : Infinity;
  const generation = cached ? cached.generation + 1 : 0;
  isrCache.set(pathname, { pageProps, expires, generation, prebuilt: false });
  return { pageProps, state: cached ? "stale" : "miss" };
}

// Run the `getInitialProps` lifecycle. A custom `_app` with `App.getInitialProps`
// owns the whole thing (it decides whether/how to call the page's); otherwise the
// page's own `Component.getInitialProps` runs. Returns the resolved pageProps.
async function resolveInitialProps(Component, context, router) {
  if (typeof App.getInitialProps === "function") {
    const appResult = await App.getInitialProps({ Component, ctx: context, router });
    return (appResult && appResult.pageProps) || {};
  }
  if (typeof Component.getInitialProps === "function") {
    return (await Component.getInitialProps(context)) || {};
  }
  return {};
}

// Build-time static generation: run getStaticProps (and getStaticPaths for dynamic
// routes) for every SSG page and return the manifest the orchestrator seeds. Skips
// pages that opt out via notFound/redirect; never renders getServerSideProps pages.
export async function prerender() {
  const entries = [];
  for (const page of pages) {
    const mod = page.mod;
    if (typeof mod.getStaticProps !== "function") continue;
    let paramSets;
    if (page.keys.length) {
      if (typeof mod.getStaticPaths !== "function") {
        throw new Error(
          `pages-router prerender: dynamic route ${page.pattern} has getStaticProps but no getStaticPaths`,
        );
      }
      const result = await mod.getStaticPaths({});
      paramSets = ((result && result.paths) || []).map((entry) =>
        typeof entry === "string" ? paramsFromPath(page, entry) : entry.params || {},
      );
    } else {
      paramSets = [{}];
    }
    for (const params of paramSets) {
      const result = await mod.getStaticProps({ params });
      if (result && (result.notFound || result.redirect)) continue;
      entries.push({
        url: urlFor(page, params),
        pattern: page.pattern,
        pageProps: (result && result.props) || {},
        revalidate: result && typeof result.revalidate === "number" ? result.revalidate : null,
      });
    }
  }
  return { entries };
}

// Neutralize `</script>` breakout and HTML-comment injection in the inline JSON.
function escapeJson(json) {
  return json
    .replace(/</g, "\\u003c")
    .replace(/>/g, "\\u003e")
    .replace(/\u2028/g, "\\u2028")
    .replace(/\u2029/g, "\\u2029");
}

function serverRouter(pathname, query, params, pattern) {
  const noop = async () => {};
  return {
    pathname: pattern,
    route: pattern,
    asPath: pathname,
    query: { ...query, ...params },
    basePath: "",
    isReady: true,
    isFallback: false,
    isPreview: false,
    push: noop,
    replace: noop,
    back() {},
    forward() {},
    reload() {},
    prefetch: async () => {},
    beforePopState() {},
    events: createRouterEvents(),
  };
}

function renderDocument(Component, pageProps, pathname, query, params, pattern) {
  const head = [];
  const collector = { push: (children) => head.push(children) };
  const router = serverRouter(pathname, query, params, pattern);
  const appHtml = renderToString(
    <RouterContext.Provider value={router}>
      <HeadManagerContext.Provider value={collector}>
        <App Component={Component} pageProps={pageProps} />
      </HeadManagerContext.Provider>
    </RouterContext.Provider>,
  );
  const nextData = {
    props: { pageProps },
    page: pattern,
    query: { ...query, ...params },
    buildId: "diffpack",
  };
  const docCtx = {
    appHtml,
    head,
    nextDataJson: escapeJson(JSON.stringify(nextData)),
    clientEntry: CLIENT_ENTRY,
  };
  const documentHtml = renderToStaticMarkup(
    <DocumentContext.Provider value={docCtx}>
      <Document />
    </DocumentContext.Provider>,
  );
  return "<!DOCTYPE html>" + documentHtml;
}

function makeRes() {
  return {
    statusCode: 200,
    _headers: {},
    _body: "",
    finished: false,
    status(code) {
      this.statusCode = code;
      return this;
    },
    setHeader(key, value) {
      this._headers[String(key).toLowerCase()] = value;
      return this;
    },
    getHeader(key) {
      return this._headers[String(key).toLowerCase()];
    },
    json(obj) {
      this._headers["content-type"] = "application/json; charset=utf-8";
      this._body = JSON.stringify(obj);
      this.finished = true;
      return this;
    },
    send(body) {
      this._body = typeof body === "string" ? body : JSON.stringify(body);
      this.finished = true;
      return this;
    },
    write(chunk) {
      this._body += chunk;
      return this;
    },
    end(body) {
      if (body !== undefined) this._body += body;
      this.finished = true;
      return this;
    },
    redirect(code, url) {
      if (typeof code === "string") {
        url = code;
        code = 307;
      }
      this.statusCode = code;
      this._headers.location = url;
      this.finished = true;
      return this;
    },
  };
}

function parseBody(headers, bodyText) {
  if (!bodyText) return undefined;
  const type = (headers && (headers["content-type"] || headers["Content-Type"])) || "";
  if (type.includes("application/json")) {
    try {
      return JSON.parse(bodyText);
    } catch {
      return bodyText;
    }
  }
  return bodyText;
}

function renderError(statusCode, query, pathname) {
  if (query.__nextDataReq) {
    return {
      status: statusCode,
      headers: { "content-type": "application/json; charset=utf-8" },
      body: JSON.stringify({ pageProps: { statusCode }, page: "/_error" }),
    };
  }
  const html = renderDocument(ErrorPage, { statusCode }, pathname, query, {}, "/_error");
  return {
    status: statusCode,
    headers: { "content-type": "text/html; charset=utf-8" },
    body: html,
  };
}

export async function handleRequest(method, pathname, query, headers, bodyText) {
  query = query || {};

  // API routes (`pages/api/*`).
  const apiMatch = matchPath(apiRoutes, pathname);
  if (apiMatch) {
    const handler = apiMatch.page.handler;
    if (typeof handler !== "function") {
      return {
        status: 500,
        headers: { "content-type": "application/json; charset=utf-8" },
        body: JSON.stringify({ error: `API route ${pathname} has no default export` }),
      };
    }
    const req = {
      method: method || "GET",
      url: pathname,
      query: { ...query, ...apiMatch.params },
      headers: headers || {},
      cookies: {},
      body: parseBody(headers, bodyText),
    };
    const res = makeRes();
    await handler(req, res);
    return { status: res.statusCode, headers: res._headers, body: res._body };
  }

  // Page routes.
  const match = matchPath(pages, pathname);
  if (!match) return renderError(404, query, pathname);

  const { page, params } = match;
  const mod = page.mod;
  const Component = mod.default;
  if (typeof Component !== "function") return renderError(500, query, pathname);

  const context = {
    params,
    query: { ...query, ...params },
    pathname: page.pattern,
    asPath: pathname,
    req: { method, url: pathname, headers: headers || {} },
    res: makeRes(),
    resolvedUrl: pathname,
  };

  // Resolve pageProps through the data-fetching lifecycle. Exactly one of
  // getServerSideProps / getStaticProps / getInitialProps drives a page; the first
  // two are mutually exclusive with the third, mirroring Next.
  let pageProps = {};
  let isrState = null;
  if (typeof mod.getServerSideProps === "function") {
    // Per-request server rendering.
    const result = await mod.getServerSideProps(context);
    if (result && result.notFound) return renderError(404, query, pathname);
    if (result && result.redirect) {
      return {
        status: result.redirect.permanent ? 308 : 307,
        headers: { location: result.redirect.destination },
        body: "",
      };
    }
    pageProps = (result && result.props) || {};
  } else if (typeof mod.getStaticProps === "function") {
    // Static generation + ISR (served from the build-time prerender, regenerated on
    // expiry). getStaticPaths gates unknown dynamic paths.
    const resolved = await resolveStatic(page, mod, params, pathname);
    if (resolved.notFound) return renderError(404, query, pathname);
    if (resolved.redirect) {
      return {
        status: resolved.redirect.permanent ? 308 : 307,
        headers: { location: resolved.redirect.destination },
        body: "",
      };
    }
    pageProps = resolved.pageProps;
    isrState = resolved.state;
  } else {
    // getInitialProps (page and/or _app), run per request.
    const router = serverRouter(pathname, query, params, page.pattern);
    pageProps = await resolveInitialProps(Component, context, router);
  }

  // A getServerSideProps/getStaticProps page may set response headers/status via
  // context.res (e.g. res.setHeader). Surface them.
  const extraHeaders = { ...context.res._headers };
  if (isrState) extraHeaders["x-diffpack-isr"] = isrState;

  // Client navigation: return props as JSON instead of a full document.
  if (query.__nextDataReq) {
    return {
      status: 200,
      headers: { ...extraHeaders, "content-type": "application/json; charset=utf-8" },
      body: JSON.stringify({ pageProps, page: page.pattern }),
    };
  }

  const html = renderDocument(Component, pageProps, pathname, query, params, page.pattern);
  return {
    status: 200,
    headers: { ...extraHeaders, "content-type": "text/html; charset=utf-8" },
    body: html,
  };
}

export default { handleRequest, prerender, seedPrerender };
