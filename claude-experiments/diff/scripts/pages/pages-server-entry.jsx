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
// With built-in i18n configured (next.config `i18n`), a leading `/<locale>` is split
// off the request path before matching (the default locale is served unprefixed), the
// active locale is exposed on `router.locale` / `locales` / `defaultLocale`, passed to
// `getStaticProps` / `getServerSideProps` / `getStaticPaths`, serialized into
// `__NEXT_DATA__`, and rendered as `<html lang>`.
//
// The orchestrator (`pages-server.mjs`) imports this and wires it to Node's http
// server; this module runs the app's own bundled React.

import { renderToStaticMarkup, renderToString } from "react-dom/server";
import { RouterContext, createRouterEvents } from "./next-router.jsx";
import { HeadManagerContext } from "./pages-head-manager.jsx";
import { DocumentContext, matchPath } from "./pages-runtime.jsx";
import { addLocale, detectLocale, splitLocale } from "./pages-url.js";
import {
  App,
  Document,
  ErrorPage,
  apiRoutes,
  i18n,
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
// Keyed by the concrete REQUEST pathname, locale prefix included (`/`, `/blog/a`,
// `/fr/blog/a`), so the same route in two locales holds two independent entries.
// Entry shape:
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

// Every locale the app builds for; `[null]` (one nameless locale) when i18n is off,
// so locale-agnostic code paths iterate exactly once.
function localeList() {
  return i18n ? i18n.locales : [null];
}

// The getStaticPaths key for a (locale, params) pair.
function localeKey(locale, key) {
  return `${locale == null ? "" : locale}::${key}`;
}

// The i18n fields Next adds to every data-fetching context. `{}` when the app
// configures no i18n, so a plain app sees exactly the context it saw before.
function localeContext(locale) {
  if (!i18n) return {};
  return { locale, locales: i18n.locales, defaultLocale: i18n.defaultLocale };
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
    const result = await mod.getStaticPaths(localeContext(i18n ? i18n.defaultLocale : undefined));
    table.fallback = result && "fallback" in result ? result.fallback : false;
    for (const entry of (result && result.paths) || []) {
      const params = typeof entry === "string" ? paramsFromPath(page, entry) : entry.params || {};
      const locale = typeof entry === "string" ? undefined : entry.locale;
      // An entry without a `locale` is prerendered for every locale (Next's rule).
      for (const each of locale !== undefined ? [locale] : localeList()) {
        table.known.add(localeKey(each, paramsKey(page, params)));
      }
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
async function resolveStatic(page, mod, params, cacheKey, locale) {
  const now = Date.now();
  const cached = isrCache.get(cacheKey);
  if (cached && cached.expires > now) {
    return { pageProps: cached.pageProps, state: cached.prebuilt ? "static" : "hit" };
  }
  // Cache miss on a dynamic route: consult getStaticPaths for the fallback policy.
  if (!cached && page.keys.length) {
    const table = await getStaticPathsTable(page, mod);
    if (!table.known.has(localeKey(locale, paramsKey(page, params))) && table.fallback === false) {
      return { notFound: true };
    }
    // fallback true / "blocking": generate on demand below.
  }
  const result = await mod.getStaticProps({ params, ...localeContext(locale) });
  if (result && result.notFound) return { notFound: true };
  if (result && result.redirect) return { redirect: result.redirect };
  const pageProps = (result && result.props) || {};
  const revalidate = result && typeof result.revalidate === "number" ? result.revalidate : null;
  const expires = revalidate != null ? now + revalidate * 1000 : Infinity;
  const generation = cached ? cached.generation + 1 : 0;
  isrCache.set(cacheKey, { pageProps, expires, generation, prebuilt: false });
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
    // Each job is one (params, locale) pair to prerender. With i18n off there is a
    // single nameless locale, so this is the old one-entry-per-params behaviour.
    let jobs;
    if (page.keys.length) {
      if (typeof mod.getStaticPaths !== "function") {
        throw new Error(
          `pages-router prerender: dynamic route ${page.pattern} has getStaticProps but no getStaticPaths`,
        );
      }
      const result = await mod.getStaticPaths(
        localeContext(i18n ? i18n.defaultLocale : undefined),
      );
      jobs = [];
      for (const entry of (result && result.paths) || []) {
        const params = typeof entry === "string" ? paramsFromPath(page, entry) : entry.params || {};
        const locale = typeof entry === "string" ? undefined : entry.locale;
        for (const each of locale !== undefined ? [locale] : localeList()) {
          jobs.push({ params, locale: each });
        }
      }
    } else {
      jobs = localeList().map((locale) => ({ params: {}, locale }));
    }
    for (const { params, locale } of jobs) {
      const result = await mod.getStaticProps({ params, ...localeContext(locale) });
      if (result && (result.notFound || result.redirect)) continue;
      entries.push({
        url: addLocale(urlFor(page, params), locale, i18n && i18n.defaultLocale),
        pattern: page.pattern,
        locale: locale == null ? null : locale,
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

function serverRouter(pathname, query, params, pattern, locale) {
  const noop = async () => {};
  return {
    pathname: pattern,
    route: pattern,
    // Next's `asPath` excludes the locale prefix; `pathname` here is already the
    // locale-free request path.
    asPath: pathname,
    query: { ...query, ...params },
    basePath: "",
    ...localeContext(locale),
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

function renderDocument(Component, pageProps, pathname, query, params, pattern, locale) {
  const head = [];
  const collector = { push: (children) => head.push(children) };
  const router = serverRouter(pathname, query, params, pattern, locale);
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
    ...localeContext(locale),
  };
  const docCtx = {
    appHtml,
    head,
    nextDataJson: escapeJson(JSON.stringify(nextData)),
    clientEntry: CLIENT_ENTRY,
    // `<html lang>` — the default `_document` (and `next/document`'s `Html`) render it.
    locale: i18n ? locale : null,
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
  if (type.includes("application/x-www-form-urlencoded")) {
    // Next parses urlencoded bodies into a plain object (like req.query). Repeated
    // keys collapse to the last value, matching URLSearchParams -> Object semantics.
    const params = new URLSearchParams(bodyText);
    const out = {};
    for (const [key, value] of params.entries()) out[key] = value;
    return out;
  }
  return bodyText;
}

// Parse the `Cookie` request header into a `{ name: value }` map (values URL-decoded),
// mirroring Next's `req.cookies`. Header names arrive already lowercased from Node's
// http server, but we accept either case defensively. No Cookie header -> `{}`, which
// is the correct empty-cookie state, not a silenced failure.
function parseCookies(headers) {
  const raw = headers && (headers.cookie || headers.Cookie);
  const cookies = {};
  if (!raw) return cookies;
  for (const pair of String(raw).split(";")) {
    const eq = pair.indexOf("=");
    if (eq < 0) continue;
    const name = pair.slice(0, eq).trim();
    if (!name) continue;
    let value = pair.slice(eq + 1).trim();
    // A quoted cookie value ("...") is unwrapped, then percent-decoded.
    if (value.length >= 2 && value.startsWith('"') && value.endsWith('"')) {
      value = value.slice(1, -1);
    }
    try {
      value = decodeURIComponent(value);
    } catch {
      // Leave a malformed percent-sequence as-is rather than throwing.
    }
    cookies[name] = value;
  }
  return cookies;
}

function renderError(statusCode, query, pathname, locale) {
  if (query.__nextDataReq) {
    return {
      status: statusCode,
      headers: { "content-type": "application/json; charset=utf-8" },
      body: JSON.stringify({ pageProps: { statusCode }, page: "/_error" }),
    };
  }
  const html = renderDocument(ErrorPage, { statusCode }, pathname, query, {}, "/_error", locale);
  return {
    status: statusCode,
    headers: { "content-type": "text/html; charset=utf-8" },
    body: html,
  };
}

export async function handleRequest(method, pathname, query, headers, bodyText) {
  query = query || {};

  // API routes (`pages/api/*`). Next does NOT locale-prefix them, so they are matched
  // against the raw request path, before any locale split.
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
      cookies: parseCookies(headers),
      body: parseBody(headers, bodyText),
    };
    const res = makeRes();
    await handler(req, res);
    return { status: res.statusCode, headers: res._headers, body: res._body };
  }

  // Built-in i18n: split a leading `/<locale>` off the request path. The default
  // locale is served unprefixed; `routePath` is what the route table matches and what
  // `router.asPath` reports (Next's asPath excludes the locale).
  const requestPath = pathname;
  const { locale, pathname: routePath, prefixed } = splitLocale(i18n, pathname);

  // Next detects a locale from `NEXT_LOCALE` / `Accept-Language` at the bare root only,
  // and redirects there when it differs from the default locale.
  if (i18n && !prefixed && routePath === "/" && !query.__nextDataReq) {
    const detected = detectLocale(i18n, headers, parseCookies(headers));
    if (detected !== i18n.defaultLocale) {
      return { status: 307, headers: { location: addLocale("/", detected, i18n.defaultLocale) }, body: "" };
    }
  }

  // Page routes.
  const match = matchPath(pages, routePath);
  if (!match) return renderError(404, query, routePath, locale);

  const { page, params } = match;
  const mod = page.mod;
  const Component = mod.default;
  if (typeof Component !== "function") return renderError(500, query, routePath, locale);

  const context = {
    params,
    query: { ...query, ...params },
    pathname: page.pattern,
    asPath: routePath,
    req: { method, url: requestPath, headers: headers || {} },
    res: makeRes(),
    resolvedUrl: routePath,
    ...localeContext(locale),
  };

  // Resolve pageProps through the data-fetching lifecycle. Exactly one of
  // getServerSideProps / getStaticProps / getInitialProps drives a page; the first
  // two are mutually exclusive with the third, mirroring Next.
  let pageProps = {};
  let isrState = null;
  if (typeof mod.getServerSideProps === "function") {
    // Per-request server rendering.
    const result = await mod.getServerSideProps(context);
    if (result && result.notFound) return renderError(404, query, routePath, locale);
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
    const resolved = await resolveStatic(page, mod, params, requestPath, locale);
    if (resolved.notFound) return renderError(404, query, routePath, locale);
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
    const router = serverRouter(routePath, query, params, page.pattern, locale);
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
      body: JSON.stringify({ pageProps, page: page.pattern, ...localeContext(locale) }),
    };
  }

  const html = renderDocument(
    Component,
    pageProps,
    routePath,
    query,
    params,
    page.pattern,
    locale,
  );
  return {
    status: 200,
    headers: { ...extraHeaders, "content-type": "text/html; charset=utf-8" },
    body: html,
  };
}

export default { handleRequest, prerender, seedPrerender };
