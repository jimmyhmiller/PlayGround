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
    req: { method, url: pathname, headers: headers || {} },
    res: makeRes(),
    resolvedUrl: pathname,
  };

  let pageProps = {};
  const dataFn =
    typeof mod.getServerSideProps === "function"
      ? mod.getServerSideProps
      : typeof mod.getStaticProps === "function"
        ? mod.getStaticProps
        : null;
  if (dataFn) {
    const result = await dataFn(context);
    if (result && result.notFound) return renderError(404, query, pathname);
    if (result && result.redirect) {
      return {
        status: result.redirect.permanent ? 308 : 307,
        headers: { location: result.redirect.destination },
        body: "",
      };
    }
    pageProps = (result && result.props) || {};
  }

  // Client navigation: return props as JSON instead of a full document.
  if (query.__nextDataReq) {
    return {
      status: 200,
      headers: { "content-type": "application/json; charset=utf-8" },
      body: JSON.stringify({ pageProps, page: page.pattern }),
    };
  }

  const html = renderDocument(Component, pageProps, pathname, query, params, page.pattern);
  return {
    status: 200,
    headers: { "content-type": "text/html; charset=utf-8" },
    body: html,
  };
}

export default { handleRequest };
