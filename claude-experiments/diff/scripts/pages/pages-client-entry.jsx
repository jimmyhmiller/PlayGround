// diffpack pages-router CLIENT hydration entry (bundled to `public/client.js`).
//
// Reads the server-injected `__NEXT_DATA__`, resolves the current page from the
// generated route table, and hydrates `#__next` with `<App Component={Page}
// pageProps={...}/>` inside a real navigating RouterContext. `next/link` and
// `router.push/replace/back` navigate client-side: they fetch the target page's
// props (`?__nextDataReq=1`, served by the same SSR handler) and swap the view.
//
// SHALLOW routing (`router.push(url, as, { shallow: true })`) is the exception:
// it changes the URL, `asPath` and `query` WITHOUT re-running the page's data
// fetching, keeping the props the page already has. Next allows it only for a URL
// change within the same page; a shallow request that lands on a different route
// is a normal navigation (and says so).

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { hydrateRoot } from "react-dom/client";
import { RouterContext, createRouterEvents } from "./next-router.jsx";
import { HeadManagerContext } from "./pages-head-manager.jsx";
import { matchPath } from "./pages-runtime.jsx";
import { splitLocale } from "./pages-url.js";
import RouteAnnouncerPortal from "./route-announcer.jsx";
import { App, ErrorPage, i18n, pages } from "./pages-manifest.client.js";

function readNextData() {
  const el = document.getElementById("__NEXT_DATA__");
  const fallback = { props: { pageProps: {} }, page: location.pathname, query: {} };
  if (!el) return fallback;
  try {
    return JSON.parse(el.textContent);
  } catch {
    return fallback;
  }
}

function parseQuery(search) {
  const query = {};
  new URLSearchParams(search || "").forEach((value, key) => {
    query[key] = value;
  });
  return query;
}

// Resolve a REQUEST path (locale prefix included) to its page. Returns the active
// locale plus the locale-free path, so the router reports Next's `asPath`.
function resolve(requestPath) {
  const { locale, pathname } = splitLocale(i18n, requestPath);
  const match = matchPath(pages, pathname);
  if (match) {
    return {
      Component: match.page.component,
      params: match.params,
      pattern: match.page.pattern,
      locale,
      pathname,
      notFound: false,
    };
  }
  return {
    Component: ErrorPage,
    params: {},
    pattern: pathname,
    locale,
    pathname,
    notFound: true,
  };
}

// A `router.push`/`replace` target may be a string or Next's URL object.
function toUrlString(value) {
  if (typeof value === "string") return value;
  return (value && value.pathname) || "/";
}

function Root() {
  const initial = useMemo(() => {
    const data = readNextData();
    const resolved = resolve(location.pathname);
    return {
      Component: resolved.Component,
      pageProps: resolved.notFound
        ? { statusCode: 404 }
        : data.props.pageProps,
      pattern: resolved.pattern,
      params: resolved.params,
      query: { ...data.query, ...parseQuery(location.search) },
      pathname: resolved.pathname,
      // Next's `asPath` is the URL as DISPLAYED (the `as` argument), query string
      // included, locale prefix excluded. Held in state rather than read back off
      // `location` so a shallow navigation's re-render sees the new value.
      asPath: resolved.pathname + location.search,
      locale: resolved.locale,
    };
  }, []);

  const [state, setState] = useState(initial);
  const stateRef = useRef(state);
  stateRef.current = state;
  const events = useMemo(createRouterEvents, []);

  const load = useCallback(
    async (rawUrl, rawAs, rawOptions, mode) => {
      const url = toUrlString(rawUrl);
      // `as` is the URL the browser shows and what the route resolves from; it
      // defaults to `url` (the `push(href)` one-argument form).
      const as = rawAs === undefined || rawAs === null ? url : toUrlString(rawAs);
      const options = rawOptions || {};
      const [pathPart, search = ""] = as.split("?");
      const path = pathPart || "/";
      const resolved = resolve(path);
      const current = stateRef.current;
      // Next's rule: "shallow routing only works for URL changes in the same
      // page". A shallow push that resolves to a DIFFERENT page is downgraded to
      // a normal navigation — Next does the same, and staying quiet about it
      // would make a page that silently re-fetched impossible to explain.
      const shallow = Boolean(options.shallow) && resolved.pattern === current.pattern;
      if (options.shallow && !shallow) {
        console.warn(
          `next/router: shallow navigation to ${as} left the current page ` +
            `(${current.pattern} -> ${resolved.pattern}); shallow routing only applies ` +
            "within one page, so the target page's data fetching will run.",
        );
      }
      events.emit("routeChangeStart", as, { shallow });
      let pageProps = {};
      if (shallow) {
        // The whole point of a shallow navigation: NO data fetch. The page keeps
        // the props it already has and only sees the new query/asPath.
        pageProps = current.pageProps;
      } else if (resolved.notFound) {
        pageProps = { statusCode: 404 };
      } else {
        const separator = search ? "&" : "";
        const dataUrl = `${path}?${search}${separator}__nextDataReq=1`;
        try {
          const response = await fetch(dataUrl, {
            headers: { "x-nextjs-data": "1" },
          });
          const json = await response.json();
          pageProps = json.pageProps || {};
        } catch {
          window.location.href = as;
          return;
        }
      }
      if (mode === "push") history.pushState({ __diffpack: true }, "", as);
      else if (mode === "replace") history.replaceState({ __diffpack: true }, "", as);
      setState({
        Component: resolved.Component,
        pageProps,
        pattern: resolved.pattern,
        params: resolved.params,
        query: { ...parseQuery(search), ...resolved.params },
        pathname: resolved.pathname,
        asPath: resolved.pathname + (search ? `?${search}` : ""),
        locale: resolved.locale,
      });
      // Next: `options.scroll ?? !isValidShallowRoute` — a shallow URL change
      // stays where the reader is; a real navigation goes to the top unless the
      // caller (or `<Link scroll={false}>`) opted out.
      const scroll = options.scroll === undefined ? !shallow : options.scroll;
      if (mode !== "pop" && scroll) window.scrollTo(0, 0);
      events.emit("routeChangeComplete", as, { shallow });
    },
    [events],
  );

  useEffect(() => {
    const onPopState = () => {
      load(location.pathname + location.search, undefined, undefined, "pop");
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, [load]);

  const router = useMemo(
    () => ({
      get pathname() {
        return stateRef.current.pattern;
      },
      get route() {
        return stateRef.current.pattern;
      },
      get asPath() {
        return stateRef.current.asPath;
      },
      get query() {
        return { ...stateRef.current.query, ...stateRef.current.params };
      },
      basePath: "",
      get locale() {
        return i18n ? stateRef.current.locale : undefined;
      },
      locales: i18n ? i18n.locales : undefined,
      defaultLocale: i18n ? i18n.defaultLocale : undefined,
      isReady: true,
      isFallback: false,
      isPreview: false,
      push: (url, as, options) => load(url, as, options, "push"),
      replace: (url, as, options) => load(url, as, options, "replace"),
      back: () => window.history.back(),
      forward: () => window.history.forward(),
      reload: () => window.location.reload(),
      prefetch: () => Promise.resolve(),
      beforePopState: () => {},
      events,
    }),
    [load, events],
  );

  const Component = state.Component;
  return (
    <RouterContext.Provider value={router}>
      <HeadManagerContext.Provider value={null}>
        <App Component={Component} pageProps={state.pageProps} />
        {/* Next renders the route announcer as a sibling of the app, portalled
            into <body>. It renders null until its effect runs, so the markup
            hydration sees is exactly the server's. */}
        <RouteAnnouncerPortal />
      </HeadManagerContext.Provider>
    </RouterContext.Provider>
  );
}

const container = document.getElementById("__next");
if (!container) {
  throw new Error(
    "diffpack pages-router: #__next mount point not found; the document did not render <Main/>",
  );
}
hydrateRoot(container, <Root />);
