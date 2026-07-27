// diffpack pages-router CLIENT hydration entry (bundled to `public/client.js`).
//
// Reads the server-injected `__NEXT_DATA__`, resolves the current page from the
// generated route table, and hydrates `#__next` with `<App Component={Page}
// pageProps={...}/>` inside a real navigating RouterContext. `next/link` and
// `router.push/replace/back` navigate client-side: they fetch the target page's
// props (`?__nextDataReq=1`, served by the same SSR handler) and swap the view.

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
import { App, ErrorPage, pages } from "./pages-manifest.client.js";

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

function resolve(pathname) {
  const match = matchPath(pages, pathname);
  if (match) {
    return {
      Component: match.page.component,
      params: match.params,
      pattern: match.page.pattern,
      notFound: false,
    };
  }
  return { Component: ErrorPage, params: {}, pattern: pathname, notFound: true };
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
      pathname: location.pathname,
    };
  }, []);

  const [state, setState] = useState(initial);
  const stateRef = useRef(state);
  stateRef.current = state;
  const events = useMemo(createRouterEvents, []);

  const load = useCallback(
    async (rawUrl, mode) => {
      const url = typeof rawUrl === "string" ? rawUrl : rawUrl.pathname || "/";
      const [pathPart, search = ""] = url.split("?");
      const path = pathPart || "/";
      const resolved = resolve(path);
      events.emit("routeChangeStart", url);
      let pageProps = {};
      if (resolved.notFound) {
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
          window.location.href = url;
          return;
        }
      }
      if (mode === "push") history.pushState({ __diffpack: true }, "", url);
      else if (mode === "replace") history.replaceState({ __diffpack: true }, "", url);
      setState({
        Component: resolved.Component,
        pageProps,
        pattern: resolved.pattern,
        params: resolved.params,
        query: { ...parseQuery(search), ...resolved.params },
        pathname: path,
      });
      if (mode !== "pop") window.scrollTo(0, 0);
      events.emit("routeChangeComplete", url);
    },
    [events],
  );

  useEffect(() => {
    const onPopState = () => {
      load(location.pathname + location.search, "pop");
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
        return stateRef.current.pathname + location.search;
      },
      get query() {
        return { ...stateRef.current.query, ...stateRef.current.params };
      },
      basePath: "",
      isReady: true,
      isFallback: false,
      isPreview: false,
      push: (url) => load(url, "push"),
      replace: (url) => load(url, "replace"),
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
