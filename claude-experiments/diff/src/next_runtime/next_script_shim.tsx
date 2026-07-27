"use client";
// `next/script` shim (diffpack next app-router adapter).
//
// Next's own `next/script` barrel is CommonJS (`module.exports = require("./dist/client/script")`)
// and its implementation talks to Next-internal singletons (HeadManagerContext, the
// `self.__next_s` beforeInteractive runtime, Partytown). Diffpack aliases the specifier to
// THIS module instead, exactly as it does for `next/link` and `next/image`, so the component
// is a first-class `"use client"` island of the app's own graph.
//
// Behaviour, matched against `next start` on the same app:
//   * `afterInteractive` (the default) — server render emits NOTHING into the tree and calls
//     `ReactDOM.preload(src, { as: "script", nonce, … })`, so the document head carries
//     `<link rel="preload" as="script" href=… nonce=…>`; after mount the real `<script>` is
//     appended to `document.body`.
//   * `lazyOnload` — same, but the append waits for `window.load` and then an idle callback.
//   * `beforeInteractive` — the script must run before hydration, so it is preloaded AND
//     rendered as a real `<script>` element in place.
//   * `worker` (Partytown) — NOT implemented; it throws, naming the prop, rather than
//     silently downgrading to a normal script.
import * as React from "react";
import * as ReactDOMNamespace from "react-dom";

const ReactDOM = (ReactDOMNamespace as any).default || ReactDOMNamespace;
const { useEffect, useRef, createElement } = React;

// One entry per script that has been requested (keyed by `id || src`), so a remount does
// not inject the same `<script>` twice — Next keeps the same cache for the same reason.
const LoadCache = new Set<string>();
const ScriptCache = new Map<string, Promise<unknown>>();

function requestIdle(callback: () => void) {
  if (typeof window !== "undefined" && typeof (window as any).requestIdleCallback === "function") {
    (window as any).requestIdleCallback(callback);
    return;
  }
  setTimeout(callback, 0);
}

function preloadOptions(props: any) {
  const options: Record<string, unknown> = { as: "script" };
  if (props.nonce) options.nonce = props.nonce;
  if (props.crossOrigin) options.crossOrigin = props.crossOrigin;
  if (props.integrity) options.integrity = props.integrity;
  if (props.fetchPriority) options.fetchPriority = props.fetchPriority;
  return options;
}

function preinitStylesheets(stylesheets: string[] | undefined) {
  if (!stylesheets || !stylesheets.length) return;
  if (typeof ReactDOM.preinit !== "function") {
    throw new Error(
      'diffpack next/script: the "stylesheets" prop needs ReactDOM.preinit, which this ' +
        "react-dom build does not provide (react-dom >= 18.3 is required).",
    );
  }
  for (const stylesheet of stylesheets) ReactDOM.preinit(stylesheet, { as: "style" });
}

// Appends the real `<script>` to the document. Every attribute the caller passed through
// (`async`, `defer`, `type`, `crossOrigin`, `integrity`, `data-*`, …) is carried over, so a
// script diffpack injects is byte-comparable with the one Next injects.
function loadScript(props: any) {
  const {
    src = "",
    id,
    onLoad = () => {},
    onReady = null,
    onError,
    dangerouslySetInnerHTML,
    children = "",
    strategy,
    stylesheets,
    ...attributes
  } = props;
  const cacheKey = id || src;
  if (cacheKey && LoadCache.has(cacheKey)) return;
  if (cacheKey && ScriptCache.has(cacheKey)) {
    ScriptCache.get(cacheKey)!.then(onLoad, onError).then(() => {
      if (onReady) onReady();
    });
    return;
  }

  const element = document.createElement("script");
  const loaded = new Promise<void>((resolve, reject) => {
    element.addEventListener("load", function (this: HTMLScriptElement, event: Event) {
      resolve();
      if (onLoad) onLoad.call(this, event);
      if (onReady) onReady();
    });
    element.addEventListener("error", function (this: HTMLScriptElement, event: Event) {
      reject(event);
    });
  }).catch(function (event) {
    if (onError) onError(event);
  });

  if (dangerouslySetInnerHTML) {
    element.innerHTML = dangerouslySetInnerHTML.__html || "";
  } else if (children) {
    element.textContent = typeof children === "string" ? children : Array.isArray(children) ? children.join("") : "";
  } else if (src) {
    element.src = src;
  }
  if (id) element.id = id;
  for (const [key, value] of Object.entries(attributes)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "children" || key === "dangerouslySetInnerHTML") continue;
    const attribute = key === "className" ? "class" : key === "crossOrigin" ? "crossorigin" : key.toLowerCase();
    element.setAttribute(attribute, value === true ? "" : String(value));
  }
  element.setAttribute("data-nscript", props.strategy || "afterInteractive");

  if (cacheKey) {
    LoadCache.add(cacheKey);
    ScriptCache.set(cacheKey, loaded);
  }
  preinitStylesheets(stylesheets);
  document.body.appendChild(element);
}

function Script(props: any) {
  const { src = "", strategy = "afterInteractive", stylesheets } = props;

  if (strategy === "worker") {
    throw new Error(
      'diffpack next/script: strategy="worker" (Partytown) is not implemented. ' +
        "Remove the prop or use afterInteractive/lazyOnload/beforeInteractive.",
    );
  }
  if (
    strategy !== "afterInteractive" &&
    strategy !== "lazyOnload" &&
    strategy !== "beforeInteractive"
  ) {
    throw new Error(
      `diffpack next/script: unknown strategy ${JSON.stringify(strategy)}; expected ` +
        '"beforeInteractive", "afterInteractive" or "lazyOnload".',
    );
  }

  const injected = useRef(false);
  useEffect(() => {
    if (injected.current) return;
    injected.current = true;
    if (strategy === "afterInteractive") {
      loadScript(props);
    } else if (strategy === "lazyOnload") {
      if (document.readyState === "complete") {
        requestIdle(() => loadScript(props));
      } else {
        window.addEventListener("load", () => requestIdle(() => loadScript(props)));
      }
    }
  }, [props, strategy]);

  preinitStylesheets(stylesheets);

  // React Float: the document head carries the preload so the browser starts fetching the
  // script with the shell, before hydration runs the injection above. This is the ONLY
  // thing `afterInteractive` contributes to the server-rendered document.
  if (src && (strategy === "afterInteractive" || strategy === "beforeInteractive")) {
    if (typeof ReactDOM.preload !== "function") {
      throw new Error(
        "diffpack next/script: ReactDOM.preload is missing (react-dom >= 18.3 is required).",
      );
    }
    ReactDOM.preload(src, preloadOptions(props));
  }

  // `beforeInteractive` must execute before hydration, so it is part of the document.
  if (strategy === "beforeInteractive") {
    const { strategy: _s, stylesheets: _ss, onLoad: _ol, onReady: _or, onError: _oe, ...rest } = props;
    return createElement("script", { ...rest, "data-nscript": "beforeInteractive" });
  }
  return null;
}

Object.defineProperty(Script, "__nextScript", { value: true });

export default Script;
