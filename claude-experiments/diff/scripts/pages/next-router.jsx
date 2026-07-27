// `next/router` shim (pages-router). Exposes the RouterContext both the client
// hydration entry (which provides a real navigating router) and the server render
// entry (which provides a read-only router) fill, plus the `useRouter` hook pages
// call. `RouterContext` is shared by identity: pages import it through the
// `next/router` alias and the entries import this same file, so it is one context.

import { createContext, useContext } from "react";

export const RouterContext = createContext(null);

export function useRouter() {
  const router = useContext(RouterContext);
  if (!router) {
    throw new Error(
      "next/router: useRouter() was called outside a RouterContext. In diffpack's " +
        "pages-router this means the component rendered outside the App tree.",
    );
  }
  return router;
}

// A tiny router events emitter (routeChangeStart / routeChangeComplete / ...).
export function createRouterEvents() {
  const listeners = Object.create(null);
  return {
    on(type, handler) {
      (listeners[type] || (listeners[type] = [])).push(handler);
    },
    off(type, handler) {
      const list = listeners[type];
      if (!list) return;
      const index = list.indexOf(handler);
      if (index >= 0) list.splice(index, 1);
    },
    emit(type, ...args) {
      const list = listeners[type];
      if (list) for (const handler of list.slice()) handler(...args);
    },
  };
}

// `next/router` also has a default export (the `Router` singleton in Next). Pages
// that `import Router from "next/router"` get an object exposing the hook.
export default { useRouter, RouterContext, createRouterEvents };
