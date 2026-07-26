"use client";
// `next/link` shim (diffpack next app-router adapter). A `"use client"` intercepting
// component: it renders the same server-reachable `<a href>`, but on the browser a
// plain left-click on an internal href is intercepted and handed to the client
// Router (`window.__diffpack_navigate`), which fetches the target route's flight
// (`?__rsc=1`) and diff-renders it WITHOUT a full document load. Modified clicks
// (meta/ctrl/shift/alt or a non-primary button), external/non-string hrefs, an
// already-`defaultPrevented` event, or the pre-hydration window (no
// `__diffpack_navigate`) all fall through to a real navigation — no `preventDefault`.
import { createElement } from "react";

export default function Link(props) {
  const { href, children, prefetch, replace, scroll, shallow, locale, onClick, ...rest } = props;
  const resolved = typeof href === "string" ? href : (href && href.pathname) || "#";
  function handleClick(event) {
    if (onClick) onClick(event);
    if (event.defaultPrevented) return;
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    if (typeof href !== "string" || !href.startsWith("/")) return;
    if (typeof window === "undefined" || typeof window.__diffpack_navigate !== "function") return;
    event.preventDefault();
    window.__diffpack_navigate(resolved, { replace: !!replace });
  }
  return createElement("a", { href: resolved, onClick: handleClick, ...rest }, children);
}

// `useLinkStatus` (Next 15.3+/16): the pending state of an in-progress client
// navigation started by a parent `<Link>`. This adapter's soft-nav diff-renders the
// route synchronously via `__diffpack_navigate` and does not expose a per-link pending
// signal, so this returns the settled state — matching the common `{ pending }`
// destructure (a loading indicator simply never shows) rather than throwing on import.
export function useLinkStatus() {
  return { pending: false };
}
