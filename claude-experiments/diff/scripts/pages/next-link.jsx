// `next/link` shim (pages-router). Renders an <a> whose plain left-click is
// intercepted and turned into a client-side `router.push`/`router.replace`, so
// navigation re-runs the target page's data fetching and swaps the view without a
// full document load. Modified clicks, non-self targets, and the no-router case
// (defensive) fall back to native navigation.

import { useContext } from "react";
import { RouterContext } from "./next-router.jsx";

function hrefToString(href) {
  if (typeof href === "string") return href;
  if (href && typeof href === "object") {
    const path = href.pathname || "";
    const query = href.query
      ? "?" + new URLSearchParams(href.query).toString()
      : "";
    const hash = href.hash || "";
    return path + query + hash;
  }
  return "#";
}

export default function Link(props) {
  const {
    href,
    as: _as,
    children,
    replace,
    scroll: _scroll,
    shallow: _shallow,
    prefetch: _prefetch,
    passHref: _passHref,
    legacyBehavior: _legacyBehavior,
    locale: _locale,
    onClick,
    ...rest
  } = props;
  const router = useContext(RouterContext);
  const url = hrefToString(href);

  const handleClick = (event) => {
    if (onClick) onClick(event);
    if (event.defaultPrevented) return;
    if (
      event.button !== 0 ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey
    ) {
      return;
    }
    if (rest.target && rest.target !== "_self") return;
    if (!router) return; // no context: let the browser navigate natively
    event.preventDefault();
    if (replace) router.replace(url);
    else router.push(url);
  };

  return (
    <a {...rest} href={url} onClick={handleClick}>
      {children}
    </a>
  );
}
