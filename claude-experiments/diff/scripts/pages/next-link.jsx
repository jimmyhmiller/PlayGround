// `next/link` shim (pages-router). Renders an <a> whose plain left-click is
// intercepted and turned into a client-side `router.push`/`router.replace`, so
// navigation re-runs the target page's data fetching and swaps the view without a
// full document load. Modified clicks, non-self targets, and the no-router case
// (defensive) fall back to native navigation.
//
// The rendered `href` follows Next's own resolution, which is NOT "stringify the
// prop":
//   * an OBJECT `href` ({ pathname, query, hash }) is formatted, and dynamic route
//     segments in `pathname` are INTERPOLATED from `query` — the keys consumed that
//     way are removed from the query string and the leftovers are appended
//     (`{ pathname: "/users/[id]", query: { id: 7, tab: "a" } }` -> `/users/7?tab=a`);
//   * an explicit `as` prop is the *displayed* URL and wins over `href` (the classic
//     `<Link href="/users/[id]" as="/users/7">` form);
//   * with built-in i18n configured, the URL is locale-prefixed exactly as Next's
//     `addLocale` does (no prefix for the default locale; `locale={false}` opts out).

import { useContext } from "react";
import { RouterContext } from "./next-router.jsx";
import { addLocale, resolveHref } from "./pages-url.js";

export default function Link(props) {
  const {
    href,
    as: asProp,
    children,
    replace,
    scroll,
    shallow,
    prefetch: _prefetch,
    passHref: _passHref,
    legacyBehavior: _legacyBehavior,
    locale,
    onClick,
    ...rest
  } = props;
  const router = useContext(RouterContext);
  const resolvedHref = resolveHref(href);
  // `as` is the displayed URL when present; otherwise the resolved href is.
  const displayed =
    asProp !== undefined && asProp !== null ? resolveHref(asProp) : resolvedHref;
  // Locale prefixing only applies when the app configured built-in i18n (the router
  // then carries `locales`). `locale={false}` opts a link out.
  const i18nEnabled = Boolean(router && router.locales && router.locales.length);
  const targetLocale =
    locale === false ? null : locale !== undefined ? locale : router && router.locale;
  const url = i18nEnabled
    ? addLocale(displayed, targetLocale, router && router.defaultLocale)
    : displayed;

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
    // `shallow` and `scroll` are Link's pass-through navigation options, not
    // decoration: `scroll={false}` keeps the reader where they are and
    // `shallow` skips the target's data fetching.
    const options = { shallow, scroll };
    if (replace) router.replace(url, undefined, options);
    else router.push(url, undefined, options);
  };

  return (
    <a {...rest} href={url} onClick={handleClick}>
      {children}
    </a>
  );
}
