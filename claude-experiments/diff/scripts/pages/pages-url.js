// diffpack pages-router URL + locale resolution — the PURE half of the runtime.
//
// Deliberately dependency-free (no react, no DOM, no node): `next/link`, the client
// hydration entry and the server render entry all import these, so there is exactly
// ONE definition of "what URL does this href mean" and "which locale is this path",
// and it can be exercised directly by a test.

// One dynamic token in a route pattern: `[[...rest]]`, `[...rest]`, or `[id]`.
const DYNAMIC_TOKEN = /\[\[\.\.\.([^\]]+)\]\]|\[\.\.\.([^\]]+)\]|\[([^\]/]+)\]/g;

function encodeSegment(value) {
  return Array.isArray(value)
    ? value.map((part) => encodeURIComponent(String(part))).join("/")
    : encodeURIComponent(String(value));
}

// Substitute `query` values into the dynamic segments of `pathname`, returning the
// concrete path plus the set of query keys the path consumed. A required segment with
// no matching query value is a hard error (Next throws the same way) — rendering the
// literal `[id]` would produce a dead link.
function interpolate(pathname, query) {
  const consumed = new Set();
  const result = pathname.replace(
    DYNAMIC_TOKEN,
    (_token, optionalCatchAll, catchAll, dynamic) => {
      const name = optionalCatchAll || catchAll || dynamic;
      const value = query ? query[name] : undefined;
      if (value === undefined || value === null || value === "") {
        if (optionalCatchAll) {
          consumed.add(name);
          return "";
        }
        throw new Error(
          `next/link: the href "${pathname}" is missing a value for the dynamic ` +
            `segment "${name}" — pass it in the href's \`query\` (or use \`as\`)`,
        );
      }
      consumed.add(name);
      return encodeSegment(value);
    },
  );
  // An optional catch-all that dropped out leaves `//` or a trailing `/`.
  const cleaned = result.replace(/\/{2,}/g, "/");
  return {
    pathname:
      cleaned.length > 1 && cleaned.endsWith("/") ? cleaned.slice(0, -1) : cleaned,
    consumed,
  };
}

function searchFrom(query, consumed) {
  if (!query) return "";
  const params = new URLSearchParams();
  for (const key of Object.keys(query)) {
    if (consumed.has(key)) continue;
    const value = query[key];
    if (value === undefined || value === null) continue;
    if (Array.isArray(value)) for (const item of value) params.append(key, String(item));
    else params.append(key, String(value));
  }
  const search = params.toString();
  return search ? "?" + search : "";
}

// Next's `resolveHref`: a string passes through; an OBJECT href
// (`{ pathname, query, hash }`) is formatted with its dynamic segments interpolated
// from `query`, the consumed keys removed and the leftovers appended
// (`{ pathname: "/users/[id]", query: { id: 7, tab: "a" } }` -> `/users/7?tab=a`).
export function resolveHref(href) {
  if (typeof href === "string") return href;
  if (href && typeof href === "object") {
    const raw = href.pathname || "";
    const { pathname, consumed } = interpolate(raw, href.query);
    const hash = href.hash
      ? href.hash.startsWith("#")
        ? href.hash
        : "#" + href.hash
      : "";
    return pathname + searchFrom(href.query, consumed) + hash;
  }
  return "#";
}

// --- built-in i18n (next.config `i18n`) -------------------------------------------
//
// Locale routing is pure path prefixing: the default locale is served UNPREFIXED,
// every other locale lives under `/<locale>`. The locale list / default locale are
// baked into the generated manifests as `i18n` (`null` when unconfigured).

// Next's `addLocale`: prefix a path with a locale, except for the default locale
// (served unprefixed) and paths already carrying the prefix.
export function addLocale(path, locale, defaultLocale) {
  if (!locale || locale === defaultLocale) return path;
  if (path === "/" + locale || path.startsWith("/" + locale + "/")) return path;
  if (!path.startsWith("/")) return path;
  return path === "/" ? "/" + locale : "/" + locale + path;
}

// Strip a leading `/<locale>` off `pathname`. Returns the active locale plus the
// locale-free path the route table is matched against (and that `router.asPath`
// reports). With no i18n configured the path passes through and the locale is null,
// so a plain app behaves exactly as before.
export function splitLocale(i18n, pathname) {
  if (!i18n) return { locale: null, pathname, prefixed: false };
  const segment = pathname.split("/")[1];
  if (segment && i18n.locales.includes(segment)) {
    const rest = pathname.slice(segment.length + 1);
    return { locale: segment, pathname: rest === "" ? "/" : rest, prefixed: true };
  }
  return { locale: i18n.defaultLocale, pathname, prefixed: false };
}

// Next's root-path locale detection: a `NEXT_LOCALE` cookie wins, otherwise the best
// `Accept-Language` match (exact tag first, then the primary subtag). Falls back to
// the default locale. Only consulted for the bare root — Next detects there and
// nowhere else.
export function detectLocale(i18n, headers, cookies) {
  if (!i18n) return null;
  if (!i18n.localeDetection) return i18n.defaultLocale;
  const cookie = cookies && cookies.NEXT_LOCALE;
  if (cookie && i18n.locales.includes(cookie)) return cookie;
  const header =
    (headers && (headers["accept-language"] || headers["Accept-Language"])) || "";
  const ranked = String(header)
    .split(",")
    .map((part) => {
      const [tag, ...params] = part.trim().split(";");
      const q = params
        .map((p) => p.trim())
        .filter((p) => p.startsWith("q="))
        .map((p) => Number(p.slice(2)))[0];
      return { tag: tag.trim(), q: Number.isFinite(q) ? q : 1 };
    })
    .filter((entry) => entry.tag)
    .sort((a, b) => b.q - a.q);
  for (const { tag } of ranked) {
    const lower = tag.toLowerCase();
    const exact = i18n.locales.find((locale) => locale.toLowerCase() === lower);
    if (exact) return exact;
    const base = lower.split("-")[0];
    const loose = i18n.locales.find(
      (locale) => locale.toLowerCase().split("-")[0] === base,
    );
    if (loose) return loose;
  }
  return i18n.defaultLocale;
}
