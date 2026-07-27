// diffpack pages-router shared runtime.
//
// Holds the React context the custom `_document` render reads (`DocumentContext`)
// and the pure route matcher both the client hydration entry and the server render
// entry use, so a request path maps to the same page + params on both sides.
//
// A "page" record (produced by the generated `pages-manifest.*.js`) has the shape
//   { pattern: string, regex: RegExp, keys: Array<{ name, catchall }> }
// where `regex` captures one group per key in order. A catch-all key's capture is a
// "/"-joined string that `matchPath` splits back into the array Next exposes.

import { createContext } from "react";

// The custom `_document` reads the rendered app HTML, the collected <head>
// elements, the serialized `__NEXT_DATA__`, and the client bundle URL from here.
export const DocumentContext = createContext(null);

function safeDecode(value) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

// Match `pathname` against the ordered `pages` table (most-specific first). Returns
// `{ page, params }` or `null`. Catch-all params are arrays, matching Next.
export function matchPath(pages, pathname) {
  const path = pathname.length > 1 && pathname.endsWith("/")
    ? pathname.slice(0, -1)
    : pathname;
  for (const page of pages) {
    const match = page.regex.exec(path);
    if (!match) continue;
    const params = {};
    page.keys.forEach((key, index) => {
      const raw = match[index + 1];
      if (raw === undefined || raw === "") return;
      params[key.name] = key.catchall
        ? raw.split("/").map(safeDecode)
        : safeDecode(raw);
    });
    return { page, params };
  }
  return null;
}
