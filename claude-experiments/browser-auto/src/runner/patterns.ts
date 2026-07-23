/**
 * Path patterns for `expect request` / `given stub` / `expect url`:
 * literal segments, `*` matches exactly one segment, `**` matches any suffix.
 * Query strings: a pattern without `?` ignores the URL's query; a pattern
 * with `?` must match it exactly.
 */
export function matchPath(pattern: string, urlPath: string, urlQuery = ""): boolean {
  let pPath = pattern;
  let pQuery: string | null = null;
  const qIdx = pattern.indexOf("?");
  if (qIdx >= 0) {
    pPath = pattern.slice(0, qIdx);
    pQuery = pattern.slice(qIdx + 1);
  }
  if (pQuery !== null && pQuery !== urlQuery.replace(/^\?/, "")) return false;

  const pSegs = pPath.split("/").filter((s) => s !== "");
  const uSegs = urlPath.split("/").filter((s) => s !== "");
  return matchSegs(pSegs, uSegs);
}

function matchSegs(p: string[], u: string[]): boolean {
  if (p.length === 0) return u.length === 0;
  const [head, ...rest] = p;
  if (head === "**") {
    if (rest.length === 0) return true;
    for (let skip = 0; skip <= u.length; skip++) {
      if (matchSegs(rest, u.slice(skip))) return true;
    }
    return false;
  }
  if (u.length === 0) return false;
  if (head !== "*" && head !== u[0]) return false;
  return matchSegs(rest, u.slice(1));
}

export function pathOf(url: string): { path: string; query: string } {
  const u = new URL(url);
  return { path: u.pathname, query: u.search.replace(/^\?/, "") };
}
