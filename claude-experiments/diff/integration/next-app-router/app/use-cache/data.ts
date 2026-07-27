"use cache";

// A `"use cache"` data module (Next's Dynamic-IO cache directive). Every export is a
// cached async function: diffpack's react-server transform wraps each in a cache boundary
// (__diffpackUseCache) that memoizes the return keyed by arguments and runs the body inside
// a cacheTag()/cacheLife() collection scope. `cacheTag("use-cache-demo")` registers the
// cached value (and the page that reads it) under that tag, so a later
// `revalidateTag("use-cache-demo")` busts the prerendered /use-cache page; `cacheLife("hours")`
// gives the memo a soft TTL. `next build` accepts this unchanged ("use cache" + cacheTag +
// cacheLife are real next/cache exports). The rendered timestamp lets a test observe that a
// revalidateTag marks the page stale (the background regen recomputes it).
import { cacheTag, cacheLife } from "next/cache";

export async function getStamp(): Promise<number> {
  cacheTag("use-cache-demo");
  cacheLife("hours");
  return Date.now();
}
