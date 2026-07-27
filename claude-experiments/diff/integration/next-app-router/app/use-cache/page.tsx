// A STATIC app-router route whose data comes from a `"use cache"` module (./data). It reads
// NO request-scoped state, so diffpack (like `next build`) prerenders it to a static `.html`
// + `.rsc` and serves it from cache (HIT). The cached getStamp() tags itself `use-cache-demo`
// via cacheTag(), so that tag is captured at prerender time and a later
// `revalidateTag("use-cache-demo")` (in the /api/revalidate route handler or a server action)
// marks THIS page stale — the next request serves STALE and the background regen recomputes
// the timestamp. `next build` accepts this unchanged.
import { getStamp } from "./data";

export default async function UseCachePage() {
  const stamp = await getStamp();
  return (
    <main>
      <h1 id="use-cache-heading">use cache demo</h1>
      <p id="use-cache-stamp">generated-at: {stamp}</p>
    </main>
  );
}
