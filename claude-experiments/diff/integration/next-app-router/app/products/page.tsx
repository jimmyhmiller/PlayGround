// A STATIC app-router route whose data is wrapped in `unstable_cache` (next/cache) and
// tagged `products`. It reads NO request-scoped state, so diffpack (like `next build`)
// prerenders it to a static `.html` + `.rsc` and serves it straight from cache (HIT). The
// `products` tag is captured at prerender time, so a later `revalidateTag("products")` (in
// the server action or the /api/revalidate route handler) marks THIS page stale — the next
// request serves STALE and the background regen recomputes the cached timestamp. `next
// build` accepts this unchanged (unstable_cache is a real next/cache export).
import { unstable_cache } from "next/cache";

const loadStamp = unstable_cache(
  async () => Date.now(),
  ["products-stamp"],
  { tags: ["products"] },
);

export default async function ProductsPage() {
  const stamp = await loadStamp();
  return (
    <main>
      <h1 id="products-heading">Products</h1>
      <p id="products-stamp">generated-at: {stamp}</p>
    </main>
  );
}
