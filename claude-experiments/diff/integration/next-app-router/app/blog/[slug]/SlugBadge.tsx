"use client";

// A `"use client"` island under the dynamic `/blog/[slug]` route that reads the
// matched segment via `useParams()`. This exercises the app-router hooks context
// (`next/navigation` → `PathParamsContext`): the params the server matched are fed
// identically on SSR and client, so `useParams().slug` renders `hello` in the SSR
// HTML AND after hydration in the browser, with no hydration mismatch. `next build`
// accepts this unchanged.
import { useParams } from "next/navigation";

export function SlugBadge() {
  const params = useParams();
  return <span id="slug-badge">slug: {String(params.slug ?? "")}</span>;
}
