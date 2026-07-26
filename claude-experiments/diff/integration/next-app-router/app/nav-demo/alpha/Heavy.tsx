// A `"use client"` component loaded via next/dynamic(ssr:false). It renders only after
// the client mounts, so the SSR HTML shows the loading fallback and the hydrated client
// swaps in this content with no hydration mismatch.
"use client";

export default function Heavy() {
  return <div id="heavy">heavy-loaded-client-only</div>;
}
