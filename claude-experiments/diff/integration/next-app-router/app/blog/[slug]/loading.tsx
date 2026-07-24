// The `loading.tsx` convention for `/blog/[slug]`. The adapter wraps the page in a
// <Suspense fallback={<Loading/>}>; because the SSR uses onAllReady (waits for all
// Suspense), the fallback is not in the final static HTML — true fallback-in-HTML is
// streaming (a later slice). Gated structurally (Suspense composed) + non-breaking.
export default function Loading() {
  return <main id="post-loading">Loading post…</main>;
}
