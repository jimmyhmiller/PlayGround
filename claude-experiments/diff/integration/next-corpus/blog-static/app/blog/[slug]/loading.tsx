// The `loading.tsx` convention: the adapter composes a <Suspense fallback> around the
// blog route (a `loading: M<i>` level in the generated ROUTES table).
export default function Loading() {
  return <main id="post-loading">loading post…</main>;
}
