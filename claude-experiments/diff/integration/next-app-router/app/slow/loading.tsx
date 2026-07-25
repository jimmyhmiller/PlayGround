// The Suspense fallback for /slow. The app-router adapter composes a
// <Suspense fallback={<Loading/>}> around the page, so this HTML is what streams in
// the initial shell while the slow Server Component is still resolving.
export default function Loading() {
  return <p id="slow-loading">Loading slow data…</p>;
}
