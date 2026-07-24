// The app-root `not-found.tsx` convention: the body the adapter renders (wrapped in
// the root layout) for a genuinely-unknown path — a real HTTP 404, not a fall-through
// to the index route.
export default function NotFound() {
  return <main id="not-found">404 — page not found</main>;
}
