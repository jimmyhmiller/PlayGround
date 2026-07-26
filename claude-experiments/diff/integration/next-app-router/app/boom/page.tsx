// A page that throws during render with NO nearer error.tsx boundary, so the throw
// propagates to the app-root global-error boundary — proving global-error catches a
// document-level error and renders its own <html>. Marked force-dynamic so the build
// prerender phase serves it per-request (never baking a thrown page into the cache).
export const dynamic = "force-dynamic";

export default function Boom() {
  throw new Error("boom from /boom");
}
