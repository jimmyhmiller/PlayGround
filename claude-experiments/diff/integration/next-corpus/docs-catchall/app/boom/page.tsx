// A Server Component that THROWS. Paired with `boom/error.tsx`, the adapter wraps it
// in the generated client ErrorBoundary (with a segment Suspense), so the render is
// contained. No config / request read → classified `static` (the throw is a runtime
// concern, not a classification one).
export default function Boom() {
  throw new Error("boom-from-server");
}
