// A Server Component that throws during render, to exercise the `error.tsx`
// boundary convention. Because the adapter wraps this page in the generated client
// ErrorBoundary, the flight render treats the throw as a recoverable client-boundary
// subtree (the child exits 0), and the SSR/browser React catches it and renders the
// sibling `error.tsx` fallback — the render never crashes.
//
// `force-dynamic` opts this route out of Next's static prerender (SSG): a page that
// unconditionally throws cannot be statically exported (the error boundary rescues at
// REQUEST time, not at build time), so without this `next build` would fail collecting
// this route. It is rendered on-demand — exactly diffpack's per-request model.
export const dynamic = "force-dynamic";

export default function ErrorDemo() {
  throw new Error("boom-from-server");
}
