// A route with a slow async Server Component behind a Suspense boundary (the sibling
// loading.tsx supplies the fallback the adapter wraps around this page). It proves
// STREAMING SSR: the shell + fallback reach the browser immediately (fast TTFB), and
// the slow content streams in ~500ms later — without blocking the first byte.
//
// force-dynamic so BOTH toolchains render per-request (otherwise Next static-prerenders
// this at build time — no dynamic data — and serves it instantly, which wouldn't
// exercise streaming at all).
export const dynamic = "force-dynamic";

async function SlowData() {
  await new Promise((resolve) => setTimeout(resolve, 500));
  return <p id="slow-content">Slow data resolved after 500ms on the server.</p>;
}

export default function SlowPage() {
  return (
    <main>
      <h1 id="slow-heading">Streaming SSR demo</h1>
      <SlowData />
    </main>
  );
}
