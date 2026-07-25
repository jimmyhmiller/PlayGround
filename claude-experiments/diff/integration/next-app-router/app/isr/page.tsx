// ISR (Incremental Static Regeneration). `revalidate = 2` classifies this route as
// isr: it is prerendered at build time, served from the cache on every request, and
// regenerated in the background once the cached copy is older than 2 seconds. The
// rendered timestamp lets a test observe the regeneration: it stays fixed while fresh,
// then advances after a stale request triggers a rebuild.
export const revalidate = 2;

export default function IsrPage() {
  return (
    <main>
      <h1 id="isr-heading">ISR demo</h1>
      <p id="isr-value">generated-at: {Date.now()}</p>
    </main>
  );
}
