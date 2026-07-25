// `export const dynamic = "force-dynamic"` opts the route out of static prerender →
// classified `dynamic` (reason: "force-dynamic"). Rendered per request.
export const dynamic = "force-dynamic";

export default function Live() {
  return (
    <main id="live">
      <h1>live</h1>
      <p>rendered per request (force-dynamic)</p>
    </main>
  );
}
