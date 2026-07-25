// Consumes the `searchParams` prop → classified `dynamic` (reading searchParams at
// the top opts the whole route into per-request rendering).
export default async function Search({
  searchParams,
}: {
  searchParams: Promise<{ q?: string }>;
}) {
  const { q } = await searchParams;
  return (
    <main id="search">
      <h1>search</h1>
      <p id="query">query: {q ?? "none"}</p>
    </main>
  );
}
