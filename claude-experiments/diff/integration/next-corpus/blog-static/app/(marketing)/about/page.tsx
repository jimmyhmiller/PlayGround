// A route inside a route group `(marketing)` — the group contributes NO URL segment,
// so this route's path is `/about` (its layout, if any, still applies). No dynamic
// segment, no request read → classified `static`.
export default function About() {
  return (
    <main id="about">
      <h1>about (marketing group)</h1>
      <p>static route inside a route group</p>
    </main>
  );
}
