// LOCAL docs tree (no fetch, no network). The optional catch-all route reads this map
// and `generateStaticParams` enumerates its keys at build time.
export interface Doc {
  path: string[];
  title: string;
  body: string;
}

export const DOCS: Doc[] = [
  { path: [], title: "Docs home", body: "welcome to the docs" },
  { path: ["intro"], title: "Intro", body: "getting started" },
  { path: ["guide", "setup"], title: "Setup guide", body: "how to set up" },
];

export function findDoc(slug: string[] | undefined): Doc | undefined {
  const parts = slug ?? [];
  return DOCS.find(
    (d) => d.path.length === parts.length && d.path.every((p, i) => p === parts[i]),
  );
}
