// An OPTIONAL catch-all route `[[...slug]]` served from the LOCAL docs map. With
// `generateStaticParams` and no request read → classified `ssg`: the adapter
// enumerates the concrete slug arrays at build and prerenders one page each. Matches
// any depth (/intro, /guide/setup, …); "/" is served by the sibling app/page.tsx.
import { DOCS, findDoc } from "../docs";

export function generateStaticParams() {
  return DOCS.filter((d) => d.path.length > 0).map((d) => ({ slug: d.path }));
}

export default async function DocPage({
  params,
}: {
  params: Promise<{ slug?: string[] }>;
}) {
  const { slug } = await params;
  const doc = findDoc(slug);
  return (
    <main id="doc">
      <h1>doc: {(slug ?? []).join("/") || "(root)"}</h1>
      <p>{doc ? doc.body : "unknown doc"}</p>
    </main>
  );
}
