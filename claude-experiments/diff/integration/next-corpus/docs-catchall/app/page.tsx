// The docs index ("/"). A plain Server Component (no request read, no dynamic segment)
// → classified `static`. Sits alongside the optional catch-all, which serves every
// deeper path.
import Link from "next/link";
import { DOCS } from "./docs";

export default function DocsHome() {
  return (
    <main id="docs-home">
      <h1>docs index</h1>
      <ul>
        {DOCS.filter((d) => d.path.length > 0).map((d) => (
          <li key={d.path.join("/")}>
            <Link href={`/${d.path.join("/")}`}>{d.title}</Link>
          </li>
        ))}
      </ul>
    </main>
  );
}
