// The index route ("/"). A plain Server Component with no request-scoped reads and
// no dynamic segment → classified `static` (one prerendered index.html).
import Link from "next/link";
import { POSTS } from "./blog/posts";

export default function Home() {
  return (
    <main id="home">
      <h1>blog-static index</h1>
      <ul>
        {POSTS.map((p) => (
          <li key={p.slug}>
            <Link href={`/blog/${p.slug}`}>{p.title}</Link>
          </li>
        ))}
      </ul>
    </main>
  );
}
