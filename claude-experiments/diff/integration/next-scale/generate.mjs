// Generates N static app-router pages (app/p/<i>/page.tsx) for the scale benchmark.
// Committed scaffolding stays; the generated pages are gitignored.
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join, dirname } from "node:path";
const N = Number(process.argv[2] || 3000);
const root = dirname(new URL(import.meta.url).pathname);
rmSync(join(root, "app/p"), { recursive: true, force: true });
for (let i = 0; i < N; i++) {
  const dir = join(root, "app/p", String(i));
  mkdirSync(dir, { recursive: true });
  const prev = i > 0 ? i - 1 : N - 1, next = (i + 1) % N;
  writeFileSync(join(dir, "page.tsx"),
`import Link from "next/link";
export default function Page() {
  return (<main><h1 id="p${i}">Page ${i}</h1><p>Content for page number ${i}. Lorem ipsum dolor sit amet.</p><nav><Link href="/p/${prev}">prev</Link> <Link href="/p/${next}">next</Link></nav></main>);
}
`);
}
console.log(`generated ${N} pages under app/p/`);
