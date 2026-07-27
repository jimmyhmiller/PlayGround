// Build-time @vercel/og prerender runner (Node, build-time only). diffpack invokes this
// with the STANDALONE-ESM form of a code-based metadata-image generator (e.g. the
// transformed `opengraph-image.tsx`) and an output path. It imports the generator's
// default export, invokes it to obtain the `@vercel/og` `ImageResponse` (a subclass of
// the Web `Response`), reads the rendered image bytes, and writes them to disk. The
// generator resolves `react/jsx-runtime`, `@vercel/og`/`next/og` etc. through Node's own
// module resolution from the app's node_modules (this file lives inside the app tree).
//
// A build-time step legitimately uses `node:fs` — it never runs on the request path. Any
// failure exits non-zero with a clear message diffpack surfaces as the build error.
import { pathToFileURL } from "node:url";
import { writeFileSync } from "node:fs";

const [genPath, outPath] = process.argv.slice(2);
if (!genPath || !outPath) {
  console.error("og-prerender: usage: node og-prerender.mjs <generator.mjs> <out.png>");
  process.exit(2);
}

const mod = await import(pathToFileURL(genPath).href);
const fn = mod.default;
if (typeof fn !== "function") {
  console.error(
    `og-prerender: ${genPath} has no default-export function returning an ImageResponse`,
  );
  process.exit(1);
}

// The base (non-partitioned) convention: the default export is called with no params.
const res = await fn();
if (!res || typeof res.arrayBuffer !== "function") {
  console.error(
    `og-prerender: ${genPath} default export did not return an ImageResponse/Response ` +
      `(got ${res === null ? "null" : typeof res})`,
  );
  process.exit(1);
}

const bytes = Buffer.from(await res.arrayBuffer());
if (bytes.length === 0) {
  console.error(`og-prerender: ${genPath} produced an empty image`);
  process.exit(1);
}
writeFileSync(outPath, bytes);
