// The pages-router SSG prerenderer — build-time static generation for a
// diffpack-built Next pages-router app. It imports the already-bundled SSR graph
// (`server/server.mjs`, which runs the app's OWN bundled React — the explicitly
// allowed oracle) and calls its `prerender()` export, which runs `getStaticProps`
// (and `getStaticPaths` for dynamic routes) for every SSG page ONCE, warm and
// in-process (no per-page child spawn), and writes the resulting props manifest to
// `<outputDir>/prerender.json`. The live orchestrator seeds its ISR cache from that
// file so static pages are answered with zero per-request data fetch; pages with
// `revalidate` regenerate on expiry. The bundling of every graph stays native Rust.
//
//   usage: node pages-prerender.mjs <.diffpack-output dir>

import { existsSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

function die(message) {
  console.error(`pages-prerender: ${message}`);
  process.exit(1);
}

const outputDir = process.argv[2];
if (!outputDir) die("usage: node pages-prerender.mjs <.diffpack-output dir>");

const ssrEntry = join(outputDir, "server", "server.mjs");
if (!existsSync(ssrEntry)) {
  die(`SSR bundle not found at ${ssrEntry} — run the ssr build first`);
}

const ssrModule = await import(pathToFileURL(ssrEntry).href);
const prerender =
  ssrModule.prerender || (ssrModule.default && ssrModule.default.prerender);
if (typeof prerender !== "function") {
  die("the SSR bundle does not export prerender");
}

const data = await prerender();
const entries = (data && data.entries) || [];
writeFileSync(join(outputDir, "prerender.json"), JSON.stringify({ entries }));
console.log(
  `pages-prerender: prerendered ${entries.length} static entr${entries.length === 1 ? "y" : "ies"} -> prerender.json`,
);
for (const entry of entries) {
  const rev = entry.revalidate != null ? ` (revalidate ${entry.revalidate}s)` : "";
  console.log(`  ${entry.url}${rev}`);
}
