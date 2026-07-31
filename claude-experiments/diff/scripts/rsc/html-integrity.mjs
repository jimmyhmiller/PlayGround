// Structural checks on a raw served document, independent of any browser (a browser's
// parser RECOVERS from the corruption these look for, so a DOM probe can miss it).
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

// A `<script` may never appear between a `<` that opens a tag and its matching `>`.
// Streaming SSR that interleaves bytes at an arbitrary write boundary produces exactly
// that — e.g. `src="/vercel.s<script>...</script>vg"`. Returns the offending excerpts.
export function scriptsInsideTags(html) {
  const bad = [];
  let i = 0;
  while (i < html.length) {
    const lt = html.indexOf("<", i);
    if (lt < 0) break;
    const gt = html.indexOf(">", lt + 1);
    if (gt < 0) break;
    if (html.slice(lt + 1, gt).includes("<script")) {
      bad.push(html.slice(Math.max(0, lt - 60), gt + 20));
    }
    i = gt + 1;
  }
  return bad;
}

export default scriptsInsideTags;

// CLI: `node html-integrity.mjs <file>` — exit 1 and print the offending excerpts.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const file = process.argv[2];
  if (!file) {
    console.error("html-integrity: usage: node html-integrity.mjs <document.html>");
    process.exit(2);
  }
  const bad = scriptsInsideTags(readFileSync(file, "utf8"));
  if (bad.length) {
    console.error(`html-integrity: ${bad.length} <script> inside an open tag in ${file}:`);
    for (const excerpt of bad.slice(0, 3)) console.error(`  ...${excerpt}...`);
    process.exit(1);
  }
}
