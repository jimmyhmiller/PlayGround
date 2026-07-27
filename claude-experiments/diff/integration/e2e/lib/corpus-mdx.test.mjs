// Corpus regression tests: `node --test integration/e2e/lib/corpus-mdx.test.mjs`
//
// FINDINGS #36. `next-mdx` and `next-pages-mdx` were in the corpus and green, and
// between them they exercised almost none of diffpack's MDX surface:
// `next-mdx/mdx-components.tsx` returns `const components: MDXComponents = {}`, so the
// override path was never observed; neither app writes a GFM construct, configures a
// remark/rehype plugin, or has frontmatter. A regression in any of it would have gone
// unnoticed while the suite reported 2/2.
//
// The fix was to add first-party fixtures that use those features for real. This file
// is what keeps them honest: it asserts the FIXTURE SOURCES still contain each feature,
// so nobody can quietly hollow one out (an empty override map, a table deleted) and
// leave a green run that proves nothing. It reads the checked-in fixtures under
// `integration/e2e/fixtures/`, never the gitignored `apps/` copies, so it needs no
// materialized corpus and no network.
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const corpus = JSON.parse(readFileSync(join(root, "corpus.json"), "utf8"));

const walk = (dir) => {
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === "node_modules") continue;
    const full = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(full));
    else if (statSync(full).isFile()) out.push(full);
  }
  return out;
};

const app = (id) => {
  const entry = corpus.apps.find((a) => a.id === id);
  assert.ok(entry, `${id} is missing from corpus.json`);
  assert.ok(entry.firstParty, `${id} must declare its first-party fixture path`);
  const dir = join(root, entry.firstParty);
  assert.ok(existsSync(dir), `${id}: fixture directory ${entry.firstParty} does not exist`);
  const files = walk(dir);
  const read = (suffix) =>
    files.filter((f) => f.endsWith(suffix)).map((f) => readFileSync(f, "utf8"));
  return { entry, dir, files, read, mdx: read(".mdx").join("\n---\n") };
};

test("every corpus entry declares exactly one origin", () => {
  for (const entry of corpus.apps) {
    const origins = [entry.source ? "source" : null, entry.firstParty ? "firstParty" : null].filter(
      Boolean
    );
    assert.deepEqual(
      origins.length,
      1,
      `${entry.id}: expected exactly one of "source"/"firstParty", got [${origins.join(", ")}]`
    );
    if (entry.firstParty) {
      assert.ok(
        existsSync(join(root, entry.firstParty, "package.json")),
        `${entry.id}: ${entry.firstParty} has no package.json`
      );
      assert.ok(
        entry.firstPartyReason,
        `${entry.id}: a first-party fixture must say why no pinned third-party app covers it`
      );
    }
  }
});

test("the MDX fixture exercises a NON-EMPTY mdx-components override map", () => {
  const { files } = app("next-mdx-features");
  const overrides = files.find((f) => /mdx-components\.(js|jsx|ts|tsx)$/.test(f));
  assert.ok(overrides, "next-mdx-features must ship an mdx-components file");
  const source = readFileSync(overrides, "utf8");
  // The defect being guarded: `const components: MDXComponents = {}` — a file that
  // exists, is loaded, and overrides nothing, so the whole override path is untested.
  for (const tag of ["h1", "table", "del", "a"]) {
    assert.match(source, new RegExp(`\\b${tag}:`), `mdx-components overrides no ${tag}`);
  }
  // Each override must be OBSERVABLE by the e2e probe, which records `data-testid`
  // (not arbitrary attributes) — an override that changed nothing visible would be a
  // green run that proves nothing.
  assert.ok(
    (source.match(/data-testid=/g) ?? []).length >= 4,
    "each override must mark itself with a data-testid the probe compares"
  );
});

test("the MDX fixture exercises GFM, imports, JSX and exports", () => {
  const { mdx } = app("next-mdx-features");
  assert.match(mdx, /^\|.*\|$/m, "no GFM table");
  assert.match(mdx, /:-+:|:-+\||\|-+:/, "the table exercises no column alignment");
  assert.match(mdx, /~~[^~]+~~/, "no GFM strikethrough");
  assert.match(mdx, /^- \[[ x]\] /m, "no GFM task list");
  assert.match(mdx, /www\.[a-z]/, "no GFM autolink literal");
  assert.match(mdx, /\[\^[^\]]+\]/, "no GFM footnote");
  assert.match(mdx, /^import .* from ".*";$/m, "nothing is imported into the MDX");
  assert.match(mdx, /<[A-Z]\w*/, "no component is used as JSX inside the MDX");
  assert.match(mdx, /^export const /m, "no `export const` from the MDX");
  assert.match(mdx, /```/, "no fenced code block");
});

test("the plugin fixture exercises frontmatter and a real remark/rehype pipeline", () => {
  const { mdx, read } = app("next-pages-mdx-plugins");
  assert.match(mdx, /^---\n[\s\S]*?\n---\n/, "the MDX page has no YAML frontmatter");
  assert.match(mdx, /frontmatter\.\w+/, "the frontmatter is never read back by the page");
  assert.match(mdx, /^export const /m, "no `export const` from the MDX");
  assert.match(mdx, /^import .* from ".*";$/m, "nothing is imported into the MDX");
  assert.match(mdx, /<[A-Z]\w*/, "no component is used as JSX inside the MDX");

  // The point of this fixture is the OTHER compiler: plugins diffpack's native emitter
  // cannot run, so every file goes through the app's own @mdx-js/mdx. If the config
  // stopped configuring them, the fixture would silently become a duplicate of the
  // native-path one.
  const config = read("next.config.mjs").join("\n");
  assert.match(config, /remarkPlugins:\s*\[/, "no remarkPlugins configured");
  assert.match(config, /rehypePlugins:\s*\[/, "no rehypePlugins configured");
  assert.match(config, /remark-frontmatter/, "frontmatter is not stripped by the app's pipeline");
  const pkg = JSON.parse(readFileSync(join(app("next-pages-mdx-plugins").dir, "package.json"), "utf8"));
  assert.ok(
    pkg.dependencies["@mdx-js/mdx"],
    "the app must install @mdx-js/mdx: diffpack compiles this app's MDX with it"
  );
});

test("the native-compiler fixture does NOT configure a plugin diffpack must delegate", () => {
  // The negative half. `next-mdx-features` exists to cover diffpack's own Rust MDX
  // compiler; configuring any plugin beyond a bare `remark-gfm` would route it to the
  // app's pipeline instead and leave the native emitter untested again.
  const { read } = app("next-mdx-features");
  const config = read("next.config.mjs").join("\n");
  assert.match(config, /remarkGfm/, "the fixture must opt into GFM the way an app does");
  assert.equal(
    /rehypePlugins|recmaPlugins/.test(config),
    false,
    "a rehype/recma plugin would send this fixture to the app's own MDX pipeline"
  );
  const remark = config.match(/remarkPlugins:\s*\[([^\]]*)\]/);
  assert.ok(remark, "no remarkPlugins block");
  assert.deepEqual(
    remark[1]
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean),
    ["remarkGfm"],
    "only remark-gfm keeps this fixture on the native compiler"
  );
});
