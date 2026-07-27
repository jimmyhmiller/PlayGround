// Harness regression tests: `node --test integration/e2e/lib/apps.test.mjs`
//
// FINDINGS #20 was recorded as "diffpack injects styles into an app that ships
// none" (`next-github-pages` rendering in Inter where the reference rendered in
// Times). The app ships a stylesheet — a `next/font` `@font-face` chunk plus the
// class its layout applies. What was wrong was the ORACLE: an `output: 'export'`
// app with a `basePath` bakes that prefix into every emitted asset URL, and the
// suite served `out/` at `/`, so every `/<basePath>/_next/...` request missed —
// and the static server's last-resort `index.html` fallback then answered those
// misses with an HTML document. The reference page loaded no CSS and no JS at
// all, which read as "the app has no styles" and as a dead client-side
// navigation, and would have driven a "fix" that deleted a real feature.

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { freePort, startStaticServer, needsWebpackFlag } from "./apps.mjs";

const fixture = () => {
  const root = mkdtempSync(join(tmpdir(), "diffpack-static-server-"));
  mkdirSync(join(root, "_next/static"), { recursive: true });
  writeFileSync(join(root, "index.html"), "<!doctype html><p>home</p>");
  mkdirSync(join(root, "about"), { recursive: true });
  writeFileSync(join(root, "about/index.html"), "<!doctype html><p>about</p>");
  writeFileSync(join(root, "_next/static/app.css"), "body{color:red}");
  return root;
};

const withServer = async (root, basePath, fn) => {
  const server = await startStaticServer(root, await freePort(), basePath);
  try {
    await fn(`http://127.0.0.1:${server.port}`);
  } finally {
    await server.kill();
  }
};

test("a basePath'd static export serves its own baked asset URLs", async () => {
  await withServer(fixture(), "/gh-pages-test", async (origin) => {
    const css = await fetch(`${origin}/gh-pages-test/_next/static/app.css`);
    assert.equal(css.status, 200);
    assert.match(css.headers.get("content-type"), /text\/css/);
    assert.equal(await css.text(), "body{color:red}");

    const page = await fetch(`${origin}/gh-pages-test`);
    assert.equal(page.status, 200);
    assert.match(await page.text(), /home/);

    const about = await fetch(`${origin}/gh-pages-test/about`);
    assert.equal(about.status, 200);
    assert.match(await about.text(), /about/);
  });
});

// FINDINGS #31: `next-pages-mdx` was the one app the suite never compared,
// because its own toolchain could not build it. Its `next.config.js` is
// `module.exports = withMDX({ pageExtensions: [...] })` — the string "webpack"
// appears nowhere, so the harness's textual `--webpack` probe stayed silent,
// Next 16 built with Turbopack, and Turbopack refused a config whose `webpack`
// function `@next/mdx` had installed at runtime ("Call retries were exceeded
// { type: 'WorkerError' }"). The retry now reads Next's own diagnosis of that
// case off the failed build instead of trying to out-guess it statically.
test("a plugin-added webpack config is recognized from the failed build's output", () => {
  const pluginAdded = {
    ok: false,
    stdout:
      "ERROR: This build is using Turbopack, with a `webpack` config and no `turbopack` config.\n" +
      "   NOTE: your `webpack` config may have been added by a configuration plugin.\n",
    stderr: "> Build error occurred\nError: Call retries were exceeded\n  type: 'WorkerError'\n",
  };
  assert.equal(needsWebpackFlag(pluginAdded), true);

  // An unrelated failure must NOT be retried as a builder-choice problem.
  assert.equal(
    needsWebpackFlag({ ok: false, stdout: "", stderr: "Type error: 'params' is not assignable" }),
    false
  );
  assert.equal(needsWebpackFlag({ ok: false, stdout: "", stderr: "" }), false);
  // Missing streams must not throw.
  assert.equal(needsWebpackFlag({ ok: false }), false);
});

// Turbopack's OTHER refusal of a webpack-shaped config, with wording that shares
// not one word with the first: an MDX app configuring remark/rehype plugins puts
// live functions in a loader rule's options, and Turbopack runs loaders in another
// process. `next-mdx-features` failed exactly this way and was reported as
// "reference build failed" — the app excluded from comparison — until the retry
// learned this second phrasing.
test("a loader whose options Turbopack cannot serialize is a --webpack retry too", () => {
  const unserializable = {
    ok: false,
    stdout: "▲ Next.js 16.2.11 (Turbopack)\n Creating an optimized production build ...\n",
    stderr:
      "> Build error occurred\n" +
      'Error: loader /app/node_modules/@next/mdx/mdx-js-loader.js for match "{*,next-mdx-rule}" ' +
      "does not have serializable options. Ensure that options passed are plain JavaScript " +
      "objects and values.\n",
  };
  assert.equal(needsWebpackFlag(unserializable), true);

  // Still not a catch-all: a loader failing for any other reason is not a
  // builder-choice problem and must not be retried as one.
  assert.equal(
    needsWebpackFlag({
      ok: false,
      stdout: "",
      stderr: "Error: loader /app/node_modules/sass-loader/dist/cjs.js failed: no such file",
    }),
    false
  );
});

test("a missing asset 404s instead of silently becoming the index document", async () => {
  await withServer(fixture(), "", async (origin) => {
    const missing = await fetch(`${origin}/_next/static/gone.css`);
    assert.equal(missing.status, 404);
    const missingScript = await fetch(`${origin}/_next/static/gone.js`);
    assert.equal(missingScript.status, 404);
    // An extension-less path is still a client-side route: the SPA fallback stands.
    const route = await fetch(`${origin}/some/client/route`);
    assert.equal(route.status, 200);
    assert.match(await route.text(), /home/);
  });
});
