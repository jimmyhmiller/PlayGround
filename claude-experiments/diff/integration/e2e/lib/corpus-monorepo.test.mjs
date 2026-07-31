// Corpus regression tests for the three things the harness had to learn before
// it could express cal.com: `node --test integration/e2e/lib/corpus-monorepo.test.mjs`
//
// The gap. `fetch.mjs` could only materialize an app by COPYING one subdirectory
// out of a pinned checkout and running `npm install` inside the copy. cal.com is
// not expressible that way at all: `apps/web` imports 20 sibling workspace
// packages that resolve only from the repository root, the package manager is
// yarn 4 through corepack, and `apps/web/next.config.ts` throws unless three
// environment variables are set. An app that cannot be expressed is an app that
// is never tested, and this one is the largest real Next.js application the suite
// can reach.
//
// These tests read only `corpus.json` and the pure helpers in `apps.mjs`, so they
// need no materialized corpus, no network and no 3.4 GB of node_modules.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  appDirOf,
  e2eDir,
  envFileOf,
  installDirOf,
  licenseMatches,
  materializedRootOf,
  renderEnvFile,
  runnerFor,
  selectApps,
} from "./apps.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const corpus = JSON.parse(readFileSync(join(here, "..", "corpus.json"), "utf8"));
const entry = (id) => {
  const app = corpus.apps.find((a) => a.id === id);
  assert.ok(app, `${id} is missing from corpus.json`);
  return app;
};

// --- 1. monorepo entries --------------------------------------------------

test("a monorepo entry is materialized whole and built in its own checkout", () => {
  const app = entry("next-calcom");
  assert.deepEqual(app.monorepo, {
    appDir: "apps/web",
    packageManager: "corepack yarn",
    installAt: "root",
  });
  const root = materializedRootOf(app);
  assert.equal(root, join(e2eDir, ".cache", "calcom"));
  // Built in the named subdirectory of the repository, not in a copy of it.
  assert.equal(appDirOf(app), join(root, "apps", "web"));
  // Installed where the workspace hoists to. Asking `apps/web` whether it has
  // node_modules would answer "no" for a perfectly installed workspace.
  assert.equal(installDirOf(app), root);
  // The `.env` the app's own config loads (`dotenv({ path: "../../.env" })`).
  assert.equal(envFileOf(app), join(root, ".env"));
});

test("the ~35 subdir-copy entries are unaffected: root, app dir and install dir are all apps/<id>", () => {
  const subdirApps = corpus.apps.filter((a) => !a.monorepo);
  assert.ok(subdirApps.length >= 30, `expected the existing corpus, got ${subdirApps.length} entries`);
  for (const app of subdirApps) {
    const expected = join(e2eDir, "apps", app.id);
    assert.equal(materializedRootOf(app), expected, app.id);
    assert.equal(appDirOf(app), expected, app.id);
    assert.equal(installDirOf(app), expected, app.id);
    assert.equal(envFileOf(app), join(expected, ".env"), app.id);
    // …and they keep running their toolchain the way they always did.
    assert.deepEqual(runnerFor(app), ["npx", "--no-install"], app.id);
  }
});

test("a monorepo entry runs its toolchain through the package manager it declares", () => {
  // `npx --no-install next build` cannot find `next` from apps/web: a yarn
  // workspace links binaries into the ROOT's node_modules/.bin. Declared, not
  // guessed.
  assert.deepEqual(runnerFor(entry("next-calcom")), ["corepack", "yarn"]);
});

test("every monorepo entry is complete, exclusive, and alone in its checkout", () => {
  const seen = new Map();
  for (const app of corpus.apps) {
    if (!app.monorepo) continue;
    for (const key of ["appDir", "packageManager", "installAt"]) {
      assert.ok(app.monorepo[key], `${app.id}: "monorepo" is missing "${key}"`);
    }
    assert.ok(app.source, `${app.id}: a monorepo entry is a third-party checkout and needs a "source"`);
    assert.ok(!app.subdir, `${app.id}: a monorepo entry is built in place — "subdir" would be a copy`);
    assert.ok(!app.firstParty, `${app.id}: a monorepo entry cannot also be a first-party fixture`);
    // Two apps building in one checkout would overwrite each other's output.
    assert.ok(!seen.has(app.source), `${app.id}: source "${app.source}" is already built in place by ${seen.get(app.source)}`);
    seen.set(app.source, app.id);
    // `"sparse": []` is what makes the WHOLE repository get checked out; a
    // sparse checkout would silently miss the sibling workspace packages.
    assert.deepEqual(
      corpus.sources[app.source].sparse,
      [],
      `${app.id}: source "${app.source}" must declare "sparse": [] (the whole repository)`
    );
  }
});

// --- 2. declared env ------------------------------------------------------

test("declared env is written as a literal .env and is auditable", () => {
  const rendered = renderEnvFile({ A: "one", B: "http://localhost:3000" }, { id: "t" });
  assert.match(rendered, /^# Generated by integration\/e2e\/fetch\.mjs/m);
  assert.match(rendered, /^A="one"$/m);
  assert.match(rendered, /^B="http:\/\/localhost:3000"$/m);
});

test("an env value that cannot be written literally is refused, not mangled", () => {
  // Silently escaping (or dropping) a quote would hand the app a value that is
  // not the declared one, and the provenance file would then be wrong.
  assert.throws(() => renderEnvFile({ A: 'he said "hi"' }, { id: "t" }), /must be a string with no quote/);
  assert.throws(() => renderEnvFile({ A: "line\nbreak" }, { id: "t" }), /must be a string with no quote/);
  assert.throws(() => renderEnvFile({ A: 1 }, { id: "t" }), /must be a string/);
  assert.throws(() => renderEnvFile({ "not-an-identifier": "x" }, { id: "t" }), /shell-safe identifier/);
});

test("no declared env value can be a real secret", () => {
  // Everything declared here is written into a provenance file that is meant to
  // be read. A value that is not obviously a dummy is either a leak or a lie.
  const dummy = /diffpack[-_]e2e|localhost|127\.0\.0\.1|no_such|^[01]$/;
  for (const app of corpus.apps) {
    for (const [key, value] of Object.entries(app.env ?? {})) {
      assert.match(value, dummy, `${app.id}: env ${key} does not look like a dummy value`);
    }
    if (app.env) {
      assert.ok(app.envNote, `${app.id}: declared env must say why the build needs it`);
    }
  }
});

test("cal.com's declared env covers exactly the assertions its own config makes", () => {
  const app = entry("next-calcom");
  // apps/web/next.config.ts (at the pinned SHA) throws on each of these.
  for (const key of ["NEXTAUTH_SECRET", "CALENDSO_ENCRYPTION_KEY", "NEXTAUTH_URL"]) {
    assert.ok(app.env[key], `next-calcom must declare ${key}: its config throws without it`);
  }
  // The database is deliberately dead: an oracle that needs a live service is
  // not an oracle. If this ever points somewhere real, the reference build stops
  // measuring the bundler.
  assert.match(app.env.DATABASE_URL, /no_such_database/);
});

// --- 3. opt-in heavy apps -------------------------------------------------

test("a heavy app is not part of a default fetch or a default run", () => {
  const { selected, excludedHeavy } = selectApps(corpus.apps);
  assert.deepEqual(
    excludedHeavy.map((a) => a.id),
    ["next-calcom"]
  );
  assert.equal(
    selected.some((a) => a.id === "next-calcom"),
    false
  );
  // …and nothing else was lost: the default corpus is exactly the non-heavy set.
  assert.deepEqual(
    selected.map((a) => a.id),
    corpus.apps.filter((a) => !a.heavy).map((a) => a.id)
  );
});

test("--heavy includes it, and a filter alone does not", () => {
  assert.equal(selectApps(corpus.apps, { heavy: true }).selected.some((a) => a.id === "next-calcom"), true);
  // Naming it is not consent to a 20-minute install: the flag is.
  const filtered = selectApps(corpus.apps, { filters: ["next-calcom"] });
  assert.deepEqual(filtered.selected, []);
  assert.deepEqual(
    filtered.excludedHeavy.map((a) => a.id),
    ["next-calcom"]
  );
  assert.deepEqual(
    selectApps(corpus.apps, { filters: ["next-calcom"], heavy: true }).selected.map((a) => a.id),
    ["next-calcom"]
  );
});

test("filtering the ordinary way still works and is unchanged by heaviness", () => {
  assert.deepEqual(
    selectApps(corpus.apps, { filters: ["next-hello-world", "vite-react-ts"] }).selected.map((a) => a.id),
    ["next-hello-world", "vite-react-ts"]
  );
});

test("every heavy entry says what makes it heavy", () => {
  for (const app of corpus.apps.filter((a) => a.heavy)) {
    assert.ok(app.heavyNote, `${app.id}: "heavy" must say what the cost is`);
  }
});

// --- 4. what the reference build is allowed to relax -----------------------

test("a declared type/lint relaxation must say why", () => {
  for (const app of corpus.apps.filter((a) => a.relaxChecks)) {
    assert.ok(app.relaxChecksNote, `${app.id}: "relaxChecks" must say why the app's own gates cannot pass`);
  }
  const app = entry("next-calcom");
  assert.equal(app.relaxChecks, true);
  assert.match(app.relaxChecksNote, /type/i);
});

// --- 5. the license is read, not assumed ----------------------------------

test("license verification recognizes the licenses the corpus declares", () => {
  assert.equal(licenseMatches("MIT", "MIT License\n\nCopyright (c) 2020-present Cal.com, Inc.\n"), true);
  assert.equal(
    licenseMatches("MIT", "                    GNU AFFERO GENERAL PUBLIC LICENSE\n                       Version 3\n"),
    false
  );
  assert.equal(
    licenseMatches("AGPL-3.0", "                    GNU AFFERO GENERAL PUBLIC LICENSE\n"),
    true
  );
  // An id nobody taught the harness is refused, not quietly treated as "matches".
  assert.throws(() => licenseMatches("WTFPL", "anything"), /unknown license id/);
});

test("cal.com's license is recorded off the pinned tree, with the AGPL history written down", () => {
  const src = corpus.sources[entry("next-calcom").source];
  // cal.com's platform was AGPL-3.0 for years; at the pinned SHA the repository
  // has been relicensed and ships an MIT LICENSE. The corpus records what the
  // tree says — and fetch.mjs re-checks it — because a provenance file that
  // repeats a project's reputation instead of reading its LICENSE is worthless.
  assert.equal(src.license, "MIT");
  assert.match(src.licenseNote, /AGPL-3\.0/);
  assert.match(src.licenseNote, /clone|CLONE/);
  assert.match(src.licenseNote, /vendor|Vendor|vendored/);
  assert.ok(src.copyright);
});

// --- 6. the entry itself ---------------------------------------------------

test("next-calcom is a real, driveable corpus entry", () => {
  const app = entry("next-calcom");
  assert.equal(app.kind, "next-app");
  assert.equal(app.heavy, true);
  assert.ok(app.why, "every corpus entry says why it is in the corpus");
  assert.ok(app.routes.length >= 2, "at least two routes are compared");
  assert.ok(app.routesNote, "the route choice must be justified: most cal.com routes need a database");
  for (const route of app.routes) assert.match(route, /^\//);
});
