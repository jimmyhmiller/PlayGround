// The real-application end-to-end truth test.
//
//   node integration/e2e/run.mjs                 # whole corpus
//   node integration/e2e/run.mjs next-mdx        # id substring filter
//   node integration/e2e/run.mjs --no-build      # reuse existing build output
//   node integration/e2e/run.mjs --build-only
//   node integration/e2e/run.mjs --jobs 4        # parallel builds (default 3)
//
// For every pinned third-party app the same untouched source is built twice —
// once by its own toolchain, once by diffpack — then both deployments are
// driven by the same script in the same real browser and compared across
// text, structure, computed styles, layout, assets, links, hydration,
// interaction, client-side navigation, and the browser's error channels.
//
// Exit code 0 only when every app that built produced no `fail` finding.
import { existsSync, mkdirSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { adapters, cleanBuildOutput, basePathOf, diffpackBin, sleep } from "./lib/apps.mjs";
import { Browser, closeAll } from "./lib/browser.mjs";
import {
  DETERMINISM_INIT,
  PROBE_SOURCE,
  SETTLE_SOURCE,
  LOCATION_SOURCE,
  clickSource,
  navigateSource,
} from "./lib/probe.mjs";
import { compareRecords, compareErrors, maskVolatileText, isFailure } from "./lib/compare.mjs";
import { scriptsInsideTags } from "../../scripts/rsc/html-integrity.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const resultsDir = join(here, "results");
const corpus = JSON.parse(readFileSync(join(here, "corpus.json"), "utf8"));

const args = process.argv.slice(2);
const flag = (name) => args.includes(name);
const option = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const filters = args.filter((a) => !a.startsWith("--") && args[args.indexOf(a) - 1] !== "--jobs");
const noBuild = flag("--no-build");
const buildOnly = flag("--build-only");
const jobs = Number.parseInt(option("--jobs", "3"), 10);
const maxClicks = Number.parseInt(option("--clicks", "3"), 10);

const apps = corpus.apps.filter((a) => !filters.length || filters.some((f) => a.id.includes(f)));
if (!apps.length) {
  console.error("no apps matched");
  process.exit(2);
}
if (!existsSync(diffpackBin)) {
  console.error(`diffpack binary missing: ${diffpackBin} (cargo build --release)`);
  process.exit(2);
}

mkdirSync(resultsDir, { recursive: true });
const initScriptPath = join(resultsDir, "determinism-init.js");
writeFileSync(initScriptPath, DETERMINISM_INIT);

const appDirOf = (app) => join(here, "apps", app.id);
const outDirOf = (app) => join(resultsDir, app.id);

// --- phase 1: build both sides -------------------------------------------
const buildResults = new Map();

const buildOne = async (app) => {
  const appDir = appDirOf(app);
  const out = outDirOf(app);
  mkdirSync(out, { recursive: true });
  const adapter = adapters[app.kind];
  if (!adapter) return { skipped: `unknown kind ${app.kind}` };
  if (!existsSync(join(appDir, "node_modules"))) {
    return { skipped: "node_modules missing (run fetch.mjs)" };
  }
  cleanBuildOutput(appDir);

  const reference = adapter.buildReference(app, appDir);
  writeFileSync(join(out, "build-reference.log"), `${reference.stdout}\n--- stderr ---\n${reference.stderr}`);
  const diffpack = adapter.buildDiffpack(app, appDir);
  writeFileSync(join(out, "build-diffpack.log"), `${diffpack.stdout}\n--- stderr ---\n${diffpack.stderr}`);
  return {
    reference: { ok: reference.ok, ms: reference.ms, status: reference.status, timedOut: reference.timedOut },
    diffpack: { ok: diffpack.ok, ms: diffpack.ms, status: diffpack.status, timedOut: diffpack.timedOut },
    diffpackError: diffpack.ok ? null : tail(diffpack.stderr || diffpack.stdout),
    referenceError: reference.ok ? null : tail(reference.stderr || reference.stdout),
  };
};

function tail(text, lines = 25) {
  return (text ?? "").trim().split("\n").slice(-lines).join("\n");
}

if (!noBuild) {
  console.log(`== building ${apps.length} app(s), ${jobs} at a time ==`);
  const queue = [...apps];
  const workers = Array.from({ length: Math.max(1, jobs) }, async () => {
    while (queue.length) {
      const app = queue.shift();
      const started = Date.now();
      const result = await buildOne(app);
      buildResults.set(app.id, result);
      const status = result.skipped
        ? `SKIP (${result.skipped})`
        : `reference=${result.reference.ok ? "ok" : "FAIL"} diffpack=${result.diffpack.ok ? "ok" : "FAIL"}`;
      console.log(`  ${app.id.padEnd(30)} ${status}  ${Math.round((Date.now() - started) / 1000)}s`);
    }
  });
  await Promise.all(workers);
  writeFileSync(join(resultsDir, "builds.json"), `${JSON.stringify(Object.fromEntries(buildResults), null, 2)}\n`);
} else if (existsSync(join(resultsDir, "builds.json"))) {
  for (const [id, value] of Object.entries(JSON.parse(readFileSync(join(resultsDir, "builds.json"), "utf8")))) {
    buildResults.set(id, value);
  }
}

if (buildOnly) {
  console.log("\n--build-only: stopping before the browser phase");
  process.exit(0);
}

// --- phase 2: drive both deployments in a real browser --------------------
const report = { generatedAt: new Date().toISOString(), apps: [] };
// Which served documents actually carried inline flight scripts (i.e. were streamed).
const streamedRoutes = [];

const probeRoute = async (browser, origin, route) => {
  await browser.clearObservations();
  const opened = await browser.open(`${origin}${route}`);
  if (opened.status !== 0 && !opened.stdout.includes("http")) {
    return {
      record: null,
      observations: await browser.observations(),
      openFailure: opened.timedOut ? "agent-browser open timed out" : opened.stderr || opened.stdout,
    };
  }
  await browser.eval(SETTLE_SOURCE);
  const probe = await browser.eval(PROBE_SOURCE);
  return {
    record: probe.ok ? probe.value : null,
    probeRaw: probe.ok ? null : probe.raw,
    observations: await browser.observations(),
  };
};

const sides = ["reference", "diffpack"];

for (const app of apps) {
  const build = buildResults.get(app.id);
  const entry = { id: app.id, kind: app.kind, why: app.why, build, findings: [], routes: [] };
  report.apps.push(entry);
  const out = outDirOf(app);
  mkdirSync(out, { recursive: true });

  if (!build || build.skipped) {
    entry.status = "skipped";
    entry.reason = build?.skipped ?? "not built";
    console.log(`\n## ${app.id}: SKIP (${entry.reason})`);
    continue;
  }
  if (!build.reference.ok) {
    entry.status = "reference-build-failed";
    entry.findings.push({
      channel: "reference",
      severity: "info",
      summary: `${app.id}: the app's own toolchain failed to build it — it cannot serve as an oracle`,
      detail: build.referenceError,
    });
    console.log(`\n## ${app.id}: reference build FAILED (app excluded from comparison)`);
    continue;
  }
  if (!build.diffpack.ok) {
    entry.status = "diffpack-build-failed";
    entry.findings.push({
      channel: "build",
      severity: "fail",
      summary: `${app.id}: diffpack could not build an app that ${app.kind === "vite" ? "vite" : "next"} builds`,
      detail: build.diffpackError,
    });
    console.log(`\n## ${app.id}: diffpack build FAILED`);
    console.log(`   ${(build.diffpackError ?? "").split("\n").slice(-4).join("\n   ")}`);
    continue;
  }

  const appDir = appDirOf(app);
  const adapter = adapters[app.kind];
  const logs = { reference: [], diffpack: [] };
  const servers = {};
  let serveFailed = false;
  // Apps with server-side persistence (a counter written to disk, a seeded
  // store) must start each side from the same state, or the second side served
  // inherits the first side's mutations and the difference is reported against
  // the bundler instead of the app. `resetFiles` names them in corpus.json.
  const resetAppState = () => {
    for (const relative of app.resetFiles ?? []) {
      rmSync(join(appDirOf(app), relative), { force: true, recursive: true });
    }
  };

  for (const side of sides) {
    resetAppState();
    const method = side === "reference" ? "serveReference" : "serveDiffpack";
    const server = await adapter[method](app, appDir, logs[side]);
    servers[side] = server;
    if (!server.ok) {
      serveFailed = true;
      entry.findings.push({
        channel: "serve",
        severity: side === "diffpack" ? "fail" : "info",
        summary: `${app.id}: the ${side} build did not serve (${server.reason})`,
        detail: logs[side].join("").slice(-4000),
      });
    }
  }
  writeFileSync(join(out, "server-reference.log"), logs.reference.join(""));
  writeFileSync(join(out, "server-diffpack.log"), logs.diffpack.join(""));

  if (serveFailed) {
    entry.status = "serve-failed";
    for (const side of sides) await servers[side]?.kill?.();
    console.log(`\n## ${app.id}: serve FAILED`);
    continue;
  }

  const base = basePathOf(appDir);
  const origins = {
    reference: `http://127.0.0.1:${servers.reference.port}`,
    diffpack: `http://127.0.0.1:${servers.diffpack.port}`,
  };
  const browsers = {
    reference: new Browser(`dp-e2e-ref-${app.id}`, { initScript: initScriptPath }),
    diffpack: new Browser(`dp-e2e-dp-${app.id}`, { initScript: initScriptPath }),
  };

  console.log(`\n## ${app.id} (${app.kind})`);
  const routes = (app.routes ?? ["/"]).map((r) => (base && !r.startsWith(base) ? `${base}${r === "/" ? "" : r}` : r));

  try {
    for (const route of routes) {
      const observed = {};
      for (const side of sides) {
        observed[side] = await probeRoute(browsers[side], origins[side], route);
        writeFileSync(
          join(out, `probe-${route.replace(/[^a-z0-9]+/gi, "_") || "root"}-${side}.json`),
          `${JSON.stringify(observed[side], null, 2)}\n`
        );
      }
      // Raw-document channel. The browser's parser RECOVERS from a `<script>` spliced
      // into the middle of a tag, so the DOM probe above can look almost fine while the
      // served bytes are corrupt. Retain the raw HTML as evidence and check it.
      const routeSlug = route.replace(/[^a-z0-9]+/gi, "_") || "root";
      const rawFindings = [];
      for (const side of sides) {
        const raw = await fetch(`${origins[side]}${route}`)
          .then((r) => r.text())
          .catch((error) => `<!-- e2e: fetch failed: ${error.message} -->`);
        writeFileSync(join(out, `document-${routeSlug}-${side}.html`), raw);
        // Record whether this document was streamed at all. A prerendered page
        // carries no inline flight scripts, so it cannot demonstrate anything
        // about the streaming SSR path — the suite has to say so rather than
        // let a static page stand in as coverage.
        if (side === "diffpack") {
          streamedRoutes.push({ app: app.id, route, flightScripts: (raw.match(/__DF_FLIGHT/g) ?? []).length });
        }
        const split = scriptsInsideTags(raw);
        if (split.length) {
          rawFindings.push({
            channel: "html-integrity",
            severity: side === "diffpack" ? "fail" : "info",
            summary: `${route}: the ${side} document has a <script> inside an open tag (${split.length})`,
            detail: split.slice(0, 5),
          });
        }
      }
      const routeFindings = [
        ...rawFindings,
        ...compareRecords(observed.reference.record, observed.diffpack.record, { label: route, volatile: app.volatile }),
        ...compareErrors(observed.reference.observations, observed.diffpack.observations, { label: route }),
      ];
      entry.findings.push(...routeFindings);
      entry.routes.push({ route, findings: routeFindings.length });
      const fails = routeFindings.filter(isFailure).length;
      console.log(`   ${route.padEnd(26)} ${fails === 0 ? "OK" : `${fails} difference(s)`}`);
      for (const f of routeFindings.filter(isFailure).slice(0, 4)) {
        console.log(`      - [${f.channel}] ${f.summary}`);
      }
      if (fails > 4) console.log(`      ... +${fails - 4} more (see results/${app.id}/findings.json)`);
    }

    // Interaction: click the same elements, in the same order, on both sides.
    const firstRoute = routes[0];
    const interaction = { reference: [], diffpack: [] };
    for (const side of sides) {
      // Clicking can mutate server-side state; both sides must start from the same place.
      resetAppState();
      await browsers[side].open(`${origins[side]}${firstRoute}`);
      await browsers[side].eval(SETTLE_SOURCE);
      for (let i = 0; i < maxClicks; i++) {
        const clicked = await browsers[side].eval(clickSource(i));
        if (!clicked.ok || !clicked.value?.clicked) break;
        const probe = await browsers[side].eval(PROBE_SOURCE);
        interaction[side].push({
          index: i,
          label: clicked.value.label,
          bodyText: probe.ok ? probe.value.bodyText : null,
          pathname: probe.ok ? probe.value.location.pathname : null,
        });
      }
    }
    writeFileSync(join(out, "interaction.json"), `${JSON.stringify(interaction, null, 2)}\n`);
    const interactionFindingsBefore = entry.findings.length;
    if (interaction.reference.length !== interaction.diffpack.length) {
      entry.findings.push({
        channel: "interaction",
        severity: "fail",
        summary: `${app.id}: ${interaction.reference.length} click(s) possible on the reference, ${interaction.diffpack.length} on diffpack`,
        detail: interaction,
      });
    } else {
      for (let i = 0; i < interaction.reference.length; i++) {
        const a = interaction.reference[i];
        const b = interaction.diffpack[i];
        const same =
          a.label === b.label &&
          a.pathname === b.pathname &&
          maskVolatileText(a.bodyText, app.volatile) === maskVolatileText(b.bodyText, app.volatile);
        if (!same) {
          entry.findings.push({
            channel: "interaction",
            severity: "fail",
            summary: `${app.id}: click #${i} ("${a.label}") produced a different result`,
            detail: { reference: a, diffpack: b },
          });
        }
      }
    }
    if (interaction.reference.length) {
      const ok = entry.findings.length === interactionFindingsBefore;
      console.log(`   ${"interaction".padEnd(26)} ${ok ? `OK (${interaction.reference.length} click(s))` : "differs"}`);
    }

    // Client-side navigation across the first internal link. The click is
    // scheduled, never awaited in-page (see navigateSource), then the landing
    // location is polled — a full document navigation would otherwise destroy
    // the execution context an awaited evaluation is suspended in.
    const nav = {};
    for (const side of sides) {
      resetAppState();
      await browsers[side].open(`${origins[side]}${firstRoute}`);
      await browsers[side].eval(SETTLE_SOURCE);
      const scheduled = await browsers[side].eval(navigateSource(0));
      let landed = null;
      if (scheduled.ok && scheduled.value?.scheduled) {
        for (let attempt = 0; attempt < 20; attempt++) {
          await sleep(250);
          const where = await browsers[side].eval(LOCATION_SOURCE);
          if (!where.ok) continue;
          landed = where.value;
          if (landed.pathname !== scheduled.value.from && landed.readyState === "complete") break;
        }
      }
      const navigated = Boolean(landed && landed.pathname !== scheduled.value?.from);
      const probe = navigated ? await browsers[side].eval(PROBE_SOURCE) : null;
      nav[side] = {
        result: scheduled.ok ? { ...scheduled.value, navigated, now: landed?.pathname ?? null } : null,
        bodyText: probe?.ok ? probe.value.bodyText : null,
        title: probe?.ok ? probe.value.title : null,
      };
    }
    writeFileSync(join(out, "navigation.json"), `${JSON.stringify(nav, null, 2)}\n`);
    if (JSON.stringify(nav.reference) !== JSON.stringify(nav.diffpack)) {
      entry.findings.push({
        channel: "navigation",
        severity: "fail",
        summary: `${app.id}: client-side navigation to the first internal link diverged`,
        detail: nav,
      });
      console.log(`   ${"navigation".padEnd(26)} differs`);
    } else if (nav.reference.result?.navigated) {
      console.log(`   ${"navigation".padEnd(26)} OK (-> ${nav.reference.result.now})`);
    }

    for (const side of sides) {
      await browsers[side].screenshot(join(out, `screenshot-${side}.png`));
    }
  } finally {
    for (const side of sides) {
      await browsers[side].close();
      await servers[side]?.kill?.();
    }
    await sleep(200);
  }

  const fails = entry.findings.filter(isFailure).length;
  entry.status = fails === 0 ? "pass" : "differs";
  writeFileSync(join(out, "findings.json"), `${JSON.stringify(entry.findings, null, 2)}\n`);
}

closeAll();

// --- report ---------------------------------------------------------------
report.streaming = streamedRoutes;
writeFileSync(join(resultsDir, "report.json"), `${JSON.stringify(report, null, 2)}\n`);

const rows = report.apps.map((a) => ({
  id: a.id,
  status: a.status,
  fails: a.findings.filter(isFailure).length,
  channels: [...new Set(a.findings.filter(isFailure).map((f) => f.channel))].join(","),
}));

const summary = [
  "# diffpack real-application e2e — results",
  "",
  `Generated ${report.generatedAt}`,
  "",
  "| app | status | failing findings | channels |",
  "| --- | --- | --- | --- |",
  ...rows.map((r) => `| ${r.id} | ${r.status} | ${r.fails} | ${r.channels || "-"} |`),
  "",
].join("\n");
writeFileSync(join(resultsDir, "SUMMARY.md"), `${summary}\n`);

console.log(`\n${summary}`);
const streamed = streamedRoutes.filter((r) => r.flightScripts > 0);
if (streamedRoutes.length && !streamed.length) {
  console.log(
    "\nNOTE: no route in this run was served by the streaming SSR path — every document was\n" +
      "prerendered, so nothing here exercises inline flight injection. Add or select a\n" +
      "dynamically-rendered app-router route to cover it."
  );
} else if (streamed.length) {
  console.log(`\nstreaming SSR exercised on ${streamed.length} route(s): ${streamed.map((r) => `${r.app}${r.route}`).join(", ")}`);
}
// Only apps that were actually COMPARED can be said to behave identically.
// Skipped apps and apps whose own toolchain could not build them were never
// measured, and must not inflate the numerator.
const compared = rows.filter((r) => r.status !== "skipped" && r.status !== "reference-build-failed");
const bad = compared.filter((r) => r.status !== "pass");
const unmeasured = rows.length - compared.length;
console.log(
  `\n${compared.length - bad.length}/${compared.length} compared app(s) behave identically to their own toolchain` +
    (unmeasured ? ` (${unmeasured} not compared: skipped or reference build failed)` : "") +
    "."
);
process.exit(bad.length ? 1 : 0);
