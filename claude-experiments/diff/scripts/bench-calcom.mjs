// One script: the whole diffpack-vs-Turbopack table for cal.com.
//
//   node scripts/bench-calcom.mjs
//   node scripts/bench-calcom.mjs --pairs 3 --samples 7
//   node scripts/bench-calcom.mjs --only dev            # skip the build section
//   node scripts/bench-calcom.mjs --app /tmp/dpe2e/calcom --port 3000
//
// It measures, on the SAME cal.com checkout, with the SAME detection code on both
// sides, and prints the finished table (plus JSON):
//
//   PRODUCTION BUILD  (interleaved diffpack/Turbopack pairs, outputs wiped each time)
//     1. cold build wall
//     2. cold build CPU (user time, whole tree)
//     3. cold build peak RSS, whole process tree (sampled) — the comparable one
//     4. …and ru_maxrss, its largest single process, labelled as under-reporting
//   DEV SERVER        (cold, per bundler: wipe output, boot, first 200, second route)
//     4. cold start -> first 200 on /auth/login
//     5. second route (/pro/30min), first request after that first 200
//   HMR: edit -> updated DOM  (Playwright, one page.evaluate poll per 10ms, both sides)
//     6. island (leaf client component)      modules/auth/login-view.tsx
//     7. shared client component             components/PageWrapperAppDir.tsx (on the booker)
//     8. server component                    app/(use-page-wrapper)/auth/login/page.tsx
//     9. global CSS                          styles/globals.css
//    10. sustained island edits at ~1/sec    (the contended path, no settle gap)
//    11. first-edit warmup, per class        (the setup edit: route compile + first delivery)
//   EDIT -> FRESHLY SERVED DOCUMENT  (no browser; one fetch per 25ms, both sides)
//    12. island edit
//    13. server-component edit
//
// METHOD, and why each choice is the fair one:
//   * Builds are INTERLEAVED (dp, tp, dp, tp, ...) so ambient load lands on both
//     roughly equally; every raw sample is reported, never just the median.
//   * Wall is measured by this process. CPU (user) comes from `/usr/bin/time -l`,
//     which accumulates waited-for descendants and is therefore a true tree total.
//   * PEAK MEMORY is SAMPLED over each side's whole process tree (250 ms, summed,
//     maxed), because `/usr/bin/time -l`'s `maximum resident set size` is `ru_maxrss`
//     — the largest SINGLE process — and both sides are trees: diffpack's production
//     build runs client, react-server and ssr concurrently, `next build` spawns
//     workers. ru_maxrss under-reports both, unequally. It is still reported, on its
//     own clearly labelled row, and it is never the headline.
//   * `next build` is given `--turbopack` EXPLICITLY. Next 16 defaults to it, but the
//     default is version- and config-dependent (Next hard-exits when it auto-selects
//     Turbopack for a project with a webpack config and no turbopack config), and a
//     harness about Turbopack must not depend on that resolving the way it does today.
//   * Both sides build with this checkout's `typescript.ignoreBuildErrors` and
//     `eslint.ignoreDuringBuilds` (the benchmark next.config wrapper): diffpack does no
//     type checking, so leaving `tsc` inside `next build` would time a compiler only
//     one side runs.
//   * READY means the real page: 200, plus `--ready-marker` (default "Cal.diy") in the
//     body, plus a closed `</html>`. A bare 200 from an error shell or an unfinished
//     stream would otherwise hand a side startup time it did not earn; those are
//     counted and printed.
//   * FORCE_COLOR=0 everywhere: this environment exports FORCE_COLOR=3 and Node
//     ANSI-colorizes bare numbers, which has corrupted a parsed run before.
//   * The port is checked before every boot and the full descendant tree is killed
//     after: a leaked dev server answering from a previous run once made Turbopack
//     look 2.3x faster than it is.
//   * The browser opens `localhost`, not 127.0.0.1 — Next 16 blocks its dev HMR
//     WebSocket as cross-origin otherwise and silently degrades to a hard reload.
//   * Every edit writes a unique marker and detection is a boolean over the DOM /
//     document text, so "was it already showing that?" cannot pass.
//   * A sample that never lands is recorded as TIMEOUT and counted in the cell; it
//     is never dropped, and never averaged away.
//
// Edited files are snapshotted and restored on exit, including SIGINT/SIGTERM.
// Requires: cargo build --release; the cal.com checkout with node_modules and a
// seeded Postgres (see docs/STATUS_2026-07-28.md); Playwright resolved from it.

import { spawn, execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, rmSync, mkdirSync, existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { loadavg } from "node:os";
import { sampleTreeRss } from "./tree-rss.mjs";

const scriptsDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = dirname(scriptsDir);

const args = parseArgs(process.argv.slice(2));
const APP = args.app ?? "/tmp/dpe2e/calcom";
const WEB = join(APP, "apps", "web");
const PORT = args.port ?? 3000;
const PAIRS = args.pairs ?? 2;
const SAMPLES = args.samples ?? 5;
const GAP_MS = args.gap ?? 4200;       // clears the deferred re-emit window between samples
const SETTLE_MS = args.settle ?? 4200; // after the warmup edit, before the measured ones
const CADENCE_GAP_MS = args.cadence ?? 800; // "~1/sec" sustained-edit gap
const ONLY = args.only ?? "all";       // all | build | dev
const OUT = args.out ?? join(repoRoot, "bench", "results", "calcom-vs-turbopack.json");
// A 200 alone is not proof a page rendered: a dev server that answers an error shell, or
// that streams a document it never closes, would be credited with a startup time it did
// not earn — and the two sides need not fail the same way. Ready therefore means 200 AND
// the body carries `READY_MARKER` AND the document is closed. Bare 200s are counted and
// reported rather than ignored, since a side that produces them is exactly the side a
// status-code-only gate would flatter. Declared HERE, with the other config: the sections
// below run during module evaluation, so a `const` further down the file is still in its
// temporal dead zone when `waitFor200` first reads it.
const READY_MARKER = args.readyMarker ?? "Cal.diy";

const diffpackBin = join(repoRoot, "target", "release", "diffpack");
const nextBin = join(APP, "node_modules", ".bin", "next");
const BASE = `http://localhost:${PORT}`;
const HEALTH = `http://127.0.0.1:${PORT}`;

// ---------------------------------------------------------------------------
// The two bundlers. Everything below is written once and run for both.

const SIDES = [
  {
    key: "diffpack",
    label: "diffpack",
    // Every generated tree a cold run has to remove. `.diffpack-next/` is generated
    // glue rather than compiled output and is rewritten on every boot, but a cold run
    // that wipes all of one side's generated files and part of the other's leaves a
    // question open for no reason.
    outDirs: [join(WEB, ".diffpack-output"), join(WEB, ".diffpack-next")],
    build: () => ({ cmd: diffpackBin, argv: ["build-app", ".", "production"], cwd: WEB }),
    dev: () => ({ cmd: diffpackBin, argv: ["dev", WEB, String(PORT)], cwd: repoRoot }),
  },
  {
    key: "turbopack",
    label: "Turbopack",
    outDirs: [join(WEB, ".next")],
    // `--turbopack` explicitly: Next 16 defaults to it for `build`, but the default is
    // version- and config-dependent (and Next hard-exits when it auto-selects
    // Turbopack for a project with a webpack config and no turbopack config), so a
    // harness whose whole subject is Turbopack should not be relying on it.
    build: () => ({ cmd: nextBin, argv: ["build", "--turbopack"], cwd: WEB }),
    dev: () => ({ cmd: nextBin, argv: ["dev", "--turbopack", "--port", String(PORT)], cwd: WEB }),
  },
];

// ---------------------------------------------------------------------------
// The edit classes. Each names its anchor in the real source; a missing anchor is
// a hard error, never a skipped or silently-zero measurement.

const islandFile = join(WEB, "modules", "auth", "login-view.tsx");
const sharedFile = join(WEB, "components", "PageWrapperAppDir.tsx");
const serverFile = join(WEB, "app", "(use-page-wrapper)", "auth", "login", "page.tsx");
const cssFile = join(WEB, "styles", "globals.css");

const island = {
  key: "island",
  name: "island (leaf client component)",
  url: "/auth/login",
  file: islandFile,
  anchor: ">Cal.diy<",
  setup: (s) => s.replace(">Cal.diy<", ">Cal.diy HMRMARK_0<"),
  step: (s, n) => s.replace(/HMRMARK_\d+/, `HMRMARK_${n}`),
  detect: (n) => `document.body.textContent.includes("HMRMARK_${n}")`,
};

const HMR_KINDS = [
  island,
  {
    key: "shared",
    name: "shared client component",
    url: "/pro/30min",
    file: sharedFile,
    anchor: "return (\n    <>",
    setup: (s) => s.replace("return (\n    <>", `return (\n    <><span data-hmrshared="0" style={{ display: "none" }} />`),
    step: (s, n) => s.replace(/data-hmrshared="\d+"/, `data-hmrshared="${n}"`),
    detect: (n) => `!!document.querySelector('[data-hmrshared="${n}"]')`,
  },
  {
    key: "server",
    name: "server component",
    url: "/auth/login",
    file: serverFile,
    anchor: "return <Login {...props} />;",
    setup: (s) => s.replace(
      "return <Login {...props} />;",
      `return (<><span data-hmrsrv="0" style={{ display: "none" }} /><Login {...props} /></>);`,
    ),
    step: (s, n) => s.replace(/data-hmrsrv="\d+"/, `data-hmrsrv="${n}"`),
    detect: (n) => `!!document.querySelector('[data-hmrsrv="${n}"]')`,
  },
  {
    key: "css",
    name: "global CSS",
    url: "/auth/login",
    file: cssFile,
    anchor: "", // appended, no anchor needed
    timeoutMs: 45000,
    setup: (s) => `${s}\n:root { --hmrmark: 0; }\n`,
    step: (s, n) => s.replace(/--hmrmark: \d+/, `--hmrmark: ${n}`),
    detect: (n) => `getComputedStyle(document.documentElement).getPropertyValue("--hmrmark").trim() === "${n}"`,
  },
  {
    // Same edit as `island`, with no settle gap: the contended path, where a
    // bundler that queues work behind a deferred re-emit shows it.
    key: "cadence",
    name: `sustained edits @ ~1/sec`,
    url: "/auth/login",
    file: islandFile,
    anchor: "HMRMARK_", // planted by the island kind, which runs first
    gapMs: CADENCE_GAP_MS,
    settleMs: 0,
    plantedAnchor: true, // the anchor is planted by an earlier kind, not present in pristine source
    setup: (s) => s, // no setup edit; reuse the island marker
    setupDetect: () => `document.body.textContent.includes("HMRMARK_")`,
    step: (s, n) => s.replace(/HMRMARK_\d+/, `HMRMARK_9${n}${n}`),
    detect: (n) => `document.body.textContent.includes("HMRMARK_9${n}${n}")`,
  },
];

// The fresh-document kinds run from pristine sources (the HMR section restores
// before handing over), so each plants its own marker from a real source anchor.
const FRESH_KINDS = [
  {
    key: "fresh-island",
    name: "island edit",
    url: "/auth/login",
    file: islandFile,
    anchor: ">Cal.diy<",
    setup: (s) => s.replace(">Cal.diy<", ">Cal.diy FRESHMARK_0<"),
    step: (s, n) => s.replace(/FRESHMARK_\d+/, `FRESHMARK_${n}`),
    marker: (n) => `FRESHMARK_${n}`,
  },
  {
    key: "fresh-server",
    name: "server-component edit",
    url: "/auth/login",
    file: serverFile,
    anchor: "return <Login {...props} />;",
    setup: (s) => s.replace(
      "return <Login {...props} />;",
      `return (<><span data-freshsrv="0" style={{ display: "none" }} /><Login {...props} /></>);`,
    ),
    step: (s, n) => s.replace(/data-freshsrv="\d+"/, `data-freshsrv="${n}"`),
    marker: (n) => `data-freshsrv="${n}"`,
  },
];

// ---------------------------------------------------------------------------

const EDITED_FILES = [islandFile, sharedFile, serverFile, cssFile];
const originals = new Map();

preflight();
process.on("SIGINT", () => { restoreAll(); process.exit(130); });
process.on("SIGTERM", () => { restoreAll(); process.exit(143); });

const report = {
  meta: {
    date: new Date().toISOString(),
    app: APP,
    appCommit: gitHead(APP),
    diffpackCommit: gitHead(repoRoot),
    node: process.version,
    next: nextVersion(),
    port: PORT,
    buildPairs: PAIRS,
    hmrSamples: SAMPLES,
    gapMs: GAP_MS,
    cadenceGapMs: CADENCE_GAP_MS,
    loadAtStart: loadavg().map((x) => Math.round(x * 10) / 10),
  },
  build: {},
  dev: {},
};

let failure = null;
try {
  if (ONLY === "all" || ONLY === "build") await buildSection();
  if (ONLY === "all" || ONLY === "dev") await devSection();
} catch (err) {
  // Keep every measurement taken before the failure: a partial table plus the
  // reason is worth more than a stack trace and nothing.
  failure = err;
  report.meta.error = String(err && err.stack ? err.stack : err);
} finally {
  restoreAll();
}

report.meta.loadAtEnd = loadavg().map((x) => Math.round(x * 10) / 10);
mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(report, null, 2));
printTable(report, OUT);
if (failure) {
  console.error(`\nRUN INCOMPLETE — the table above covers only what was measured before this failure:\n${failure.stack ?? failure}`);
  process.exit(1);
}
process.exit(0);

// ---------------------------------------------------------------------------
// Sections

async function buildSection() {
  banner("PRODUCTION BUILD");
  for (const s of SIDES) report.build[s.key] = { wallMs: [], cpuUserS: [], peakRssTreeMb: [], peakRssSingleMb: [], rssSamples: [] };

  for (let pair = 1; pair <= PAIRS; pair++) {
    for (const side of SIDES) {
      const { cmd, argv, cwd } = side.build();
      for (const dir of side.outDirs) rmSync(dir, { recursive: true, force: true });
      const load0 = loadavg()[0];
      const t0 = Date.now();
      const res = await runTimed(cmd, argv, cwd);
      const wall = Date.now() - t0;
      if (res.code !== 0) throw new Error(`${side.label} build failed (exit ${res.code}); last output:\n${tail(res.output, 3000)}`);
      const r = report.build[side.key];
      r.wallMs.push(wall);
      r.cpuUserS.push(res.userS);
      r.peakRssTreeMb.push(res.treePeakMb);
      r.peakRssSingleMb.push(res.maxRssMb);
      r.rssSamples.push(res.rssSamples);
      console.log(
        `  [${pair}/${PAIRS}] ${side.label.padEnd(9)} wall=${fmtS(wall)} cpu(user)=${res.userS.toFixed(1)}s ` +
        `peakRSS(tree)=${fmtMb(res.treePeakMb)} peakRSS(largest single proc)=${fmtMb(res.maxRssMb)} ` +
        `[${res.rssSamples} samples] (load ${load0.toFixed(1)} -> ${loadavg()[0].toFixed(1)})`,
      );
    }
  }
}

async function devSection() {
  banner("DEV SERVER, HMR AND FRESH-DOCUMENT LATENCY");
  const { chromium } = await loadPlaywright();

  for (const side of SIDES) {
    console.log(`\n---- ${side.label} dev ----`);
    restoreAll(); // every side starts from pristine sources
    for (const dir of side.outDirs) rmSync(dir, { recursive: true, force: true });
    requirePortFree();

    const out = { startup: null, secondRouteMs: null, hmr: {}, fresh: {} };
    const { proc, getLog } = boot(side);
    try {
      const t0 = Date.now();
      await waitFor200(`${HEALTH}/auth/login`, 180000, getLog);
      out.startup = Date.now() - t0;
      console.log(`  cold start -> first 200 on /auth/login: ${fmtS(out.startup)}`);

      const t1 = Date.now();
      await waitFor200(`${HEALTH}/pro/30min`, 180000, getLog);
      out.secondRouteMs = Date.now() - t1;
      console.log(`  second route /pro/30min: ${out.secondRouteMs} ms`);

      out.hmr = await hmrMatrix(chromium, side);
      restoreAll();
      await settleAfterRestore();
      out.fresh = await freshDocMatrix(side);
    } finally {
      killTree(proc);
      await waitPortFree(60000);
    }
    report.dev[side.key] = out;
  }
}

// Edit -> updated DOM, for every edit class. One `page.evaluate` per 10 ms, the
// same detection expression on both sides, so nothing about the poll favours either.
async function hmrMatrix(chromium, side) {
  const results = {};
  const browser = await chromium.launch();
  const context = await browser.newContext({ timezoneId: "Europe/London" });
  await context.addCookies([{ url: BASE, name: "calcom-timezone-dialog", value: "1", expires: -1 }]);
  try {
    for (const k of HMR_KINDS) {
      const page = await context.newPage();
      try {
        await page.goto(`${BASE}${k.url}`, { waitUntil: "domcontentloaded", timeout: 120000 });
        await page.waitForTimeout(4000); // let hydration and the HMR socket settle
        const timeout = k.timeoutMs ?? 45000;
        const waitFor = async (expr) => {
          const t0 = performance.now();
          for (;;) {
            try { if (await page.evaluate(`(() => ${expr})()`)) return performance.now() - t0; } catch {}
            if (performance.now() - t0 > timeout) return null;
            await sleep(10);
          }
        };

        const before = readFileSync(k.file, "utf8");
        if (k.anchor && !before.includes(k.anchor)) {
          throw new Error(`${side.label}/${k.key}: anchor ${JSON.stringify(k.anchor)} not found in ${k.file} — refusing to measure an edit that changes nothing`);
        }
        const setupText = k.setup(before);
        let warmup = null;
        if (setupText !== before) {
          writeFileSync(k.file, setupText);
          warmup = await waitFor(k.setupDetect ? k.setupDetect() : k.detect(0));
          if (warmup === null) throw new Error(`${side.label}/${k.key}: the setup edit never reached the page within ${timeout} ms`);
        } else if (k.setupDetect) {
          const present = await waitFor(k.setupDetect());
          if (present === null) throw new Error(`${side.label}/${k.key}: expected marker from a previous kind is not on the page`);
        }
        await sleep(k.settleMs ?? SETTLE_MS);

        const samples = [];
        for (let n = 1; n <= SAMPLES; n++) {
          writeFileSync(k.file, k.step(readFileSync(k.file, "utf8"), n));
          const dt = await waitFor(k.detect(n));
          samples.push(dt === null ? "TIMEOUT" : Math.round(dt));
          await sleep(k.gapMs ?? GAP_MS);
        }
        results[k.key] = { name: k.name, warmupMs: warmup === null ? null : Math.round(warmup), samples };
        console.log(`  ${k.name}: median ${fmtCell(results[k.key])}  raw [${samples.join(", ")}]${warmup === null ? "" : `  warmup ${Math.round(warmup)} ms`}`);
      } finally {
        await page.close();
      }
    }
  } finally {
    await browser.close();
  }
  return results;
}

// Edit -> a freshly served document contains the marker. No browser: this is the
// curl-equivalent axis, and it is the one that catches an HMR push that updated
// the page while the server kept rendering stale HTML.
async function freshDocMatrix(side) {
  const results = {};
  for (const k of FRESH_KINDS) {
    const url = `${HEALTH}${k.url}`;
    await fetch(url).then((r) => r.text()).catch(() => {});
    const before = readFileSync(k.file, "utf8");
    const setupText = k.setup(before);
    if (setupText === before) throw new Error(`${side.label}/${k.key}: setup edit was a no-op on ${k.file}`);
    writeFileSync(k.file, setupText);
    const warmup = await pollDoc(url, k.marker(0), 180000);
    if (warmup === null) throw new Error(`${side.label}/${k.key}: the setup marker was never served within 180 s`);
    await sleep(GAP_MS);

    const samples = [];
    for (let n = 1; n <= SAMPLES; n++) {
      writeFileSync(k.file, k.step(readFileSync(k.file, "utf8"), n));
      const dt = await pollDoc(url, k.marker(n), 60000);
      samples.push(dt === null ? "TIMEOUT" : Math.round(dt));
      await sleep(GAP_MS);
    }
    results[k.key] = { name: k.name, warmupMs: Math.round(warmup), samples };
    console.log(`  ${k.name}: median ${fmtCell(results[k.key])}  raw [${samples.join(", ")}]`);
  }
  return results;
}

async function pollDoc(url, marker, timeoutMs) {
  const t0 = performance.now();
  for (;;) {
    try {
      const res = await fetch(url, { headers: { accept: "text/html", "cache-control": "no-cache" }, cache: "no-store" });
      const text = await res.text();
      if (text.includes(marker)) return performance.now() - t0;
    } catch {}
    if (performance.now() - t0 > timeoutMs) return null;
    await sleep(25);
  }
}

// ---------------------------------------------------------------------------
// Reporting

function printTable(r, outPath) {
  const rows = [];
  const b = r.build, d = r.dev;
  const bd = b.diffpack, bt = b.turbopack;
  let i = 1;
  const row = (axis, dpCell, tpCell, adv) => rows.push([String(i++), axis, dpCell, tpCell, adv]);
  const section = (title) => rows.push(["section", title, "", "", ""]);
  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

  if (bd && bt) {
    section(`Production build (${plural(PAIRS, "interleaved pair")}, every sample shown)`);
    row(
      "Cold build wall",
      bd.wallMs.map(fmtS).join(" / "), bt.wallMs.map(fmtS).join(" / "),
      ratio(median(bd.wallMs), median(bt.wallMs)),
    );
    row(
      "Cold build CPU (user)",
      bd.cpuUserS.map((x) => `${x.toFixed(1)} s`).join(" / "), bt.cpuUserS.map((x) => `${x.toFixed(1)} s`).join(" / "),
      ratio(median(bd.cpuUserS), median(bt.cpuUserS)),
    );
    // The comparable memory axis: both sides are process trees, so this is the summed
    // RSS of the tree, sampled every 250 ms.
    row(
      "Cold build peak RSS (whole process tree, sampled)",
      bd.peakRssTreeMb.map(fmtMb).join(" / "), bt.peakRssTreeMb.map(fmtMb).join(" / "),
      ratio(median(bd.peakRssTreeMb), median(bt.peakRssTreeMb)),
    );
    // Reported for continuity with `/usr/bin/time -l` and labelled for what it is:
    // ru_maxrss is the largest single process, so it under-reports both trees, and not
    // by the same factor. Not the headline.
    row(
      "…largest single process only (ru_maxrss, under-reports both)",
      bd.peakRssSingleMb.map(fmtMb).join(" / "), bt.peakRssSingleMb.map(fmtMb).join(" / "),
      ratio(median(bd.peakRssSingleMb), median(bt.peakRssSingleMb)),
    );
  }

  const dd = d.diffpack, dt = d.turbopack;
  if (dd && dt) {
    section("Dev server (cold, output wiped)");
    row("Cold start -> first 200", fmtS(dd.startup), fmtS(dt.startup), ratio(dd.startup, dt.startup));
    row("Second route (/pro/30min)", `${dd.secondRouteMs} ms`, `${dt.secondRouteMs} ms`, ratio(dd.secondRouteMs, dt.secondRouteMs));

    section(`HMR: edit -> updated DOM (${plural(SAMPLES, "sample")} each, medians)`);
    for (const k of HMR_KINDS) {
      const a = dd.hmr[k.key], z = dt.hmr[k.key];
      if (!a || !z) continue;
      row(k.name, fmtCell(a), fmtCell(z), ratio(median(ok(a.samples)), median(ok(z.samples))));
    }
    // Three edit classes in one row, so the advantage is three ratios in the same
    // order — a triple has no single multiplier, and leaving the cell blank made it
    // look like the axis had not been compared.
    const WARMUP_CLASSES = ["island", "shared", "server"];
    const warm = (side) => WARMUP_CLASSES.map((k) => side.hmr[k]?.warmupMs ?? null);
    const warmCell = (side) => `${warm(side).map((ms) => ms ?? "n/a").join(" / ")} ms`;
    row(
      "First-edit warmup (island / shared / server)",
      warmCell(dd),
      warmCell(dt),
      WARMUP_CLASSES.map((_, i) => ratio(warm(dd)[i], warm(dt)[i]) || "n/a").join(" / "),
    );

    section(`Edit -> freshly served document (curl-equivalent, ${plural(SAMPLES, "sample")}, medians)`);
    for (const k of FRESH_KINDS) {
      const a = dd.fresh[k.key], z = dt.fresh[k.key];
      if (!a || !z) continue;
      row(k.name, fmtCell(a), fmtCell(z), ratio(median(ok(a.samples)), median(ok(z.samples))));
    }
  }

  const md = [];
  md.push(`| # | Axis | diffpack | Turbopack | Advantage |`);
  md.push(`|---|---|---:|---:|---|`);
  for (const row of rows) {
    if (row[0] === "section") md.push(`| **${row[1]}** | | | | |`);
    else md.push(`| ${row[0]} | ${row[1]} | ${row[2]} | ${row[3]} | ${row[4]} |`);
  }
  const m = r.meta;
  console.log(`\n${"=".repeat(72)}\ncal.com: diffpack vs Turbopack\n${"=".repeat(72)}\n`);
  console.log(md.join("\n"));
  console.log(
    `\napp ${m.app} @ ${m.appCommit}, diffpack @ ${m.diffpackCommit}, Next ${m.next}, Node ${m.node}` +
    `\nload average ${m.loadAtStart[0]} at start, ${m.loadAtEnd[0]} at end. Every raw sample is in the JSON.` +
    `\nfull JSON -> ${outPath}`,
  );
}

function fmtCell(cell) {
  const good = ok(cell.samples);
  const timeouts = cell.samples.length - good.length;
  if (!good.length) return `${cell.samples.length} timeouts/${cell.samples.length}`;
  const med = median(good);
  const base = med >= 1000 ? fmtS(med) : `${med} ms`;
  return timeouts ? `${base} + ${timeouts} timeouts/${cell.samples.length}` : base;
}

// Function declarations, not const arrows: the sections above run under top-level
// await, so anything they call has to be hoisted.
function ok(samples) { return samples.filter((x) => x !== "TIMEOUT"); }
function median(xs) { return xs.length ? [...xs].sort((a, b) => a - b)[Math.floor((xs.length - 1) / 2)] : null; }
// diffpack's advantage. Below 1x means Turbopack won the axis, and it is said
// so in words rather than left as a number the eye reads as a win.
function ratio(a, z) {
  if (!a || !z) return "";
  const v = z / a;
  return v >= 1 ? `${v.toFixed(1)}x` : `${(1 / v).toFixed(1)}x SLOWER`;
}
function fmtS(ms) { return `${(ms / 1000).toFixed(2)} s`; }
function fmtMb(mb) { return mb >= 1024 ? `${(mb / 1024).toFixed(2)} GiB` : `${Math.round(mb)} MiB`; }
function tail(s, n) { return s.length > n ? s.slice(-n) : s; }

// ---------------------------------------------------------------------------
// Process / environment helpers

// Run a build under `/usr/bin/time -l` (macOS: bytes) and return wall-independent
// resource facts. Wall is timed by the caller so it covers spawn to exit exactly.
//
// CPU (user) comes from `/usr/bin/time -l`, which accumulates waited-for descendants,
// so it is a true tree total. PEAK MEMORY does not: `maximum resident set size` is
// `ru_maxrss`, the largest SINGLE process in the tree. Both sides here are trees
// (diffpack's production build runs client, react-server and ssr concurrently; `next
// build` spawns workers), so ru_maxrss under-reports both, by different amounts, and
// comparing the two is comparing each side's biggest process rather than its
// footprint. So the tree is sampled too, and `treePeakMb` is the comparable number.
function runTimed(cmd, argv, cwd) {
  return new Promise((resolve) => {
    const p = spawn("/usr/bin/time", ["-l", cmd, ...argv], { cwd, env: benchEnv(), stdio: ["ignore", "pipe", "pipe"] });
    const rssSampler = sampleTreeRss(p.pid);
    let out = "";
    p.stdout.on("data", (d) => (out += d));
    p.stderr.on("data", (d) => (out += d));
    p.on("close", (code) => {
      const treePeakMb = rssSampler.stop();
      const user = /([0-9.]+)\s+user/.exec(out);
      const rss = /([0-9]+)\s+maximum resident set size/.exec(out);
      if (!user || !rss) {
        resolve({ code: code === 0 ? 1 : code, output: `${out}\n(could not parse /usr/bin/time -l output)`, userS: 0, maxRssMb: 0, treePeakMb: 0, rssSamples: 0 });
        return;
      }
      resolve({
        code,
        output: out,
        userS: Number(user[1]),
        maxRssMb: Number(rss[1]) / (1024 * 1024),
        treePeakMb: treePeakMb ?? 0,
        rssSamples: rssSampler.count(),
      });
    });
  });
}

function boot(side) {
  const { cmd, argv, cwd } = side.dev();
  let log = "";
  const proc = spawn(cmd, argv, { cwd, env: benchEnv(), stdio: ["ignore", "pipe", "pipe"] });
  proc.stdout.on("data", (d) => (log += d));
  proc.stderr.on("data", (d) => (log += d));
  proc.on("exit", (code) => { log += `\n[dev process exited with ${code}]\n`; });
  return { proc, getLog: () => log };
}

function benchEnv() {
  return { ...process.env, FORCE_COLOR: "0", NEXT_TELEMETRY_DISABLED: "1" };
}

async function waitFor200(url, timeoutMs, getLog) {
  const deadline = Date.now() + timeoutMs;
  let shellHits = 0;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(url, { signal: AbortSignal.timeout(30000), redirect: "manual" });
      const body = await r.text().catch(() => "");
      if (r.status === 200) {
        if (body.includes(READY_MARKER) && body.includes("</html>")) {
          if (shellHits) {
            console.log(`    (${url} answered 200 without ${JSON.stringify(READY_MARKER)} ${shellHits}x first; those did not count)`);
          }
          return { shellHits };
        }
        shellHits++;
      }
    } catch {}
    await sleep(20);
  }
  throw new Error(
    `no 200 carrying ${JSON.stringify(READY_MARKER)} from ${url} within ${timeoutMs} ms ` +
    `(${shellHits} bare 200s; a wrong --ready-marker looks exactly like this). Server log tail:\n${tail(getLog(), 4000)}`,
  );
}

// Kill the whole descendant tree: both dev servers outlive a bare kill of the
// parent (diffpack keeps its react-server worker, next keeps next-server).
function killTree(proc) {
  const pid = proc.pid;
  if (!pid) return;
  const all = [];
  const walk = (root) => {
    all.push(root);
    let kids = "";
    try { kids = execFileSync("pgrep", ["-P", String(root)], { encoding: "utf8" }); } catch {}
    for (const line of kids.split("\n")) { const k = Number(line.trim()); if (k) walk(k); }
  };
  walk(pid);
  for (const p of all.reverse()) { try { process.kill(p, "SIGTERM"); } catch {} }
  setTimeout(() => { for (const p of all) { try { process.kill(p, "SIGKILL"); } catch {} } }, 1500);
}

function portOwners() {
  try { return execFileSync("lsof", ["-tnP", `-iTCP:${PORT}`, "-sTCP:LISTEN"], { encoding: "utf8" }).trim().split("\n").filter(Boolean); }
  catch { return []; }
}

function requirePortFree() {
  const owners = portOwners();
  if (owners.length) throw new Error(`port ${PORT} is held by pid(s) ${owners.join(", ")} — refusing to measure against someone else's server`);
}

async function waitPortFree(timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!portOwners().length) return;
    await sleep(200);
  }
  throw new Error(`port ${PORT} still held after teardown by pid(s) ${portOwners().join(", ")}`);
}

// After restoring sources, give the running dev server a moment to process the
// change so the next section's setup edit is not racing a rebuild of the restore.
function settleAfterRestore() { return sleep(GAP_MS); }

// Playwright is resolved from the app's own node_modules (this repo does not
// depend on it). It is CommonJS, so the dynamic import exposes it under
// `default` rather than as named exports.
async function loadPlaywright() {
  const require_ = createRequire(join(APP, "package.json"));
  let entry;
  try { entry = require_.resolve("@playwright/test"); }
  catch { throw new Error(`@playwright/test is not installed in ${APP} — the DOM axis needs it (yarn add -D @playwright/test there)`); }
  const mod = await import(pathToFileURL(entry).href);
  const chromium = mod.chromium ?? mod.default?.chromium;
  if (!chromium) throw new Error(`@playwright/test at ${entry} exposes no chromium export`);
  return { chromium };
}

function preflight() {
  if (!existsSync(diffpackBin)) throw new Error(`missing ${diffpackBin}; run \`cargo build --release\` first`);
  if (!existsSync(WEB)) throw new Error(`missing ${WEB}; pass --app <cal.com checkout>`);
  if (!existsSync(nextBin)) throw new Error(`missing ${nextBin}; install the cal.com checkout's node_modules first`);
  for (const f of EDITED_FILES) {
    if (!existsSync(f)) throw new Error(`missing edit target ${f}`);
    originals.set(f, readFileSync(f, "utf8"));
  }
  for (const k of [...HMR_KINDS, ...FRESH_KINDS]) {
    // Anchors planted by an earlier kind are checked when that kind runs; the rest
    // must exist in pristine source, or the "edit" would change nothing.
    if (!k.anchor || k.plantedAnchor) continue;
    if (!originals.get(k.file).includes(k.anchor)) {
      throw new Error(`edit anchor ${JSON.stringify(k.anchor)} is gone from ${k.file} — refusing to run: the ${k.key} edit would change nothing`);
    }
  }
  requirePortFree();
}

function restoreAll() {
  for (const [f, text] of originals) {
    try { if (readFileSync(f, "utf8") !== text) writeFileSync(f, text); } catch {}
  }
}

function gitHead(dir) {
  try { return execFileSync("git", ["-C", dir, "rev-parse", "--short", "HEAD"], { encoding: "utf8" }).trim(); }
  catch { return "unknown"; }
}

function nextVersion() {
  try { return JSON.parse(readFileSync(join(APP, "node_modules", "next", "package.json"), "utf8")).version; }
  catch { return "unknown"; }
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
function banner(t) { console.log(`\n${"=".repeat(72)}\n${t}\n${"=".repeat(72)}`); }

function parseArgs(a) {
  const o = {};
  for (let i = 0; i < a.length; i++) {
    const k = a[i];
    if (k === "--app") o.app = a[++i];
    else if (k === "--port") o.port = Number(a[++i]);
    else if (k === "--pairs") o.pairs = Number(a[++i]);
    else if (k === "--samples") o.samples = Number(a[++i]);
    else if (k === "--gap") o.gap = Number(a[++i]);
    else if (k === "--settle") o.settle = Number(a[++i]);
    else if (k === "--cadence") o.cadence = Number(a[++i]);
    else if (k === "--only") o.only = a[++i];
    else if (k === "--ready-marker") o.readyMarker = a[++i];
    else if (k === "--out") o.out = a[++i];
    else throw new Error(`unknown arg ${k}`);
  }
  if (o.only && !["all", "build", "dev"].includes(o.only)) throw new Error(`--only must be all|build|dev`);
  return o;
}
