// SPA dev-server HMR browser oracle (test-only; NEVER referenced by the build).
// Starts `diffpack dev` against this plain Vite React SPA (HTML entry, no SSR),
// loads `/` in real headless Chrome, and proves the STATE-PRESERVING hot-update
// workflow that diffpack's low diff times make a daily advantage:
//
//   1. The SPA is served and mounts: React renders (`Get started` <h1>), the
//      counter starts at 0 and updates on click, with zero console errors.
//   2. The diffpack-injected client is the WebSocket HMR + React Fast Refresh
//      preamble.
//   3. After clicking the counter to a known value, editing src/App.tsx's heading
//      produces a STATE-PRESERVING hot update via React Fast Refresh: the new
//      heading swaps into the SAME live document (a page-scoped window probe and
//      the very same <h1> DOM node survive) AND the counter's useState value is
//      PRESERVED across the update — the crown-jewel Fast Refresh guarantee. It is
//      NOT a navigation and NOT a remount.
//   4. The dev loop's incremental instrumentation proves the edit re-transformed
//      exactly ONE module, re-rendered exactly ONE chunk, and landed under the
//      low-diff-time budget — the incremental thesis, exercised live on a real
//      Vite SPA with no SSR server in the loop at all.
//
// The source file is always restored afterward. Node/Chrome are test oracles
// only; the build path is native Rust.
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { readFileSync, writeFileSync, existsSync, rmSync } from "node:fs";
import puppeteer from "puppeteer-core";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, "..", "..");
const DIFFPACK = join(repoRoot, "target", "release", "diffpack");
const CHROME = [
  process.env.CHROME,
  `${process.env.HOME}/.cache/ms-playwright/chromium-1194/chrome-linux/chrome`,
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].filter(Boolean).find((p) => existsSync(p));
const PORT = 8972;
const BASE = `http://127.0.0.1:${PORT}`;
const APP_TSX = join(here, "src", "App.tsx");

const ORIGINAL_HEADING = "Get started";
const NEW_HEADING = `Get started HMR ${Date.now()}`;
// The low-diff-time budget: a leaf edit's whole incremental rebuild (re-transform
// + emit) must land well under this. Observed ~6ms on the dev box; the ceiling is
// generous for a loaded CI host while still an order of magnitude under a build.
const DIFF_TIME_BUDGET_MS = 250;
const CLICKS = 3;
const EXTRA_TSX = join(here, "src", "Extra.tsx");
const EXTRA_MARKER = `EXTRA-NEW-FILE-OK-${Date.now()}`;
const APP_CSS = join(here, "src", "App.css");

const originalSource = readFileSync(APP_TSX, "utf8");
const originalCss = readFileSync(APP_CSS, "utf8");
if (!originalCss.includes("border-radius: 5px;")) {
  console.error("App.css does not contain the expected `border-radius: 5px;`; refusing to edit");
  process.exit(2);
}
if (!originalSource.includes(`<h1>${ORIGINAL_HEADING}</h1>`)) {
  console.error(`App.tsx does not contain <h1>${ORIGINAL_HEADING}</h1>; refusing to edit`);
  process.exit(2);
}

const results = [];
const record = (name, ok, detail) => results.push({ name, ok, detail });

let devLog = "";
const dev = spawn(DIFFPACK, ["dev", ".", String(PORT)], { cwd: here, stdio: ["ignore", "pipe", "pipe"] });
dev.stdout.on("data", (d) => (devLog += d));
dev.stderr.on("data", (d) => (devLog += d));

function waitForServer(timeoutMs) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error("dev server did not come up:\n" + devLog)), timeoutMs);
    const tick = async () => {
      try {
        const r = await fetch(BASE + "/");
        if (r.ok) { clearTimeout(t); return resolve(); }
      } catch {}
      setTimeout(tick, 250);
    };
    tick();
  });
}

// Wait for a NEW rebuild-instrumentation line (after `sinceLen` bytes of log) and
// parse its client counters + elapsed time. The SPA path has no server segment.
function waitForRebuild(sinceLen, timeoutMs) {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + timeoutMs;
    const tick = () => {
      const fresh = devLog.slice(sinceLen);
      const m = fresh.match(/rebuilt \d+ file\(s\) in ([\d.]+)ms \| client transformed=(\d+) changed=(\d+) rendered_chunks=(\d+) \| (.+)/);
      if (m) {
        return resolve({
          elapsedMs: Number(m[1]),
          clientTransformed: Number(m[2]),
          clientChanged: Number(m[3]),
          clientRendered: Number(m[4]),
          clientNote: m[5].split("\n")[0].trim(),
          line: fresh.split("\n").find((l) => l.includes("rebuilt")),
        });
      }
      if (Date.now() > deadline) return reject(new Error("no rebuild instrumentation within timeout:\n" + fresh));
      setTimeout(tick, 150);
    };
    tick();
  });
}

let browser;
try {
  await waitForServer(60000);

  browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ["--no-sandbox", "--disable-gpu"] });
  const page = await browser.newPage();
  const jsErrors = [];
  // A missing static asset (favicon.svg / icons.svg the fixture references but does
  // not ship — a 404 under `vite dev` too) is a fixture-completeness matter, not an
  // HMR/JS-execution error; ignore it as the TanStack oracle does.
  const ignorable = (text) => /Failed to load resource/i.test(text) || /favicon|icons\.svg/i.test(text);
  page.on("console", (m) => { if (m.type() === "error" && !ignorable(m.text())) jsErrors.push(m.text()); });
  page.on("pageerror", (e) => { const t = String(e.message || e); if (!ignorable(t)) jsErrors.push(t); });

  // === Phase 1: served, mounts, interactive ===
  const rawHtml = await (await fetch(BASE + "/")).text();
  await page.goto(BASE + "/", { waitUntil: "load", timeout: 25000 });
  const heading = await page.$eval("h1", (el) => el.textContent).catch(() => null);
  record("initial: React mounted (<h1> rendered)", heading === ORIGINAL_HEADING, `h1=${JSON.stringify(heading)}`);
  const startCount = await page.$eval("button.counter", (el) => el.textContent.trim()).catch(() => null);
  record("initial: counter starts at 0", startCount === "Count is 0", `button=${JSON.stringify(startCount)}`);
  record(
    "initial: diffpack WebSocket HMR + Fast Refresh client injected",
    /new WebSocket\(/.test(rawHtml) && rawHtml.includes("/__diffpack_hmr/ws") && rawHtml.includes("$RefreshRuntime$"),
    "WebSocket HMR client present in served HTML",
  );

  // Drive the counter to a known non-zero value; tag the window and grab the live
  // <h1> node. All three must survive a state-preserving update.
  for (let i = 0; i < CLICKS; i++) await page.click("button.counter");
  const beforeCount = await page.$eval("button.counter", (el) => el.textContent.trim()).catch(() => null);
  record(`counter advanced to ${CLICKS} on click`, beforeCount === `Count is ${CLICKS}`, `button=${JSON.stringify(beforeCount)}`);
  await page.evaluate(() => (window.__hmr_probe__ = "before-edit"));
  const h1Before = await page.$("h1");

  // === Phase 2: edit the heading, await the state-preserving hot update ===
  const logLenBeforeEdit = devLog.length;
  writeFileSync(APP_TSX, originalSource.replace(`<h1>${ORIGINAL_HEADING}</h1>`, `<h1>${NEW_HEADING}</h1>`));

  try {
    const rebuild = await waitForRebuild(logLenBeforeEdit, 30000);
    record("edit changed exactly ONE module (live incremental)", rebuild.clientChanged === 1, `client changed=${rebuild.clientChanged}, transformed=${rebuild.clientTransformed} (${rebuild.line})`);
    record("incremental emit re-rendered exactly ONE chunk (live)", rebuild.clientRendered === 1, `client rendered_chunks=${rebuild.clientRendered}`);
    record(`rebuild under low-diff-time budget (<${DIFF_TIME_BUDGET_MS}ms)`, rebuild.elapsedMs < DIFF_TIME_BUDGET_MS, `elapsed=${rebuild.elapsedMs}ms`);
  } catch (e) {
    record("edit changed exactly ONE module (live incremental)", false, String(e.message || e));
    record("incremental emit re-rendered exactly ONE chunk (live)", false, "no instrumentation");
    record(`rebuild under low-diff-time budget (<${DIFF_TIME_BUDGET_MS}ms)`, false, "no instrumentation");
  }

  // The Fast Refresh update swaps the new heading into the live tree.
  const updated = await page
    .waitForFunction((g) => document.querySelector("h1")?.textContent === g, { timeout: 20000 }, NEW_HEADING)
    .then(() => true)
    .catch(() => false);
  record("Fast Refresh applied (new heading rendered)", updated, `updated=${updated}`);

  // Crown jewel: hook state preserved. The counter keeps its value across the
  // update, the page-scoped probe survives, and the SAME <h1> node was updated in
  // place (a reload/remount would reset the count and detach the node).
  const afterCount = await page.$eval("button.counter", (el) => el.textContent.trim()).catch(() => null);
  const probeSurvived = await page.evaluate(() => window.__hmr_probe__ === "before-edit");
  const sameNodeUpdated = await h1Before
    .evaluate((el, g) => el.isConnected && el.textContent === g, NEW_HEADING)
    .catch(() => false);
  record(
    `state PRESERVED across hot update (counter stays ${CLICKS}, no remount/reload)`,
    afterCount === `Count is ${CLICKS}` && probeSurvived && sameNodeUpdated,
    `count=${JSON.stringify(afterCount)}, probeSurvived=${probeSurvived}, sameNodeUpdated=${sameNodeUpdated}`,
  );

  // === Phase 2b: CSS hot-swap without reload (state preserved) ===
  // Editing App.css swaps the stylesheet in place: the counter's computed
  // border-radius changes, the useState value stays put, and the page never
  // reloads (the window probe survives).
  const radiusBefore = await page.$eval("button.counter", (el) => getComputedStyle(el).borderRadius).catch(() => null);
  writeFileSync(APP_CSS, originalCss.replace("border-radius: 5px;", "border-radius: 21px;"));
  const cssSwapped = await page
    .waitForFunction(() => getComputedStyle(document.querySelector("button.counter")).borderRadius === "21px", { timeout: 20000 })
    .then(() => true)
    .catch(() => false);
  const countAfterCss = await page.$eval("button.counter", (el) => el.textContent.trim()).catch(() => null);
  const probeAfterCss = await page.evaluate(() => window.__hmr_probe__ === "before-edit");
  record(
    "CSS hot-swap applies without reload (style changes, state preserved)",
    cssSwapped && countAfterCss === `Count is ${CLICKS}` && probeAfterCss,
    `radius ${JSON.stringify(radiusBefore)}->21px swapped=${cssSwapped}, count=${JSON.stringify(countAfterCss)}, probe=${probeAfterCss}`,
  );
  writeFileSync(APP_CSS, originalCss);

  // === Phase 3: adding a NEW file, then importing it, works (no server crash) ===
  // Create a brand-new component the graph never reached, then edit the (now
  // restored) App.tsx to import and render it. The dev server must stay alive and
  // the new component must appear — the everyday "scaffold a component" flow.
  writeFileSync(APP_TSX, originalSource);
  writeFileSync(EXTRA_TSX, `export function Extra() {\n  return <p data-testid="extra">${EXTRA_MARKER}</p>\n}\n`);
  await new Promise((r) => setTimeout(r, 1200));
  const withImport = originalSource
    .replace("import { useState } from 'react'", "import { useState } from 'react'\nimport { Extra } from './Extra'")
    .replace("<h1>Get started</h1>", "<h1>Get started</h1><Extra />");
  writeFileSync(APP_TSX, withImport);
  const newFileRendered = await page
    .waitForFunction((m) => document.querySelector("[data-testid=extra]")?.textContent === m, { timeout: 20000 }, EXTRA_MARKER)
    .then(() => true)
    .catch(() => false);
  const serverAlive = dev.exitCode === null && !dev.killed;
  record("adding a new file + importing it works (component renders, server alive)", newFileRendered && serverAlive, `rendered=${newFileRendered}, serverAlive=${serverAlive}`);

  record("no uncaught JS errors across load+edit+update+new-file", jsErrors.length === 0, JSON.stringify(jsErrors.slice(0, 3)));
} catch (e) {
  record("harness", false, String(e.stack || e));
} finally {
  writeFileSync(APP_TSX, originalSource);
  writeFileSync(APP_CSS, originalCss);
  rmSync(EXTRA_TSX, { force: true });
  if (browser) await browser.close().catch(() => {});
  dev.kill("SIGKILL");
}

console.log("\n=== SPA dev-server HMR gates ===");
let pass = 0;
for (const r of results) {
  console.log(`${r.ok ? "PASS" : "FAIL"} ${r.name}: ${r.detail}`);
  if (r.ok) pass++;
}
console.log(`\n${pass}/${results.length} SPA dev-server HMR gates passed`);
process.exit(pass === results.length ? 0 : 1);
