// Dev-server live-reload browser oracle (test-only; NEVER referenced by the
// build). Starts `diffpack dev` (the long-lived, live-rebuild dev server) against
// the pinned app, loads `/` in real headless Chrome, and proves:
//
//   1. `/` is server-rendered (`Welcome Home!!!` in the INITIAL raw HTML, before
//      any client JS) and hydrates clean (window.__TSR_ROUTER__ set, no console/
//      page errors, no server-only leak).
//   2. Editing src/routes/index.tsx's greeting triggers an AUTOMATIC full-page
//      reload (the diffpack-injected SSE client calls location.reload) and the
//      NEW text is now server-rendered AND still hydrates clean.
//   3. The dev loop's incremental instrumentation proves the edit re-transformed
//      exactly ONE client module and the incremental emit re-rendered exactly ONE
//      client chunk from the long-lived process — the incremental-emit thesis
//      guard exercised live.
//
// The source file is always restored afterward. Node/Chrome are test oracles
// only; the build path is native Rust.
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { readFileSync, writeFileSync } from "node:fs";
import puppeteer from "puppeteer-core";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, "..", "..");
const DIFFPACK = join(repoRoot, "target", "release", "diffpack");
// Chrome discovery: explicit override, then the machines this repo runs on
// (Linux box: Playwright-cached Chromium; Mac: system Chrome).
import { existsSync as __chromeExists } from "node:fs";
const CHROME = [
  process.env.CHROME,
  `${process.env.HOME}/.cache/ms-playwright/chromium-1194/chrome-linux/chrome`,
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].filter(Boolean).find((p) => __chromeExists(p));
const PORT = 8971;
const BASE = `http://127.0.0.1:${PORT}`;
const INDEX_TSX = join(here, "src", "routes", "index.tsx");

const ORIGINAL_GREETING = "Welcome Home!!!";
const NEW_GREETING = `Welcome Home LIVE-RELOAD ${Date.now()}`;
const originalSource = readFileSync(INDEX_TSX, "utf8");
if (!originalSource.includes(ORIGINAL_GREETING)) {
  console.error(`index.tsx does not contain the expected greeting ${JSON.stringify(ORIGINAL_GREETING)}; refusing to edit`);
  process.exit(2);
}

const results = [];
const record = (name, ok, detail) => results.push({ name, ok, detail });

// --- boot the diffpack dev server, capturing stdout for the instrumentation ---
let devLog = "";
const dev = spawn(DIFFPACK, ["dev", ".", String(PORT)], {
  cwd: here,
  stdio: ["ignore", "pipe", "pipe"],
});
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

// Wait until a NEW rebuild-instrumentation line appears (after `sinceLen` bytes
// of log), and return its parsed client transformed/rendered counts.
function waitForRebuild(sinceLen, timeoutMs) {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + timeoutMs;
    const tick = () => {
      const fresh = devLog.slice(sinceLen);
      const m = fresh.match(/client transformed=(\d+) changed=(\d+) rendered_chunks=(\d+) \| server transformed=(\d+) changed=(\d+) rendered_chunks=(\d+)/);
      if (m) {
        return resolve({
          clientTransformed: Number(m[1]),
          clientChanged: Number(m[2]),
          clientRendered: Number(m[3]),
          serverTransformed: Number(m[4]),
          serverChanged: Number(m[5]),
          serverRendered: Number(m[6]),
          line: fresh.split("\n").find((l) => l.includes("rebuilt")),
        });
      }
      if (Date.now() > deadline) return reject(new Error("no rebuild instrumentation within timeout:\n" + fresh));
      setTimeout(tick, 150);
    };
    tick();
  });
}

const classify = (text) => {
  if (/module is not defined/i.test(text)) return "moduleNotDefined";
  if (/node builtin node:async_hooks/i.test(text) || /No Start context found/i.test(text) || /async_hooks/i.test(text)) return "serverLeak";
  return "js";
};

// Instrument a page: collect JS errors and server-only leaks (ignoring the known
// framework CSS/tailwind gap, consistent with the other oracles).
function instrument(page) {
  const jsErrors = [];
  const serverLeaks = [];
  page.on("console", (m) => {
    if (m.type() !== "error") return;
    const text = m.text();
    if (/tailwindcss/i.test(text) || /Failed to load resource/i.test(text)) return;
    const k = classify(text);
    if (k === "serverLeak") serverLeaks.push(text);
    else jsErrors.push(text);
  });
  page.on("pageerror", (e) => {
    const text = String(e.message || e);
    const k = classify(text);
    if (k === "serverLeak") serverLeaks.push(text);
    else jsErrors.push(text);
  });
  return { jsErrors, serverLeaks };
}

async function loadAndHydrate(page, greeting) {
  // Raw server HTML BEFORE any client JS.
  const rawHtml = await (await fetch(BASE + "/")).text();
  // Not networkidle0: the SSE reload channel is a persistent connection, so the
  // page never reaches 0 in-flight requests. `load` + an explicit hydration wait
  // is the correct signal here.
  await page.goto(BASE + "/", { waitUntil: "load", timeout: 25000 });
  const hydrated = await page
    .waitForFunction(() => window.__TSR_ROUTER__ !== undefined || window.$_TSR === undefined, { timeout: 12000 })
    .then(() => true)
    .catch(() => false);
  const routerPresent = await page.evaluate(() => typeof window.__TSR_ROUTER__ !== "undefined");
  const h3 = await page.$eval("h3", (el) => el.textContent).catch(() => null);
  return { rawHtml, hydrated, routerPresent, h3, ssrHasGreeting: rawHtml.includes(greeting) };
}

let browser;
try {
  await waitForServer(60000);

  browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: true,
    args: ["--no-sandbox", "--disable-gpu"],
  });
  const page = await browser.newPage();
  const signals = instrument(page);

  // === Phase 1: initial SSR + hydration ===
  const initial = await loadAndHydrate(page, ORIGINAL_GREETING);
  record("initial: `Welcome Home!!!` server-rendered in raw HTML", initial.ssrHasGreeting, `present=${initial.ssrHasGreeting}`);
  record("initial: hydrated (__TSR_ROUTER__ set)", initial.hydrated && initial.routerPresent, `hydrated=${initial.hydrated}, router=${initial.routerPresent}, h3=${JSON.stringify(initial.h3)}`);
  record("initial: diffpack live-reload client injected", /EventSource\(\"\/__diffpack_dev\/events\"\)/.test(initial.rawHtml), "SSE client present in served HTML");

  // Tag the live window; a full-page reload discards it, proving the reload was a
  // real navigation (not a client-side state mutation).
  await page.evaluate(() => (window.__dev_reload_probe__ = "before-edit"));

  // === Phase 2: edit the greeting, await the automatic reload ===
  const logLenBeforeEdit = devLog.length;
  const edited = originalSource.replace(ORIGINAL_GREETING, NEW_GREETING);
  writeFileSync(INDEX_TSX, edited);

  // The dev loop rebuilds and prints its instrumentation; assert incrementality.
  let rebuild;
  try {
    rebuild = await waitForRebuild(logLenBeforeEdit, 30000);
    record(
      "edit changed exactly ONE client module (live incremental)",
      rebuild.clientChanged === 1,
      `client changed=${rebuild.clientChanged}, transformed=${rebuild.clientTransformed} (${rebuild.line})`,
    );
    record(
      "incremental emit re-rendered exactly ONE client chunk (live)",
      rebuild.clientRendered === 1,
      `client rendered_chunks=${rebuild.clientRendered}`,
    );
  } catch (e) {
    record("edit changed exactly ONE client module (live incremental)", false, String(e.message || e));
    record("incremental emit re-rendered exactly ONE client chunk (live)", false, "no instrumentation");
  }

  // The injected SSE client should have reloaded the page to the new greeting.
  const reloaded = await page
    .waitForFunction((g) => document.querySelector("h3")?.textContent === g, { timeout: 20000 }, NEW_GREETING)
    .then(() => true)
    .catch(() => false);
  const probeCleared = await page.evaluate(() => window.__dev_reload_probe__ === undefined);
  record("automatic full-page reload fired (window probe cleared)", reloaded && probeCleared, `reloaded=${reloaded}, probeCleared=${probeCleared}`);

  // === Phase 3: the NEW text is server-rendered AND hydrates clean ===
  const after = await loadAndHydrate(page, NEW_GREETING);
  record("after edit: NEW greeting server-rendered in raw HTML", after.ssrHasGreeting, `h3=${JSON.stringify(after.h3)}, ssrHasNew=${after.ssrHasGreeting}`);
  record("after edit: still hydrates (__TSR_ROUTER__ set)", after.hydrated && after.routerPresent, `hydrated=${after.hydrated}, router=${after.routerPresent}`);

  // === Cross-cutting: zero JS errors / server leaks across the whole run ===
  record("no uncaught JS errors across load+edit+reload", signals.jsErrors.length === 0, JSON.stringify(signals.jsErrors));
  record("no server-only leak (async_hooks / Start context)", signals.serverLeaks.length === 0, JSON.stringify(signals.serverLeaks));
} catch (e) {
  record("harness", false, String(e.stack || e));
} finally {
  // ALWAYS restore the source file.
  writeFileSync(INDEX_TSX, originalSource);
  if (browser) await browser.close().catch(() => {});
  dev.kill("SIGKILL");
}

console.log("\n=== dev-server live-reload gates ===");
let pass = 0;
for (const r of results) {
  console.log(`${r.ok ? "PASS" : "FAIL"} ${r.name}: ${r.detail}`);
  if (r.ok) pass++;
}
console.log(`\n${pass}/${results.length} dev-server gates passed`);
process.exit(pass === results.length ? 0 : 1);
