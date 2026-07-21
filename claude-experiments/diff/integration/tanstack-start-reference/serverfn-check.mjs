// Server-function data-route oracle (test-only; NEVER referenced by the build).
// Boots the emitted server and proves the data routes work end to end in a real
// browser: /posts and /users on DIRECT load (SSR path) and on client-side SPA
// navigation (the server-fn HTTP RPC path). Reports DOM text + console for each.
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import puppeteer from "puppeteer-core";

const here = dirname(fileURLToPath(import.meta.url));
const outputRoot = join(here, ".diffpack-output");
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
const PORT = 8710;
const BASE = `http://127.0.0.1:${PORT}`;

function waitForServer() {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error("server did not listen")), 15000);
    const tick = async () => {
      try {
        const r = await fetch(BASE + "/");
        if (r.ok) { clearTimeout(t); return resolve(); }
      } catch {}
      setTimeout(tick, 200);
    };
    tick();
  });
}

let serverLog = "";
const server = spawn(process.execPath, [join(outputRoot, "server/index.mjs")], {
  env: { ...process.env, PORT: String(PORT), HOST: "127.0.0.1" },
  stdio: ["ignore", "pipe", "pipe"],
});
server.stdout.on("data", (d) => (serverLog += d));
server.stderr.on("data", (d) => (serverLog += d));

const results = [];
const record = (name, ok, detail) => results.push({ name, ok, detail });

// Attach console/error capture to a page, returning the collected arrays.
function instrument(page) {
  const errors = [];
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push(String(e.message || e)));
  return errors;
}

let browser;
try {
  await waitForServer();
  browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: true,
    args: ["--no-sandbox", "--disable-gpu"],
  });

  // --- 1. DIRECT LOAD (SSR path) of /posts ---
  {
    const page = await browser.newPage();
    const errors = instrument(page);
    await page.goto(BASE + "/posts", { waitUntil: "networkidle0", timeout: 20000 });
    const items = await page.$$eval("ul.list-disc li div", (els) => els.map((e) => e.textContent));
    const html = await page.content();
    const iterableErr = /is not iterable/.test(html) || errors.some((e) => /is not iterable/.test(e));
    const dataErrs = errors.filter((e) => !/tailwindcss|Failed to load resource/i.test(e));
    record("SSR /posts renders posts list", items.length >= 10 && !iterableErr, `${items.length} items, first=${JSON.stringify(items[0])}`);
    record("SSR /posts no data errors", !iterableErr && dataErrs.length === 0, JSON.stringify(dataErrs));
    await page.close();
  }

  // --- 2. DIRECT LOAD (SSR path) of /users ---
  {
    const page = await browser.newPage();
    const errors = instrument(page);
    await page.goto(BASE + "/users", { waitUntil: "networkidle0", timeout: 20000 });
    const items = await page.$$eval("ul.list-disc li div", (els) => els.map((e) => e.textContent));
    const html = await page.content();
    const iterableErr = /is not iterable/.test(html) || errors.some((e) => /is not iterable/.test(e));
    const dataErrs = errors.filter((e) => !/tailwindcss|Failed to load resource/i.test(e));
    record("SSR /users renders users list", items.length >= 10 && !iterableErr, `${items.length} items, first=${JSON.stringify(items[0])}`);
    record("SSR /users no data errors", !iterableErr && dataErrs.length === 0, JSON.stringify(dataErrs));
    await page.close();
  }

  // --- 3. SPA navigation / -> /posts -> /users (server-fn HTTP RPC path) ---
  {
    const page = await browser.newPage();
    const errors = instrument(page);
    await page.goto(BASE + "/", { waitUntil: "networkidle0", timeout: 20000 });
    await page.waitForFunction(() => window.__TSR_ROUTER__ !== undefined || window.$_TSR === undefined, { timeout: 10000 }).catch(() => {});
    await page.evaluate(() => (window.__nav_probe__ = "spa"));

    // -> /posts (loader runs on the client, calls the server fn over HTTP)
    await page.click('a[href="/posts"]').catch(() => {});
    await page.waitForFunction(() => location.pathname === "/posts", { timeout: 10000 }).catch(() => {});
    await page.waitForFunction(() => document.querySelectorAll("ul.list-disc li div").length >= 10, { timeout: 10000 }).catch(() => {});
    const postsItems = await page.$$eval("ul.list-disc li div", (els) => els.map((e) => e.textContent));
    const spaSurvived = await page.evaluate(() => window.__nav_probe__ === "spa");

    // -> /users (relative fetch to the /api/users route via the server)
    await page.click('a[href="/users"]').catch(() => {});
    await page.waitForFunction(() => location.pathname === "/users", { timeout: 10000 }).catch(() => {});
    await page.waitForFunction(() => document.querySelectorAll("ul.list-disc li div").length >= 10, { timeout: 10000 }).catch(() => {});
    const usersItems = await page.$$eval("ul.list-disc li div", (els) => els.map((e) => e.textContent));

    const dataErrs = errors.filter((e) => !/tailwindcss|Failed to load resource/i.test(e));
    record("SPA nav to /posts renders data (server-fn RPC)", postsItems.length >= 10 && spaSurvived, `${postsItems.length} items, first=${JSON.stringify(postsItems[0])}, spa=${spaSurvived}`);
    record("SPA nav to /users renders data", usersItems.length >= 10, `${usersItems.length} items, first=${JSON.stringify(usersItems[0])}`);
    record("SPA nav: no data/JS errors", dataErrs.length === 0, JSON.stringify(dataErrs.slice(0, 4)));
    await page.close();
  }
} catch (e) {
  record("harness", false, String(e.stack || e));
} finally {
  if (browser) await browser.close().catch(() => {});
  server.kill("SIGKILL");
}

console.log("\n=== server-function data-route gates ===");
let pass = 0;
for (const r of results) {
  console.log(`${r.ok ? "PASS" : "FAIL"} ${r.name}: ${r.detail}`);
  if (r.ok) pass++;
}
console.log(`\n${pass}/${results.length} server-fn gates passed`);
process.exit(pass === results.length ? 0 : 1);
