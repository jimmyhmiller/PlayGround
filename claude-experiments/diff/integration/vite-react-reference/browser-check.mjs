// Browser oracle for the pinned create-vite app (test-only; NEVER part of the
// build). Serves the chosen dist statically, loads it in real headless
// Chromium, and proves the app actually runs: React mounts, the counter's
// state updates on click, every image loads, and there are zero console
// errors and zero failed requests.
//
//   node browser-check.mjs <reference|diffpack>
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { existsSync } from "node:fs";
import puppeteer from "puppeteer-core";
import { startStaticServer } from "./static-server.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const which = process.argv[2];
if (which !== "reference" && which !== "diffpack") {
  console.error("usage: node browser-check.mjs <reference|diffpack>");
  process.exit(2);
}
const dist = join(here, which === "reference" ? "dist" : "dist-diffpack");

// Chrome discovery: explicit override, then the machines this repo runs on
// (Linux box uses the Playwright-cached Chromium; the Mac uses system Chrome).
const candidates = [
  process.env.CHROME,
  `${process.env.HOME}/.cache/ms-playwright/chromium-1194/chrome-linux/chrome`,
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].filter(Boolean);
const chrome = candidates.find((path) => existsSync(path));
if (!chrome) {
  console.error("no Chrome/Chromium found; set CHROME=/path/to/chrome");
  process.exit(2);
}

const results = [];
const record = (name, ok, detail = "") => {
  results.push({ name, ok });
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
};

const server = await startStaticServer(dist, 0);
let browser;
try {
  browser = await puppeteer.launch({
    executablePath: chrome,
    headless: true,
    args: ["--no-sandbox", "--disable-gpu"],
  });
  const page = await browser.newPage();
  const consoleErrors = [];
  const failedRequests = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });
  page.on("pageerror", (error) => consoleErrors.push(String(error)));
  page.on("requestfailed", (req) => failedRequests.push(`${req.url()} ${req.failure()?.errorText}`));
  page.on("response", (res) => {
    if (res.status() >= 400) failedRequests.push(`${res.url()} -> ${res.status()}`);
  });

  await page.goto(`http://127.0.0.1:${server.port}/`, { waitUntil: "networkidle0", timeout: 20000 });

  const heading = await page.$eval("h1", (el) => el.textContent).catch(() => null);
  record("React mounted (h1 rendered)", heading === "Get started", String(heading));

  const buttonText = await page.$eval("button.counter", (el) => el.textContent.trim()).catch(() => null);
  record("counter starts at 0", buttonText === "Count is 0", String(buttonText));

  await page.click("button.counter");
  await page.click("button.counter");
  const afterClicks = await page.$eval("button.counter", (el) => el.textContent.trim()).catch(() => null);
  record("counter state updates on click", afterClicks === "Count is 2", String(afterClicks));

  const images = await page.$$eval("img", (els) =>
    els.map((el) => ({ src: el.getAttribute("src"), ok: el.complete && el.naturalWidth > 0 }))
  );
  const brokenImages = images.filter((img) => !img.ok);
  record(
    `all ${images.length} images load`,
    images.length >= 3 && brokenImages.length === 0,
    brokenImages.map((img) => img.src).join(", ")
  );

  // App.css sets `.counter { border-radius: 5px; background: var(--accent-bg) }`;
  // a computed 5px radius + non-transparent background proves the stylesheet
  // loaded AND its CSS custom properties resolved.
  const styled = await page.$eval("button.counter", (el) => {
    const style = getComputedStyle(el);
    return { radius: style.borderRadius, background: style.backgroundColor };
  });
  record(
    "stylesheet applied (counter styled via CSS variables)",
    styled.radius === "5px" && styled.background !== "rgba(0, 0, 0, 0)",
    JSON.stringify(styled)
  );

  record("zero console errors", consoleErrors.length === 0, consoleErrors.slice(0, 3).join(" | "));
  record("zero failed requests", failedRequests.length === 0, failedRequests.slice(0, 3).join(" | "));
} finally {
  await browser?.close();
  server.close();
}

const passed = results.filter((r) => r.ok).length;
console.log(`\n${which}: ${passed}/${results.length} browser gates passed`);
process.exit(passed === results.length ? 0 : 1);
