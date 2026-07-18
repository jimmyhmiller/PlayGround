// Browser Tailwind oracle (test-only; NEVER referenced by the build).
// Boots the emitted `.diffpack-output/server/index.mjs`, loads `/` in headless
// Chrome, and proves the natively compiled `app.css` actually styles the app:
//   - the stylesheet <link> resolves 200 with `text/css` and no `@import`
//     survives (the `/assets/tailwindcss` 404 is GONE);
//   - real utilities are applied: body background/text colors match the theme's
//     gray scale in light AND dark (prefers-color-scheme), `.font-black` renders
//     at weight 900, and the spacing scale drives `gap`.
// Expected colors are read from a probe element that resolves the same theme
// token, so nothing is hardcoded: the assertion is "the body got the gray-50
// token", proven by matching the browser's own oklch->rgb conversion.
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import puppeteer from "puppeteer-core";

const here = dirname(fileURLToPath(import.meta.url));
const outputRoot = join(here, ".diffpack-output");
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 8579;
const BASE = `http://127.0.0.1:${PORT}`;

function waitForServer(log) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error("server did not listen: " + log().trim())), 15000);
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

let browser;
try {
  await waitForServer(() => serverLog);

  browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: true,
    args: ["--no-sandbox", "--disable-gpu"],
  });
  const page = await browser.newPage();

  const tailwindRequestFailures = [];
  page.on("requestfailed", (r) => {
    if (/tailwindcss/i.test(r.url())) tailwindRequestFailures.push(`${r.url()} ${r.failure()?.errorText}`);
  });
  page.on("response", (r) => {
    if (/\/assets\/tailwindcss/i.test(r.url()) && r.status() >= 400)
      tailwindRequestFailures.push(`${r.url()} -> HTTP ${r.status()}`);
  });

  await page.goto(BASE + "/", { waitUntil: "networkidle0", timeout: 20000 });

  // 1. The stylesheet link exists and resolves 200 text/css.
  const href = await page.$eval("link[rel=stylesheet]", (el) => el.href).catch(() => null);
  let cssStatus = 0, cssType = "", cssBody = "";
  if (href) {
    const r = await fetch(href);
    cssStatus = r.status;
    cssType = r.headers.get("content-type") || "";
    cssBody = await r.text();
  }
  record("stylesheet link resolves 200 text/css", cssStatus === 200 && /css/.test(cssType), `href=${href} status=${cssStatus} type=${cssType}`);

  // 2. No fetchable `@import` survives (the raw-copy bug), and no tailwindcss 404.
  record("no `@import` / fetchable tailwindcss in served CSS", !/@import/i.test(cssBody), `len=${cssBody.length}`);
  record("no /assets/tailwindcss 404 during load", tailwindRequestFailures.length === 0, JSON.stringify(tailwindRequestFailures));

  // Helper: resolve what the browser computes for a theme token, via a probe.
  const probeColor = (prop, value) =>
    page.evaluate(({ prop, value }) => {
      const el = document.createElement("div");
      el.style.setProperty(prop, value);
      document.body.appendChild(el);
      const got = getComputedStyle(el)[prop === "background-color" ? "backgroundColor" : "color"];
      el.remove();
      return got;
    }, { prop, value });

  const bodyStyle = () => page.evaluate(() => {
    const s = getComputedStyle(document.body);
    return { bg: s.backgroundColor, color: s.color };
  });

  // 3. Light-mode body colors == gray-50 background / gray-900 text.
  await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "light" }]);
  let expected = {
    bg: await probeColor("background-color", "oklch(98.5% 0.002 247.839)"),  // gray-50
    text: await probeColor("color", "oklch(21% 0.034 264.665)"),            // gray-900
  };
  let body = await bodyStyle();
  record("light: body background == gray-50", body.bg === expected.bg && body.bg !== "rgba(0, 0, 0, 0)", `body=${body.bg} expected=${expected.bg}`);
  record("light: body text == gray-900", body.color === expected.text, `body=${body.color} expected=${expected.text}`);

  // 4. Dark-mode body colors == gray-950 background / gray-200 text.
  await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "dark" }]);
  let expectedDark = {
    bg: await probeColor("background-color", "oklch(13% 0.028 261.692)"),   // gray-950
    text: await probeColor("color", "oklch(92.8% 0.006 264.531)"),          // gray-200
  };
  let bodyDark = await bodyStyle();
  record("dark: body background == gray-950", bodyDark.bg === expectedDark.bg, `body=${bodyDark.bg} expected=${expectedDark.bg}`);
  record("dark: body text == gray-200", bodyDark.color === expectedDark.text, `body=${bodyDark.color} expected=${expectedDark.text}`);
  await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "light" }]);

  // 5. Utilities apply: `.font-black` == 900, and the spacing scale drives gap.
  const utilProbe = await page.evaluate(() => {
    const mk = (cls) => {
      const el = document.createElement("div");
      el.className = cls;
      document.body.appendChild(el);
      const s = getComputedStyle(el);
      const out = { display: s.display, gap: s.gap, fontWeight: s.fontWeight };
      el.remove();
      return out;
    };
    return { black: mk("font-black"), flex: mk("flex gap-6") };
  });
  record("`.font-black` computes font-weight 900", utilProbe.black.fontWeight === "900", `font-weight=${utilProbe.black.fontWeight}`);
  record("`.flex.gap-6` computes display:flex gap:24px", utilProbe.flex.display === "flex" && utilProbe.flex.gap === "24px", `display=${utilProbe.flex.display} gap=${utilProbe.flex.gap}`);
} catch (e) {
  record("harness", false, String(e.stack || e));
} finally {
  if (browser) await browser.close().catch(() => {});
  server.kill("SIGKILL");
}

console.log("=== tailwind styling gates ===");
let pass = 0;
for (const r of results) {
  console.log(`${r.ok ? "PASS" : "FAIL"} ${r.name}: ${r.detail}`);
  if (r.ok) pass++;
}
console.log(`\n${pass}/${results.length} tailwind styling gates passed`);
process.exit(pass === results.length ? 0 : 1);
