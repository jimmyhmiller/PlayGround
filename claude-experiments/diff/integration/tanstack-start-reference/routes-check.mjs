// All-routes direct-load + hydrate oracle (test-only; NEVER referenced by the
// build). Boots the emitted `.diffpack-output/server/index.mjs` once, then
// DIRECT-LOADS every route URL of the pinned app in real headless Chrome and
// proves each one both server-renders the expected content AND hydrates cleanly.
//
// For every route it asserts the four invariants used by the existing checks:
//   1. the expected SSR text is present in the INITIAL server HTML (raw fetch,
//      before any client JS runs), proving the content is truly server-rendered;
//   2. window.__TSR_ROUTER__ is set after load (client executed + hydrated);
//   3. zero uncaught page/JS errors;
//   4. no server-only leak in the console (async_hooks / "No Start context" /
//      "module is not defined").
//
// The oracle NEVER weakens itself to pass: content expectations come from each
// route's own source, and network-dependent routes assert structural shape
// (the fetched item rendered) rather than a hardcoded remote value.
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import puppeteer from "puppeteer-core";

const here = dirname(fileURLToPath(import.meta.url));
const outputRoot = join(here, ".diffpack-output");
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 8811;
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

// Decode the handful of HTML entities React's SSR emits so raw-HTML substring
// checks see the same text the browser's textContent would.
function decodeEntities(s) {
  return s
    .replace(/&#x27;/g, "'")
    .replace(/&#39;/g, "'")
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&#x2190;/g, "←");
}

// Classify a console/error string exactly like hydration-check.mjs, so a
// server-only leak (async_hooks / No Start context) or a broken bundle
// (module is not defined) is caught, while the two tracked-elsewhere framework
// gaps (tailwind asset, generic resource load) are not miscounted as JS errors.
function classify(text) {
  if (/module is not defined/i.test(text)) return "moduleNotDefined";
  if (/node builtin node:async_hooks/i.test(text) || /async_hooks/i.test(text) || /No Start context found/i.test(text)) return "serverLeak";
  if (/tailwindcss/i.test(text)) return "cssAsset";
  if (/Failed to load resource/i.test(text)) return "resource";
  return "js";
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

// Load one route in a fresh page, capture all error signals, and return the
// browser-side observations plus the raw (pre-JS) server HTML.
async function loadRoute(browser, path, { waitFor } = {}) {
  const page = await browser.newPage();
  const jsErrors = [];
  const serverLeaks = [];
  const moduleNotDefined = [];
  page.on("console", (m) => {
    if (m.type() !== "error") return;
    const k = classify(m.text());
    if (k === "serverLeak") serverLeaks.push(m.text());
    else if (k === "moduleNotDefined") { moduleNotDefined.push(m.text()); jsErrors.push(m.text()); }
    else if (k === "js") jsErrors.push(m.text());
  });
  page.on("pageerror", (e) => {
    const text = String(e.message || e);
    const k = classify(text);
    if (k === "serverLeak") serverLeaks.push(text);
    else { if (k === "moduleNotDefined") moduleNotDefined.push(text); jsErrors.push(text); }
  });

  // Raw server HTML BEFORE any client JS: fetch it independently.
  const rawHtml = decodeEntities(await (await fetch(BASE + path)).text());

  await page.goto(BASE + path, { waitUntil: "networkidle0", timeout: 25000 });
  // The client bundle executed + hydrated iff __TSR_ROUTER__ is installed.
  const hydrated = await page
    .waitForFunction(() => window.__TSR_ROUTER__ !== undefined || window.$_TSR === undefined, { timeout: 12000 })
    .then(() => true)
    .catch(() => false);
  if (waitFor) {
    await page.waitForFunction(waitFor, { timeout: 12000 }).catch(() => {});
  }
  const routerPresent = await page.evaluate(() => typeof window.__TSR_ROUTER__ !== "undefined");
  // textContent (not innerText) so CSS transforms like `uppercase` don't mutate
  // the rendered text we assert on — we want what the component produced.
  const bodyText = await page.evaluate(() => document.body.textContent);
  const finalPath = await page.evaluate(() => location.pathname);

  return { page, rawHtml, hydrated, routerPresent, bodyText, finalPath, jsErrors, serverLeaks, moduleNotDefined };
}

// Assert the four invariants for a route, given its route-specific SSR/DOM
// expectation callbacks.
async function checkRoute(browser, label, path, opts) {
  const { ssrNeedles = [], domNeedles = [], expectPath = path, waitFor, extra } = opts;
  const r = await loadRoute(browser, path, { waitFor });
  try {
    // Invariant 1: expected SSR text present in the INITIAL server HTML.
    const missingSsr = ssrNeedles.filter((n) => !r.rawHtml.includes(n));
    record(`${label}: SSR content in initial HTML`, missingSsr.length === 0, missingSsr.length ? `missing ${JSON.stringify(missingSsr)}` : `all ${ssrNeedles.length} present`);

    // Invariant 2: hydrated (client executed, __TSR_ROUTER__ set) at expected URL.
    const pathOk = r.finalPath === expectPath;
    record(`${label}: hydrated + __TSR_ROUTER__ set${expectPath !== path ? ` @ ${expectPath}` : ""}`, r.hydrated && r.routerPresent && pathOk, `hydrated=${r.hydrated}, router=${r.routerPresent}, path=${r.finalPath}`);

    // Invariant 3: zero uncaught JS/page errors.
    record(`${label}: no uncaught JS errors`, r.jsErrors.length === 0, JSON.stringify(r.jsErrors.slice(0, 4)));

    // Invariant 4: no server-only leak on the client.
    record(`${label}: no server-only leak`, r.serverLeaks.length === 0, JSON.stringify([...new Set(r.serverLeaks)].slice(0, 2)));

    // Route-specific DOM assertions (post-hydration, entity-decoded via innerText).
    const missingDom = domNeedles.filter((n) => !r.bodyText.includes(n));
    if (domNeedles.length) record(`${label}: rendered content in DOM`, missingDom.length === 0, missingDom.length ? `missing ${JSON.stringify(missingDom)} in ${JSON.stringify(r.bodyText.slice(0, 200))}` : `all present`);

    if (extra) await extra(r);
  } finally {
    await r.page.close().catch(() => {});
  }
}

let browser;
try {
  await waitForServer(() => serverLog);
  browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: true,
    args: ["--no-sandbox", "--disable-gpu"],
  });

  // / — the home route.
  await checkRoute(browser, "/", "/", {
    ssrNeedles: ["Welcome Home!!!"],
    domNeedles: ["Welcome Home!!!"],
  });

  // /posts — layout loader (fetchPosts server fn) renders the list.
  await checkRoute(browser, "/posts", "/posts", {
    ssrNeedles: ["Non-existent Post"],
    waitFor: () => document.querySelectorAll("ul.list-disc li div").length >= 10,
    extra: async (r) => {
      const n = await r.page.$$eval("ul.list-disc li div", (e) => e.length);
      record("/posts: >=10 post items", n >= 10, `${n} items`);
    },
  });

  // /posts/1 — dynamic param loader (fetchPost server fn), title + Deep View link.
  await checkRoute(browser, "/posts/1", "/posts/1", {
    ssrNeedles: ["Deep View"],
    waitFor: () => !!document.querySelector("h4.text-xl.font-bold.underline"),
    extra: async (r) => {
      const title = await r.page.$eval("h4.text-xl.font-bold.underline", (e) => e.textContent).catch(() => null);
      record("/posts/1: fetched post title rendered", !!title && title.length > 0, `h4=${JSON.stringify(title)}`);
      const deep = await r.page.$('a[href="/posts/1/deep"]');
      record("/posts/1: Deep View link present", !!deep, `present=${!!deep}`);
    },
  });

  // /posts/1/deep — pathless-of-posts dynamic route (posts_ opts out of layout).
  await checkRoute(browser, "/posts/1/deep", "/posts/1/deep", {
    ssrNeedles: ["← All Posts"],
    waitFor: () => !!document.querySelector("h4.text-xl.font-bold.underline"),
    domNeedles: ["← All Posts"],
    extra: async (r) => {
      const title = await r.page.$eval("h4.text-xl.font-bold.underline", (e) => e.textContent).catch(() => null);
      record("/posts/1/deep: fetched post title rendered", !!title && title.length > 0, `h4=${JSON.stringify(title)}`);
    },
  });

  // /users — layout loader fetches /api/users (nested server route + middleware).
  await checkRoute(browser, "/users", "/users", {
    ssrNeedles: ["Non-existent User"],
    waitFor: () => document.querySelectorAll("ul.list-disc li div").length >= 10,
    extra: async (r) => {
      const n = await r.page.$$eval("ul.list-disc li div", (e) => e.length);
      record("/users: >=10 user items", n >= 10, `${n} items`);
    },
  });

  // /users/1 — dynamic param loader fetches /api/users/1.
  await checkRoute(browser, "/users/1", "/users/1", {
    ssrNeedles: ["View as JSON"],
    waitFor: () => !!document.querySelector("h4.text-xl.font-bold.underline"),
    domNeedles: ["View as JSON"],
    extra: async (r) => {
      const name = await r.page.$eval("h4.text-xl.font-bold.underline", (e) => e.textContent).catch(() => null);
      record("/users/1: fetched user name rendered", !!name && name.length > 0, `h4=${JSON.stringify(name)}`);
    },
  });

  // /deferred — React streaming: awaited server fn in initial HTML, then two
  // Suspense/Await boundaries resolve via the stream.
  await checkRoute(browser, "/deferred", "/deferred", {
    ssrNeedles: ["John Doe"], // the awaited `person` is in the initial HTML
    waitFor: () =>
      !!document.querySelector('[data-testid="deferred-person"]') &&
      !!document.querySelector('[data-testid="deferred-stuff"]'),
    extra: async (r) => {
      const regular = await r.page.$eval('[data-testid="regular-person"]', (e) => e.textContent).catch(() => null);
      record("/deferred: regular-person (awaited) rendered", !!regular && /John Doe/.test(regular), `text=${JSON.stringify(regular)}`);
      const dPerson = await r.page.$eval('[data-testid="deferred-person"]', (e) => e.textContent).catch(() => null);
      record("/deferred: deferred-person Suspense resolved", !!dPerson && /Tanner Linsley/.test(dPerson), `text=${JSON.stringify(dPerson)}`);
      const dStuff = await r.page.$eval('[data-testid="deferred-stuff"]', (e) => e.textContent).catch(() => null);
      record("/deferred: deferred-stuff Suspense resolved", !!dStuff && /Hello deferred!/.test(dStuff), `text=${JSON.stringify(dStuff)}`);
    },
  });

  // /redirect — beforeLoad throws redirect({ to: '/posts' }); a direct browser
  // load must end up on /posts with the posts list rendered.
  await checkRoute(browser, "/redirect", "/redirect", {
    expectPath: "/posts",
    waitFor: () => location.pathname === "/posts" && document.querySelectorAll("ul.list-disc li div").length >= 10,
    extra: async (r) => {
      const n = await r.page.$$eval("ul.list-disc li div", (e) => e.length).catch(() => 0);
      record("/redirect: landed on /posts with list", r.finalPath === "/posts" && n >= 10, `path=${r.finalPath}, items=${n}`);
    },
  });

  // /route-a and /route-b — pathless nested layout: _pathlessLayout ("I'm a
  // layout") > _nested-layout ("I'm a nested layout") > leaf.
  await checkRoute(browser, "/route-a", "/route-a", {
    ssrNeedles: ["I'm a layout", "I'm a nested layout", "I'm A!"],
    domNeedles: ["I'm a layout", "I'm a nested layout", "I'm A!"],
  });
  await checkRoute(browser, "/route-b", "/route-b", {
    ssrNeedles: ["I'm a layout", "I'm a nested layout", "I'm B!"],
    domNeedles: ["I'm a layout", "I'm a nested layout", "I'm B!"],
  });

  // 404 — an unknown path renders the app's NotFound component (still hydrates).
  await checkRoute(browser, "404", "/this-route-does-not-exist", {
    ssrNeedles: ["Go back", "Start Over"],
    domNeedles: ["Go back", "Start Over"],
  });
} catch (e) {
  record("harness", false, String(e.stack || e));
} finally {
  if (browser) await browser.close().catch(() => {});
  server.kill("SIGKILL");
}

console.log("\n=== all-routes hydration gates ===");
let pass = 0;
for (const r of results) {
  console.log(`${r.ok ? "PASS" : "FAIL"} ${r.name}: ${r.detail}`);
  if (r.ok) pass++;
}
console.log(`\n${pass}/${results.length} route gates passed`);
process.exit(pass === results.length ? 0 : 1);
