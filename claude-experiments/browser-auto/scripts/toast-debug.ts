/** Instrumented repro for the appear-watcher miss. */
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { startShopServer } from "../fixtures/shop/server.js";
import { world } from "../fixtures/shop/world.js";
import { composeWorld, seed as makeSeed } from "../src/world/index.js";
import { applyWorld } from "../src/world/adapter.js";

const BASE = join(dirname(fileURLToPath(import.meta.url)), "..");

function gaps(ticks: number[]): string {
  const out: string[] = [];
  for (let i = 1; i < ticks.length; i++) {
    const d = ticks[i]! - ticks[i - 1]!;
    if (d > 50) out.push(`${ticks[i - 1]}->${ticks[i]} (${d}ms)`);
  }
  return out.length ? out.join(", ") : "(none)";
}
process.env.BAT_TEST = "1";
const shop = await startShopServer();
const browser = await chromium.launch({ headless: true });

const catalog = makeSeed("catalog", {
  users: { shopper: { email: "s@t.dev", role: "customer" } },
  products: { "blue-widget": { name: "Blue Widget", price: 1999, stock: 12 } },
});

let misses = 0;
for (let i = 0; i < 15; i++) {
  await applyWorld(world, composeWorld([catalog]));
  const context = await browser.newContext();
  const page = await context.newPage();
  const errors: string[] = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(shop.url, { waitUntil: "domcontentloaded" });
  try {
    await page.getByRole("heading", { name: "Products" }).waitFor({ timeout: 5000 });
  } catch {
    console.log(`iter ${i}: PAGE STUCK — console: [${errors.join(" ; ")}]`);
    await context.close();
    continue;
  }
  await page.evaluate(`(() => {
    window.__toastLog = [];
    window.__rafLog = [];
    var tick = function () {
      window.__rafLog.push(Math.round(performance.now()));
      if (window.__rafLog.length < 600) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    new MutationObserver(function (muts) {
      for (const m of muts) {
        for (const n of m.addedNodes) {
          if (n instanceof Element && n.getAttribute("role") === "status") {
            window.__toastLog.push("add @" + Math.round(performance.now()) + " " + JSON.stringify(n.textContent));
          }
        }
        for (const n of m.removedNodes) {
          if (n instanceof Element && n.getAttribute("role") === "status") {
            window.__toastLog.push("remove @" + Math.round(performance.now()));
          }
        }
      }
    }).observe(document.body, { childList: true, subtree: true });
  })()`);

  // arm exactly like the runner does
  const toastLoc = page
    .getByRole("status", { name: "Added to cart" })
    .or(page.getByRole("status").filter({ hasText: "Added to cart" }))
    .first();
  const armedAt = Date.now();
  let watchError = "";
  const watch = toastLoc.waitFor({ state: "visible", timeout: 4000 }).then(
    () => true,
    (e) => {
      watchError = e instanceof Error ? e.message.split("\n").slice(0, 4).join(" | ") : String(e);
      return false;
    },
  );
  const alts = {
    plainRole: page.getByRole("status").first(),
    roleFiltered: page.getByRole("status").filter({ hasText: "Added to cart" }).first(),
    css: page.locator('[role="status"]').first(),
    text: page.getByText("Added to cart").first(),
  };
  const altWatches = Object.fromEntries(
    Object.entries(alts).map(([k, loc]) => [
      k,
      loc.waitFor({ state: "visible", timeout: 4000 }).then(() => true, () => false),
    ]),
  );

  const btn = page
    .getByRole("listitem")
    .filter({ hasText: "Blue Widget" })
    .getByRole("button", { name: "Add to cart" });
  await btn.click();
  const clickedAt = Date.now();

  const caught = await watch;
  const { toastLog, rafAroundToast } = (await page.evaluate(`(() => {
    var addAt = window.__toastLog.length ? Number(/@(\\d+)/.exec(window.__toastLog[0])[1]) : 0;
    return {
      toastLog: window.__toastLog,
      rafAroundToast: window.__rafLog.filter(function (t) { return Math.abs(t - addAt) < 400; }),
    };
  })()`)) as { toastLog: string[]; rafAroundToast: number[] };
  if (!caught) {
    misses++;
    const altResults = Object.fromEntries(
      await Promise.all(Object.entries(altWatches).map(async ([k, p]) => [k, await p] as const)),
    );
    const stillInDom = await page.locator('[role="status"]').count();
    console.log(`iter ${i}: MISSED  (arm->click ${clickedAt - armedAt}ms)  toast: [${toastLog.join(" ; ")}]`);
    console.log(`  watchError: ${watchError}`);
    console.log(`  alts: ${JSON.stringify(altResults)}  stillInDom: ${stillInDom}`);
    console.log(`  rafGapsOver50ms: ${gaps(rafAroundToast)}`);
  } else {
    console.log(`iter ${i}: caught  toast: [${toastLog.join(" ; ")}]`);
  }
  await context.close();
}
console.log(`misses: ${misses}/15`);
await browser.close();
await shop.close();
process.exit(0);
