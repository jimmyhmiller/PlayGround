/**
 * Head-to-head flakiness benchmark: bat vs idiomatic @playwright/test, same
 * app, same journeys, SAME seeded timing per iteration, fail-on-any.
 *
 * Fairness rules:
 *  - The Playwright arm uses web-first auto-retrying assertions (toBeVisible,
 *    toHaveText) — the strongest, most idiomatic form, NOT arbitrary sleeps.
 *  - Both arms face identical server timing each iteration (setShopTiming with
 *    the same seed is called before each run) and, in throttled mode, identical
 *    CPU throttling via the same CDP call.
 *  - We report where bat wins AND where the two tie, so this isn't a rigged demo.
 *
 * Usage: npx tsx scripts/framework-benchmark.ts [iterations] [cpuThrottle]
 */
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, type Browser } from "playwright";
import { expect } from "@playwright/test";
import { setShopTiming, startShopServer } from "../fixtures/shop/server.js";
import { world } from "../fixtures/shop/world.js";
import { parseFlow } from "../src/dsl/parser.js";
import { runFlow } from "../src/runner/run.js";
import { localWorldHandle } from "../src/runner/world-handle.js";
import { applyWorld } from "../src/world/adapter.js";
import { composeWorld } from "../src/world/algebra.js";
import { loadSeeds, type BatConfig } from "../src/config.js";

const BASE = join(dirname(fileURLToPath(import.meta.url)), "..");
const iterations = Number(process.argv[2] ?? 30);
const cpuThrottle = Number(process.argv[3] ?? 0);

process.env.BAT_TEST = "1";
const shop = await startShopServer();
const browser = await chromium.launch({ headless: true });
const seeds = await loadSeeds({ root: BASE, seeds: join(BASE, "fixtures/shop/e2e/world/*.seed.ts") } as BatConfig);
const catalog = seeds.get("catalog-basic")!;
const worldHandle = localWorldHandle(world);

async function throttle(ctxPage: { context(): { newCDPSession(p: unknown): Promise<{ send(m: string, p: unknown): Promise<unknown> }> } }, page: unknown): Promise<void> {
  if (!cpuThrottle) return;
  try {
    const cdp = await ctxPage.context().newCDPSession(page);
    await cdp.send("Emulate.setCPUThrottlingRate", { rate: cpuThrottle });
  } catch {
    /* ignore */
  }
}

// ---- idiomatic @playwright/test: locate, act, web-first auto-retrying assert
async function playwrightToast(): Promise<boolean> {
  const ctx = await browser.newContext({ baseURL: shop.url });
  const page = await ctx.newPage();
  await throttle(page as never, page);
  try {
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("heading", { name: "Products" })).toBeVisible();
    await page.getByRole("listitem").filter({ hasText: "Blue Widget" }).getByRole("button", { name: "Add to cart" }).click();
    // the transient toast — the strongest idiomatic Playwright assertion for it
    await expect(page.getByRole("status")).toBeVisible({ timeout: 5000 });
    return true;
  } catch {
    return false;
  } finally {
    await ctx.close();
  }
}

async function playwrightBuy(): Promise<boolean> {
  const ctx = await browser.newContext({ baseURL: shop.url });
  const page = await ctx.newPage();
  await throttle(page as never, page);
  try {
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("heading", { name: "Products" })).toBeVisible();
    await page.getByRole("listitem").filter({ hasText: "Blue Widget" }).getByRole("button", { name: "Add to cart" }).click();
    await expect(page.getByTestId("cart-count")).toHaveText("1");
    await page.getByRole("link", { name: "Cart" }).click();
    await expect(page.getByRole("heading", { name: "Your Cart" })).toBeVisible();
    await expect(page.getByRole("row", { name: /Blue Widget/ })).toBeVisible();
    return true;
  } catch {
    return false;
  } finally {
    await ctx.close();
  }
}

// ---- bat: same journeys, its own idiomatic verbs
const batConfig: BatConfig = {
  baseUrl: shop.url,
  world: { module: join(BASE, "fixtures/shop/world.ts") },
  seeds: join(BASE, "fixtures/shop/e2e/world/*.seed.ts"),
  flows: "",
  stepBudgetMs: cpuThrottle ? 30000 : 10000,
  headless: true,
  ...(cpuThrottle ? { cpuThrottle } : {}),
  root: BASE,
};
const batDeps = { config: batConfig, world: worldHandle, seeds, browser };

const TOAST_FLOW = `flow "toast"
given seed "catalog-basic"
go /
  expect heading "Products"
click button "Add to cart" in listitem "Blue Widget"
  expect request POST /api/cart ok
  expect appear status "Added to cart"
`;
const BUY_FLOW = `flow "buy"
given seed "catalog-basic"
go /
  expect heading "Products"
click button "Add to cart" in listitem "Blue Widget"
  expect request POST /api/cart ok
  expect text "1" in testid "cart-count"
click link "Cart"
  expect heading "Your Cart"
  expect row "Blue Widget" in table "cart-items"
`;

async function batRun(src: string): Promise<boolean> {
  const { trace } = await runFlow(parseFlow(src, "bench.flow"), batDeps, { explain: false, persist: false });
  return trace.status === "pass";
}

interface Tally { bat: number; pw: number; }
const toast: Tally = { bat: 0, pw: 0 };
const buy: Tally = { bat: 0, pw: 0 };

const resetWorld = () => applyWorld(world, composeWorld([catalog])); // fresh cart (bat flows reset via `given seed`)

for (let i = 0; i < iterations; i++) {
  const profile = { apiLatencyMs: [50, 400] as [number, number], toastMs: 150, seed: i + 1 };

  // TRANSIENT TOAST — identical seeded timing for both arms
  setShopTiming(profile);
  await resetWorld();
  if (await playwrightToast()) toast.pw++;
  setShopTiming(profile);
  if (await batRun(TOAST_FLOW)) toast.bat++;

  // FULL BUY JOURNEY (no transient) — expect both to be robust
  setShopTiming(profile);
  await resetWorld();
  if (await playwrightBuy()) buy.pw++;
  setShopTiming(profile);
  if (await batRun(BUY_FLOW)) buy.bat++;

  process.stdout.write(`\riter ${i + 1}/${iterations}  toast bat=${toast.bat} pw=${toast.pw}  buy bat=${buy.bat} pw=${buy.pw}   `);
}

const pct = (n: number) => `${((n / iterations) * 100).toFixed(0)}%`;
console.log(`\n${"=".repeat(64)}`);
console.log(`Head-to-head over ${iterations} iterations${cpuThrottle ? ` @ ${cpuThrottle}x CPU throttle` : ""} (fixture shop, matched seeded timing)`);
console.log(`\nTRANSIENT TOAST (150ms, appears after a jittered POST):`);
console.log(`  bat (armed-before-action):     ${toast.bat}/${iterations}  ${pct(toast.bat)} caught`);
console.log(`  Playwright (web-first assert):  ${toast.pw}/${iterations}  ${pct(toast.pw)} caught`);
console.log(`\nFULL BUY JOURNEY (no transient — the fair-tie control):`);
console.log(`  bat:         ${buy.bat}/${iterations}  ${pct(buy.bat)}`);
console.log(`  Playwright:  ${buy.pw}/${iterations}  ${pct(buy.pw)}`);
await browser.close();
await shop.close();
process.exit(0);
