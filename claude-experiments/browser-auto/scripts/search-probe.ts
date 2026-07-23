/** Probe the search transition specifically: type into the box and watch the
 * DOM + RSC traffic evolve. */
import { chromium } from "playwright";
import { applyWorld } from "../src/world/adapter.js";
import { composeWorld } from "../src/world/algebra.js";
import type { Seed, WorldAdapter } from "../src/world/types.js";

const worldMod = (await import("../apps/nextjs-dashboard/e2e/world/world.js")) as Record<string, unknown>;
const world = (worldMod.default ?? worldMod) as WorldAdapter & {
  session: (k: string, ctx: unknown) => Promise<{ cookies: Array<{ name: string; value: string }> }>;
};
const seedMod = (await import("../apps/nextjs-dashboard/e2e/world/dashboard.seed.js")) as Record<string, unknown>;
await applyWorld(world, composeWorld([(seedMod.default ?? seedMod) as Seed]));
const session = await world.session("admin", {});
const cookieHeader = session.cookies.map((c) => `${c.name}=${c.value}`).join("; ");

// server-side check first
const res = await fetch("http://localhost:3000/dashboard/invoices?page=1&query=delba", {
  headers: { cookie: cookieHeader },
});
const html = await res.text();
console.log("SSR ?query=delba:", res.status, "| contains Delba row:", html.includes("delba@oliveira.com"));

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext();
await context.addCookies(session.cookies.map(({ name, value }) => ({ name, value, url: "http://localhost:3000" })));
const page = await context.newPage();
page.on("console", (m) => m.type() === "error" && console.log("[console.error]", m.text().slice(0, 200)));
page.on("pageerror", (e) => console.log("[pageerror]", e.message.slice(0, 200)));
let t0 = Date.now();
page.on("requestfinished", (r) => {
  if (["fetch", "xhr", "document"].includes(r.resourceType()))
    console.log(`  [net +${Date.now() - t0}ms]`, r.method(), r.url().slice(0, 100));
});

await page.goto("http://localhost:3000/dashboard/invoices", { waitUntil: "domcontentloaded" });
// wait for the DESKTOP table to carry content, then act promptly (like bat does)
await page
  .getByRole("row", { name: /Evil Rabbit/ })
  .first()
  .waitFor({ timeout: 15000 });
console.log(`table content visible; typing immediately (hub=${process.env.PROBE_HUB === "1"})`);
t0 = Date.now();
await page.getByPlaceholder("Search invoices...").fill("delba");
let converged = false;
for (let i = 0; i < 40; i++) {
  const delba = await page.getByText("delba@oliveira.com").count();
  const rabbit = await page.getByText("Evil Rabbit").count();
  const rows = await page.getByRole("row").count();
  console.log(`+${Date.now() - t0}ms rows=${rows} delbaEmail=${delba} rabbit=${rabbit} url=${new URL(page.url()).search}`);
  if (rows === 2 && delba > 0 && rabbit === 0) {
    console.log(`CONVERGED at +${Date.now() - t0}ms: exactly the delba row`);
    converged = true;
    break;
  }
  await page.waitForTimeout(150); // deliberate: probe, not a bat flow
}
if (!converged) {
  console.log("=== NEVER CONVERGED; main innerHTML: ===");
  console.log((await page.locator("main").innerHTML().catch(() => "(no main)")).replace(/class="[^"]*"/g, "").slice(0, 3000));
}
await browser.close();
process.exit(0);
