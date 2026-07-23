/** Probe the delete server action: click Delete, watch BOTH the DOM and the
 * database converge (or not), several times. */
import { chromium } from "playwright";
import postgres from "../apps/nextjs-dashboard/node_modules/postgres/src/index.js";
import { applyWorld } from "../src/world/adapter.js";
import { composeWorld } from "../src/world/algebra.js";
import type { Seed, WorldAdapter } from "../src/world/types.js";

const sql = postgres("postgres://jimmyhmiller@localhost:5432/bat_dashboard", { max: 1 });
const worldMod = (await import("../apps/nextjs-dashboard/e2e/world/world.js")) as Record<string, unknown>;
const world = (worldMod.default ?? worldMod) as WorldAdapter & {
  session: (k: string, ctx: unknown) => Promise<{ cookies: Array<{ name: string; value: string }> }>;
};
const seedMod = (await import("../apps/nextjs-dashboard/e2e/world/dashboard.seed.js")) as Record<string, unknown>;
const seedFile = (seedMod.default ?? seedMod) as Seed;

const browser = await chromium.launch({ headless: true });

for (let attempt = 1; attempt <= 6; attempt++) {
  await applyWorld(world, composeWorld([seedFile]));
  const session = await world.session("admin", {});
  const context = await browser.newContext();
  await context.addCookies(session.cookies.map(({ name, value }) => ({ name, value, url: "http://localhost:3000" })));
  const page = await context.newPage();
  await page.goto("http://localhost:3000/dashboard/invoices", { waitUntil: "domcontentloaded" });
  await page.getByRole("row", { name: /Evil Rabbit/ }).waitFor({ timeout: 10000 });
  await page.waitForTimeout(1500); // deliberate: simulate an unhurried human, fully hydrated page

  const t0 = Date.now();
  await page.getByRole("row", { name: /Evil Rabbit/ }).getByRole("button", { name: "Delete" }).click();

  let domGone = -1;
  let dbCount = -1;
  for (let i = 0; i < 30; i++) {
    const rabbitRows = await page.getByRole("row", { name: /Evil Rabbit/ }).count();
    const rows = await sql`SELECT count(*)::int AS n FROM invoices`;
    dbCount = rows[0]!.n;
    if (rabbitRows === 0 && domGone < 0) domGone = Date.now() - t0;
    if (domGone >= 0) break;
    await page.waitForTimeout(200); // deliberate: probe
  }
  console.log(
    `attempt ${attempt}: db invoices=${dbCount} (3→2 expected) | DOM row gone: ${domGone >= 0 ? `+${domGone}ms` : "NEVER (6s)"}`,
  );
  await context.close();
}
await browser.close();
await sql.end();
process.exit(0);
