/** Probe the dashboard app directly: seed the world, mint a session, fetch
 * the invoices page SSR HTML, and open it in a browser to watch hydration. */
import { chromium } from "playwright";
import { applyWorld } from "../src/world/adapter.js";
import { composeWorld } from "../src/world/algebra.js";
import type { Seed, WorldAdapter } from "../src/world/types.js";

const worldMod = (await import("../apps/nextjs-dashboard/e2e/world/world.js")) as Record<string, unknown>;
const world = (worldMod.default ?? worldMod) as WorldAdapter & { session: (k: string, ctx: unknown) => Promise<{ cookies?: Array<{ name: string; value: string }> }> };
const seedMod = (await import("../apps/nextjs-dashboard/e2e/world/dashboard.seed.js")) as Record<string, unknown>;
const seedFile = (seedMod.default ?? seedMod) as Seed;
await applyWorld(world, composeWorld([seedFile]));
const session = await world.session!("admin", { id: () => "" } as never);
const cookie = session.cookies!.map((c) => `${c.name}=${c.value}`).join("; ");

const res = await fetch("http://localhost:3000/dashboard/invoices", { headers: { cookie } });
const html = await res.text();
console.log("SSR status:", res.status, "| content-length header:", res.headers.get("content-length"));
console.log("SSR contains 'Evil Rabbit':", html.includes("Evil Rabbit"));
console.log("SSR contains 'Delba':", html.includes("Delba"));
console.log("SSR contains skeleton marker:", html.includes("animate-pulse"));

const { TransientHub } = await import("../src/runner/transients.js");
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext();
if (process.env.PROBE_HUB === "1") {
  await TransientHub.install(context);
  console.log("(TransientHub installed)");
}
await context.addCookies(session.cookies!.map(({ name, value }) => ({ name, value, url: "http://localhost:3000" })));
const page = await context.newPage();
page.on("console", (m) => m.type() === "error" && console.log("[console.error]", m.text().slice(0, 300)));
page.on("pageerror", (e) => console.log("[pageerror]", e.message.slice(0, 300)));
page.on("requestfinished", (r) => console.log(`  [net done +${Date.now() - t0}ms]`, r.method(), r.url().slice(0, 90)));
const t0 = Date.now();
await page.goto("http://localhost:3000/dashboard/invoices", { waitUntil: "domcontentloaded" });
for (let i = 0; i < 30; i++) {
  const rabbit = await page.getByText("Evil Rabbit").count();
  const rows = await page.getByRole("row").count();
  console.log(`+${Date.now() - t0}ms rows=${rows} rabbit=${rabbit}`);
  if (rabbit > 0 && i > 2) break;
  await page.waitForTimeout(100); // deliberate: this is a probe, not a bat flow
}
await browser.close();
process.exit(0);
