import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, type Browser } from "playwright";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { startShopServer } from "../fixtures/shop/server.js";
import { world } from "../fixtures/shop/world.js";
import { huntFlow } from "./runner/hunt.js";
import { runFlowFile } from "./runner/run.js";
import { localWorldHandle } from "./runner/world-handle.js";
import { renderReport } from "./runner/trace.js";
import { loadSeeds, type BatConfig } from "./config.js";
import type { RunDeps } from "./runner/run.js";

const FIXTURE = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "shop");
const FLAKY_CART = join(FIXTURE, "e2e/flaky/flaky-cart.flow");
const BUY_FLOW = join(FIXTURE, "e2e/flows/buy.flow");
const SEARCH_FLOW = join(FIXTURE, "e2e/flows/search.flow");

let browser: Browser;
let shop: Awaited<ReturnType<typeof startShopServer>>;
let baseConfig: BatConfig;
let deps: RunDeps;

beforeAll(async () => {
  process.env.BAT_TEST = "1";
  shop = await startShopServer();
  browser = await chromium.launch({ headless: true });
  const root = await mkdtemp(join(tmpdir(), "bat-flake-"));
  baseConfig = {
    baseUrl: shop.url,
    world: { module: join(FIXTURE, "world.ts") },
    seeds: join(FIXTURE, "e2e/world/*.seed.ts"),
    flows: join(FIXTURE, "e2e/flows/**/*.flow"),
    stepBudgetMs: 10000,
    headless: true,
    root,
  };
  const seeds = await loadSeeds(baseConfig);
  deps = { config: baseConfig, world: localWorldHandle(world), seeds, browser };
}, 60000);

afterAll(async () => {
  await browser?.close();
  await shop?.close();
});

describe("hunting a genuinely flaky app", () => {
  it("demonstrates the flaky-cart race with completion-order evidence", async () => {
    const report = await huntFlow(FLAKY_CART, deps, { runs: 24 });

    // the app's refetch race is a ~50/50 coin flip: with 24 runs the chance
    // of not observing both outcomes is < 1e-6
    expect(report.verdict).toBe("FLAKY");
    expect(report.passes).toBeGreaterThan(0);
    expect(report.fails).toBeGreaterThan(0);

    // one signature: the badge shows the stale value
    expect(report.signatures).toHaveLength(1);
    const sig = report.signatures[0]!;
    expect(sig.failedEffects).toEqual(['expect text "1" in testid "flaky-count"']);
    expect(Object.values(sig.observed).flat().join(" ")).toContain('"0"');

    // the money claim: outcome is fully determined by response completion order
    expect(report.orderDeterminesOutcome).toBe(true);
    expect(report.reportText).toContain("FULLY DETERMINED by response completion order");
    expect(report.reportText).toContain("a race in the app");
    // finish order is start-to-last; the LAST response wins the badge, so
    // POST-finished-first (stale GET applied last) is the failing order
    expect(report.reportText).toMatch(/POST \/api\/cart → GET \/api\/cart\s+→\s+0 pass, \d+ fail/);
    expect(report.reportText).toMatch(/GET \/api\/cart → POST \/api\/cart\s+→\s+\d+ pass, 0 fail/);
  }, 180000);

  it("a stable flow hunts clean", async () => {
    const report = await huntFlow(SEARCH_FLOW, deps, { runs: 5 });
    expect(report.verdict).toBe("STABLE");
    expect(report.reportText).toContain("no failures in 5 runs");
  }, 120000);
});

describe("simulated bad conditions", () => {
  it("latency alone never fails a flow, and is recorded in the trace", async () => {
    const conditioned: RunDeps = {
      ...deps,
      config: { ...baseConfig, stepBudgetMs: 20000, conditions: { latencyMs: [200, 900], seed: 7 } },
    };
    const { trace } = await runFlowFile(BUY_FLOW, conditioned);
    if (trace.status !== "pass") throw new Error(renderReport(trace));
    expect(trace.conditions).toEqual({ latencyMs: [200, 900], seed: 7 });
    // attribution is visible on the wire records
    const injected = trace.steps.flatMap((s) => s.requests).filter((r) => r.injected?.includes("injected latency"));
    expect(injected.length).toBeGreaterThan(0);
  }, 120000);

  it("injected failures are attributed, never mysterious", async () => {
    const conditioned: RunDeps = {
      ...deps,
      config: { ...baseConfig, conditions: { failRate: 1, seed: 3 } },
    };
    const { trace } = await runFlowFile(BUY_FLOW, conditioned);
    expect(trace.status).toBe("fail");
    const report = renderReport(trace);
    expect(report).toContain("SIMULATED BAD CONDITIONS ACTIVE");
    expect(report).toContain("seed 3");
    expect(report).toContain("[injected failure (conditions)]");
  }, 120000);

  it("stubbed routes are immune to chaos", async () => {
    const { parseFlow } = await import("./dsl/parser.js");
    const { runFlow } = await import("./runner/run.js");
    const flow = parseFlow(
      `flow "stub beats chaos"
given seed "catalog-basic"
given stub GET /api/products 200 json []
go /
  expect heading "Products"
  expect count listitem 0 in list "product-list"
`,
      "inline.flow",
    );
    const conditioned: RunDeps = {
      ...deps,
      // fail every non-stubbed api request; the stubbed one must still work.
      // Note the document request also passes through conditions, so only
      // fail at 1.0 would kill page load — use latency instead + assert stub data.
      config: { ...baseConfig, stepBudgetMs: 20000, conditions: { latencyMs: [100, 300], seed: 11 } },
    };
    const { trace } = await runFlow(flow, conditioned);
    if (trace.status !== "pass") throw new Error(renderReport(trace));
    const productsReq = trace.steps[0]!.requests.find((r) => r.url.includes("/api/products"));
    expect(productsReq?.injected).toBeUndefined();
  }, 120000);
});

describe("report clarity for step-level request failures", () => {
  it("completion order appears in failure reports", async () => {
    const { parseFlow } = await import("./dsl/parser.js");
    const { runFlow } = await import("./runner/run.js");
    const flow = parseFlow(
      `flow "flaky cart failing report"
given seed "catalog-basic"
go /flaky-cart
  expect heading "Flaky Cart"
click button "Add to cart"
  expect request POST /api/cart ok
  expect text "wrong-on-purpose" in testid "flaky-count"
`,
      "inline.flow",
    );
    const { trace } = await runFlow(flow, deps);
    expect(trace.status).toBe("fail");
    const report = renderReport(trace);
    expect(report).toContain("response completion order:");
    expect(report).toMatch(/finished #\d+/);
  }, 60000);
});
