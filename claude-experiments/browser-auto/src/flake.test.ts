import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, type Browser } from "playwright";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { startShopServer } from "../fixtures/shop/server.js";
import { world } from "../fixtures/shop/world.js";
import { parseFlow } from "./dsl/parser.js";
import { runFlow, runFlowFile } from "./runner/run.js";
import { localWorldHandle } from "./runner/world-handle.js";
import { renderReport, type FlowTrace } from "./runner/trace.js";
import { loadSeeds, type BatConfig } from "./config.js";
import type { RunDeps } from "./runner/run.js";

const FIXTURE = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "shop");
const FLAKY_CART = join(FIXTURE, "e2e/flaky/flaky-cart.flow");
const BUY_FLOW = join(FIXTURE, "e2e/flows/buy.flow");

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

describe("automatic failure triage: THE APP IS FAULTY", () => {
  it("a race in the app is diagnosed as nondeterministic, with order evidence", async () => {
    // 12 reruns of the ~50/50 race: P(no passing rerun) ≈ 0.02% — the verdict
    // is statistically forced without making this test itself flaky.
    const racy: RunDeps = { ...deps, config: { ...baseConfig, diagnoseReruns: 12 } };
    let trace: FlowTrace | null = null;
    for (let i = 0; i < 20 && !trace; i++) {
      const result = await runFlowFile(FLAKY_CART, racy);
      if (result.trace.status === "fail") trace = result.trace;
    }
    if (!trace) throw new Error("no failure in 20 runs — the fixture race disappeared?");

    const d = trace.diagnosis!;
    expect(d.verdict).toBe("app-inconsistent");
    expect(d.headline).toContain("THE APP IS FAULTY");
    expect(d.details.join("\n")).toContain("variance can only come from the app");

    const report = renderReport(trace);
    expect(report).toContain("diagnosis: THE APP IS FAULTY (nondeterministic)");
    expect(report).toContain("a race in the app");
    expect(d.orderEvidence!.length).toBeGreaterThan(1);
    expect(d.orderEvidence!.every((o) => o.passes === 0 || o.fails === 0)).toBe(true);
  }, 240000);

  it("a consistently wrong expectation is diagnosed as deterministic, not flaky", async () => {
    const flow = parseFlow(
      `flow "wrong expectation"
given seed "catalog-basic"
go /
  expect heading "Products"
click button "Add to cart" in listitem "Blue Widget"
  expect request POST /api/cart ok
  expect text "99" in testid "cart-count"
`,
      "inline.flow",
    );
    const { trace } = await runFlow(flow, deps);
    expect(trace.status).toBe("fail");
    const d = trace.diagnosis!;
    expect(d.verdict).toBe("app-behavior-mismatch");
    expect(d.headline).toContain("CONSISTENTLY");
    expect(d.reruns.failedSame).toBe(4);
    expect(d.reruns.passed).toBe(0);
    const report = renderReport(trace);
    expect(report).toContain("this is not flakiness");
    expect(report).toContain("the expectation is stale");
  }, 120000);
});

describe("automatic failure triage: THE TEST IS FAULTY", () => {
  it("ambiguous targets are a test fault — no reruns needed", async () => {
    const flow = parseFlow(
      `flow "ambiguous"
given seed "catalog-basic"
go /
  expect heading "Products"
click button "Add to cart"
  expect text "1" in testid "cart-count"
`,
      "inline.flow",
    );
    const { trace } = await runFlow(flow, deps);
    const d = trace.diagnosis!;
    expect(d.verdict).toBe("test-fault");
    expect(d.headline).toContain("THE TEST IS FAULTY");
    expect(d.reruns.total).toBe(0);
  }, 60000);

  it("a misspelled target name gets a did-you-mean", async () => {
    const flow = parseFlow(
      `flow "typo"
given seed "catalog-basic"
go /
  expect heading "Prodcuts"
`,
      "inline.flow",
    );
    const quick: RunDeps = { ...deps, config: { ...baseConfig, stepBudgetMs: 4000, diagnoseReruns: 2 } };
    const { trace } = await runFlow(flow, quick);
    const d = trace.diagnosis!;
    expect(d.verdict).toBe("test-fault");
    expect(d.headline).toContain("THE TEST IS LIKELY FAULTY");
    expect(d.details.join("\n")).toContain('did you mean heading "Products"');
  }, 120000);

  it("a misspelled testid suggests the nearest real testid", async () => {
    const flow = parseFlow(
      `flow "testid typo"
given seed "catalog-basic"
go /
  expect heading "Products"
  expect text "0" in testid "cart-cnt"
`,
      "inline.flow",
    );
    // a missing element waits out the step budget on every rerun; keep this quick
    const quick: RunDeps = { ...deps, config: { ...baseConfig, stepBudgetMs: 4000, diagnoseReruns: 2 } };
    const { trace } = await runFlow(flow, quick);
    const d = trace.diagnosis!;
    expect(d.verdict).toBe("test-fault");
    expect(d.details.join("\n")).toContain('testid "cart-count"');
  }, 120000);
});

describe("automatic failure triage: CHAOS-INDUCED", () => {
  it("failures under injected conditions that pass clean are attributed to the conditions", async () => {
    const conditioned: RunDeps = {
      ...deps,
      config: { ...baseConfig, conditions: { failRate: 1, seed: 3 }, diagnoseReruns: 1 },
    };
    const { trace } = await runFlowFile(BUY_FLOW, conditioned);
    expect(trace.status).toBe("fail");
    const d = trace.diagnosis!;
    expect(d.verdict).toBe("conditions-induced");
    expect(d.headline).toContain("CHAOS-INDUCED");
    const report = renderReport(trace);
    expect(report).toContain("SIMULATED BAD CONDITIONS ACTIVE");
    expect(report).toContain("passes without the injected conditions");
  }, 120000);

  it("latency alone never fails a flow, and is recorded in the trace", async () => {
    const conditioned: RunDeps = {
      ...deps,
      config: { ...baseConfig, stepBudgetMs: 20000, conditions: { latencyMs: [200, 900], seed: 7 } },
    };
    const { trace } = await runFlowFile(BUY_FLOW, conditioned);
    if (trace.status !== "pass") throw new Error(renderReport(trace));
    expect(trace.conditions).toEqual({ latencyMs: [200, 900], seed: 7 });
    const injected = trace.steps.flatMap((s) => s.requests).filter((r) => r.injected?.includes("injected latency"));
    expect(injected.length).toBeGreaterThan(0);
  }, 120000);
});

describe("report clarity", () => {
  it("failure reports carry completion order + the diagnosis section", async () => {
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
    expect(report).toContain("diagnosis:");
    expect(report).toMatch(/evidence base: \d+ automatic rerun/);
  }, 120000);
});
