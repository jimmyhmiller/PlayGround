import { mkdir, mkdtemp, readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, type Browser } from "playwright";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { startShopServer } from "../fixtures/shop/server.js";
import { world } from "../fixtures/shop/world.js";
import { parseFlow } from "./dsl/parser.js";
import { replayStep, runFlow, runFlowFile } from "./runner/run.js";
import { httpWorldHandle, localWorldHandle } from "./runner/world-handle.js";
import { renderReport } from "./runner/trace.js";
import { loadSeeds, type BatConfig } from "./config.js";
import type { Seed } from "./world/types.js";
import type { RunDeps } from "./runner/run.js";

const FIXTURE = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "shop");
const REPO_RUNS = join(dirname(fileURLToPath(import.meta.url)), "..", ".bat-test-runs");

let browser: Browser;
let shop: Awaited<ReturnType<typeof startShopServer>>;
let deps: RunDeps;
let quickFailDeps: RunDeps;
let seeds: Map<string, Seed>;

beforeAll(async () => {
  process.env.BAT_TEST = "1";
  await mkdir(REPO_RUNS, { recursive: true });
  shop = await startShopServer();
  browser = await chromium.launch({ headless: true });
  // flow traces/reports persist under the repo (gitignored), never OS temp:
  // a failure must remain identifiable after the run — see vitest.config.ts
  const root = await mkdtemp(join(REPO_RUNS, "bat-e2e-"));
  const config: BatConfig = {
    baseUrl: shop.url,
    world: { module: join(FIXTURE, "world.ts") },
    seeds: join(FIXTURE, "e2e/world/*.seed.ts"),
    flows: join(FIXTURE, "e2e/flows/**/*.flow"),
    stepBudgetMs: 10000,
    headless: true,
    root,
  };
  seeds = await loadSeeds(config);
  deps = { config, world: localWorldHandle(world), seeds, browser };
  // for tests that fail BY DESIGN: failing state effects retry until the step
  // budget ends (eventually-holds semantics) and explanation reruns multiply
  // that — keep those tests quick with a small budget and fewer reruns
  quickFailDeps = { ...deps, config: { ...config, stepBudgetMs: 3000, rerunsOnFailure: 2 } };
}, 60000);

afterAll(async () => {
  await browser?.close();
  await shop?.close();
});

const BUY_FLOW = join(FIXTURE, "e2e/flows/buy.flow");

describe("the flake gauntlet", () => {
  it("buy.flow passes 5 consecutive runs against random server latency", async () => {
    for (let run = 0; run < 5; run++) {
      const { trace } = await runFlowFile(BUY_FLOW, deps);
      if (trace.status !== "pass") {
        throw new Error(`run ${run + 1} failed:\n${renderReport(trace)}`);
      }
    }
  }, 120000);

  it("catches the 150ms toast every time (armed before the click)", async () => {
    // covered by buy.flow's `expect appear status "Added to cart"`; assert the verdict is recorded
    const { trace } = await runFlowFile(BUY_FLOW, deps);
    const step2 = trace.steps[1]!;
    const toastVerdict = step2.effects.find((e) => e.rendered.includes("appear"));
    expect(toastVerdict?.pass).toBe(true);
  }, 60000);

  it("frozen clock makes Date deterministic", async () => {
    const { trace } = await runFlowFile(join(FIXTURE, "e2e/flows/clock.flow"), deps);
    if (trace.status !== "pass") throw new Error(renderReport(trace));
  }, 60000);
});

describe("explainable failures", () => {
  function flowFrom(src: string) {
    return parseFlow(src, "inline.flow");
  }

  it("wrong expectation produces observed-vs-expected, not a timeout", async () => {
    const flow = flowFrom(`flow "wrong text"
given seed "catalog-basic"
go /
  expect heading "Products"
click button "Add to cart" in listitem "Blue Widget"
  expect request POST /api/cart ok
  expect text "99" in testid "cart-count"
`);
    const { trace } = await runFlow(flow, quickFailDeps);
    expect(trace.status).toBe("fail");
    const report = renderReport(trace);
    expect(report).toContain('✗ expect text "99" in testid "cart-count"');
    expect(report).toMatch(/observed: .*text is "1"/);
    expect(report).toContain("semantic tree");
    expect(report).toContain("bat replay");
    // the request expectation itself passed — the report shows exactly which effect failed
    expect(report).toContain("✓ expect request POST /api/cart ok");
  }, 60000);

  it("ambiguous targets are hard errors listing every match", async () => {
    const flow = flowFrom(`flow "ambiguous"
given seed "catalog-basic"
go /
  expect heading "Products"
click button "Add to cart"
  expect text "1" in testid "cart-count"
`);
    const { trace } = await runFlow(flow, quickFailDeps);
    expect(trace.status).toBe("fail");
    const report = renderReport(trace);
    expect(report).toContain("ambiguous: 3 elements match");
    expect(report).toContain("never picks the first one");
  }, 60000);

  it("a page error fails the step and is attributed to it", async () => {
    const flow = flowFrom(`flow "broken page"
given seed "catalog-basic"
go /broken
  expect heading "Broken"
`);
    const { trace } = await runFlow(flow, quickFailDeps);
    expect(trace.status).toBe("fail");
    const report = renderReport(trace);
    expect(report).toContain("kaboom");
    expect(report).toContain("errors from the page during this step");
  }, 60000);

  it("a failing API surfaces the actual response status", async () => {
    const flow = flowFrom(`flow "out of stock"
given seed "catalog-basic"
go /
  expect heading "Products"
click button "Add to cart" in listitem "Green Widget"
  expect request POST /api/cart ok
  expect text "1" in testid "cart-count"
`);
    const { trace } = await runFlow(flow, quickFailDeps);
    expect(trace.status).toBe("fail");
    const report = renderReport(trace);
    expect(report).toMatch(/responded 409/);
  }, 60000);
});

describe("settlement soundness guard", () => {
  it("a NetworkTracker attached after navigation throws; before navigation is fine", async () => {
    const { NetworkTracker } = await import("./runner/settle.js");
    const ctx = await browser.newContext();
    const page = await ctx.newPage();
    try {
      // before navigation (about:blank) — legitimate
      expect(() => new NetworkTracker(page)).not.toThrow();
      // after navigating to an http page — the exact defect that caused the
      // --fast replay flake; the guard must reject it
      await page.goto(shop.url, { waitUntil: "domcontentloaded" });
      expect(() => new NetworkTracker(page)).toThrowError(/already-navigated page/);
    } finally {
      await ctx.close();
    }
  }, 30000);

  it("--fast replay of the cart step is deterministic across many runs (the fixed flake)", async () => {
    // the tracker-after-goto bug reproduced at ~50% in isolation; run the
    // exact scenario repeatedly so a regression can't hide behind a single run
    for (let i = 0; i < 8; i++) {
      await runFlowFile(BUY_FLOW, deps);
      const result = await replayStep(BUY_FLOW, 4, deps, { fast: true });
      if (result.trace.status !== "pass") {
        throw new Error(`--fast replay flaked on iteration ${i + 1}:\n${renderReport(result.trace)}`);
      }
    }
  }, 120000);
});

describe("richer observation", () => {
  function flow(src: string) {
    return parseFlow(src, "inline.flow");
  }
  it("title, count comparisons, and regex text matching", async () => {
    const { trace } = await runFlow(
      flow(`flow "observe"
given seed "catalog-basic"
allow console-errors
go /
  expect title "bat"
  expect count listitem >= 2 in list "product-list"
  expect count listitem 3 in list "product-list"
go /account
  expect matches text "/Today is \\d{4}-\\d{2}-\\d{2}/" in testid "today"
`),
      deps,
    );
    if (trace.status !== "pass") throw new Error(renderReport(trace));
  }, 60000);

  it("captures query params, attributes, and counts — and uses them later", async () => {
    const { trace } = await runFlow(
      flow(`flow "capture"
given seed "catalog-basic"
go /?token=abc123
  expect heading "Products"
  let tok = query "token"
  let cartHref = attribute "href" of link "Cart"
  let n = count listitem in list "product-list"
click link "Cart"
  expect url $cartHref
`),
      deps,
    );
    if (trace.status !== "pass") throw new Error(renderReport(trace));
    const captures = trace.steps[0]!.captures;
    expect(captures.tok).toBe("abc123");
    expect(captures.cartHref).toBe("/cart");
    expect(captures.n).toBe("3");
  }, 60000);

  it("a failed regex match reports the observed value, not a bare miss", async () => {
    const { trace } = await runFlow(
      flow(`flow "bad regex"
given seed "catalog-basic"
allow console-errors
go /account
  expect matches text "/Today is 1999/" in testid "today"
`),
      { ...deps, config: { ...deps.config, stepBudgetMs: 4000, rerunsOnFailure: 1 } as typeof deps.config },
    );
    expect(trace.status).toBe("fail");
    expect(renderReport(trace)).toMatch(/text is "Today is 20\d\d/);
  }, 60000);
});

describe("atomic replay", () => {
  it("fallback replay re-runs to the target step and passes", async () => {
    await runFlowFile(BUY_FLOW, deps); // ensure a run exists
    const result = await replayStep(BUY_FLOW, 4, deps, { fast: false });
    expect(result.trace.status).toBe("pass");
    expect(result.tier).toContain("fallback");
    const step4 = result.trace.steps[3]!;
    expect(step4.status).toBe("pass");
    // step 5 was not run
    expect(result.trace.steps[4]!.status).toBe("not-run");
  }, 60000);

  it("replays a single `for each` iteration by display index", async () => {
    const MANAGE = join(FIXTURE, "e2e/flows/manage-cart.flow");
    // display steps: 1-4 setup, 5 = loop container (2 matches), 6/7 = iterations, 8 = post-loop
    const result = await replayStep(MANAGE, 7, deps, { fast: false });
    expect(result.trace.status).toBe("pass");
    // it re-ran THROUGH iteration 2 (reproducing state via iteration 1), then stopped
    const executed = result.trace.steps.filter((s) => s.status !== "not-run");
    expect(executed).toHaveLength(7);
    const iter2 = result.trace.steps[6]!;
    expect(iter2.iteration).toContain("iteration 2/2");
    expect(iter2.iteration).toContain("Red Widget");
    // the post-loop step was not run
    expect(result.trace.steps[7]!.status).toBe("not-run");
  }, 60000);

  it("errors clearly when the replay target exceeds the run's step count", async () => {
    const MANAGE = join(FIXTURE, "e2e/flows/manage-cart.flow");
    await expect(replayStep(MANAGE, 99, deps, { fast: false })).rejects.toThrowError(/produced \d+ step\(s\).*cannot replay step 99/);
  }, 60000);

  it("--fast replay restores the L4 world snapshot + browser checkpoint", async () => {
    await runFlowFile(BUY_FLOW, deps);
    const result = await replayStep(BUY_FLOW, 4, deps, { fast: true });
    expect(result.tier).toContain("snapshot");
    if (result.trace.status !== "pass") throw new Error(renderReport(result.trace));
  }, 60000);
});

describe("http world transport", () => {
  it("drives the world through the in-app handler", async () => {
    const httpDeps: RunDeps = { ...deps, world: httpWorldHandle(`${shop.url}/api/__bat`) };
    const { trace } = await runFlowFile(BUY_FLOW, httpDeps);
    if (trace.status !== "pass") throw new Error(renderReport(trace));
    expect(trace.worldVerification?.level).toBe(4);
  }, 60000);

  it("refuses to exist without BAT_TEST=1", async () => {
    process.env.BAT_TEST = "0";
    try {
      const res = await fetch(`${shop.url}/api/__bat`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ op: "capabilities" }),
      });
      expect(res.status).toBe(404);
    } finally {
      process.env.BAT_TEST = "1";
    }
  });
});

describe("world verification in traces", () => {
  it("records proven guarantees for the L4 fixture adapter", async () => {
    const { trace, runDir } = await runFlowFile(BUY_FLOW, deps);
    expect(trace.worldVerification?.level).toBe(4);
    expect(trace.worldVerification?.proven.join("\n")).toContain("verified by read-back");
    expect(trace.worldFingerprint).toMatch(/^sha256:/);
    const persisted = JSON.parse(await readFile(join(runDir, "trace.json"), "utf8")) as { status: string };
    expect(persisted.status).toBe("pass");
  }, 60000);
});
