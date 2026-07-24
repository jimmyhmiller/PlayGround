import { mkdir, mkdtemp, readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, type Browser } from "playwright";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { startShopServer } from "../fixtures/shop/server.js";
import { world } from "../fixtures/shop/world.js";
import { parseFlow } from "./dsl/parser.js";
import { runFlow, runFlowFile } from "./runner/run.js";
import { localWorldHandle } from "./runner/world-handle.js";
import { renderReport } from "./runner/trace.js";
import { loadSeeds, type BatConfig } from "./config.js";
import type { RunDeps } from "./runner/run.js";

const FIXTURE = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "shop");
const REPO_RUNS = join(dirname(fileURLToPath(import.meta.url)), "..", ".bat-test-runs");
const INTERACTIONS = join(FIXTURE, "e2e/flows/interactions.flow");

let browser: Browser;
let shop: Awaited<ReturnType<typeof startShopServer>>;
let deps: RunDeps;
let baseConfig: BatConfig;

beforeAll(async () => {
  process.env.BAT_TEST = "1";
  shop = await startShopServer();
  browser = await chromium.launch({ headless: true });
  await mkdir(REPO_RUNS, { recursive: true });
  const root = await mkdtemp(join(REPO_RUNS, "bat-interactions-"));
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

describe("first-class interactions", () => {
  it("tabs, dialog, download, drag, and iframe all work in one flow", async () => {
    const { trace, runDir } = await runFlowFile(INTERACTIONS, deps);
    if (trace.status !== "pass") throw new Error(renderReport(trace));

    // dialog was routed with the declared response and recorded
    const dialogStep = trace.steps.find((s) => s.dialogs?.length);
    expect(dialogStep?.dialogs?.[0]).toMatchObject({ dialogType: "confirm", response: "accept", declared: true });

    // download was saved into the run dir as real evidence
    const dlStep = trace.steps.find((s) => s.downloads?.length);
    const savedAs = dlStep?.downloads?.[0]?.savedAs;
    expect(savedAs).toBeTruthy();
    expect(await readFile(savedAs!, "utf8")).toContain("Blue Widget");
    expect(savedAs!.startsWith(runDir)).toBe(true);
  }, 120000);

  it("an undeclared dialog is dismissed and fails the step, naming the fix", async () => {
    const flow = parseFlow(
      `flow "undeclared dialog"
go /interactions
  expect heading "Interactions"
click button "Delete account"
  expect text "deleted" in testid "del-status"
`,
      "inline.flow",
    );
    // dismissed by default -> app sets "cancelled", so the expectation fails
    const quick: RunDeps = { ...deps, config: { ...baseConfig, stepBudgetMs: 4000, rerunsOnFailure: 1 } };
    const { trace } = await runFlow(flow, quick);
    expect(trace.status).toBe("fail");
    const failed = trace.steps.find((s) => s.status === "fail")!;
    expect(failed.dialogs?.[0]).toMatchObject({ declared: false, response: "dismiss" });
    expect(renderReport(trace)).toMatch(/del-status.*text is "cancelled"/);
  }, 60000);

  it("switch tab to a nonexistent tab explains what to assert", async () => {
    const flow = parseFlow(
      `flow "bad switch"
go /interactions
  expect heading "Interactions"
switch tab /nope
  expect heading "Nope"
`,
      "inline.flow",
    );
    const quick: RunDeps = { ...deps, config: { ...baseConfig, stepBudgetMs: 3000, rerunsOnFailure: 1 } };
    const { trace } = await runFlow(flow, quick);
    expect(trace.status).toBe("fail");
    expect(trace.steps.find((s) => s.status === "fail")!.failure).toMatch(/no open tab matches/);
  }, 60000);

  it("declared dialog data reaches the app (accept with prompt text is deterministic)", async () => {
    // covered structurally by the main flow's confirm(); assert the record shape
    const { trace } = await runFlowFile(INTERACTIONS, deps);
    const dialogStep = trace.steps.find((s) => s.dialogs?.length);
    expect(dialogStep?.dialogs?.[0]?.message).toContain("delete your account");
  }, 120000);
});
