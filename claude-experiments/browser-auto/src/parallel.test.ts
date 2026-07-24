import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, type Browser } from "playwright";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { loadConfig } from "./config.js";
import { runPool, PoolError, type PoolResultEntry } from "./runner/pool.js";
import { renderReport } from "./runner/trace.js";

const FIXTURE = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "shop");
const BUY = join(FIXTURE, "e2e/flows/buy.flow");
const CLOCK = join(FIXTURE, "e2e/flows/clock.flow");
const SEARCH = join(FIXTURE, "e2e/flows/search.flow");

let browser: Browser;

beforeAll(async () => {
  browser = await chromium.launch({ headless: true });
}, 60000);

afterAll(async () => {
  await browser?.close();
});

describe("parallel workers with structural world isolation", () => {
  it("runs the same flows concurrently on isolated app instances — all pass", async () => {
    const config = await loadConfig(FIXTURE);
    // the same flow twice IS the isolation proof: two concurrent buys in a
    // shared world would see each other's cart mutations and fail
    const files = [BUY, BUY, SEARCH, SEARCH, CLOCK, CLOCK];
    const { results, ports } = await runPool(files, config, browser, 3);

    expect(results).toHaveLength(6);
    for (const r of results) {
      if (r.trace.status !== "pass") {
        throw new Error(`[worker ${r.worker}] ${r.file}:\n${renderReport(r.trace)}`);
      }
    }

    // three genuinely distinct environments
    expect(new Set(ports).size).toBe(3);
    expect(ports.every((p) => p > 0)).toBe(true);

    // work actually overlapped in time (not serialized through one worker)
    const overlaps = results.some((a, i) =>
      results.some((b, j) => i !== j && a.worker !== b.worker && a.startedMs < b.endedMs && b.startedMs < a.endedMs),
    );
    expect(overlaps).toBe(true);

    // every worker did something
    expect(new Set(results.map((r) => r.worker)).size).toBe(3);
  }, 240000);

  it("refuses parallel workers without an app spec, and says why", async () => {
    const config = await loadConfig(FIXTURE);
    const { app: _drop, ...noApp } = config;
    await expect(runPool([BUY], { ...noApp, baseUrl: "http://localhost:4173" }, browser, 4)).rejects.toThrowError(
      PoolError,
    );
    await expect(runPool([BUY], { ...noApp, baseUrl: "http://localhost:4173" }, browser, 4)).rejects.toThrowError(
      /isolated app instances/,
    );
  });

  it("app startup failure reports the log tail, not a bare timeout", async () => {
    const config = await loadConfig(FIXTURE);
    const broken = {
      ...config,
      app: { command: "node -e \"console.error('boom: nope'); process.exit(3)\"", startupTimeoutMs: 4000 },
    };
    await expect(runPool([BUY], broken, browser, 1)).rejects.toThrowError(/boom: nope/);
  }, 30000);

  it("results attribute worker and environment for postmortems", async () => {
    const config = await loadConfig(FIXTURE);
    const { results } = await runPool([CLOCK], config, browser, 1);
    const r = results[0] as PoolResultEntry;
    expect(r.worker).toBe(0);
    expect(r.trace.status).toBe("pass");
    expect(r.endedMs).toBeGreaterThanOrEqual(r.startedMs);
  }, 120000);
});
