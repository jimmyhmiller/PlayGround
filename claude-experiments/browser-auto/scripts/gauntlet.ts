/** Stress harness for the non-flakiness claim: run buy.flow N times in a row
 * against the random-latency fixture shop. Any failure prints the full story.
 * Usage: npx tsx scripts/gauntlet.ts [iterations]
 */
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, firefox, webkit } from "playwright";
import { startShopServer } from "../fixtures/shop/server.js";
import { world } from "../fixtures/shop/world.js";
import { runFlowFile } from "../src/runner/run.js";
import { localWorldHandle } from "../src/runner/world-handle.js";
import { loadSeeds, type BatConfig } from "../src/config.js";
import { renderReport } from "../src/runner/trace.js";

const BASE = join(dirname(fileURLToPath(import.meta.url)), "..");
const iterations = Number(process.argv[2] ?? 12);

process.env.BAT_TEST = "1";
const shop = await startShopServer();
const engineName = process.env.BROWSER ?? "chromium";
const engine = engineName === "firefox" ? firefox : engineName === "webkit" ? webkit : chromium;
console.log(`browser engine: ${engineName}`);
const browser = await engine.launch({ headless: true });
const root = await mkdtemp(join(tmpdir(), "bat-gauntlet-"));
const config: BatConfig = {
  baseUrl: shop.url,
  world: { module: join(BASE, "fixtures/shop/world.ts") },
  seeds: join(BASE, "fixtures/shop/e2e/world/*.seed.ts"),
  flows: join(BASE, "fixtures/shop/e2e/flows/**/*.flow"),
  stepBudgetMs: 8000,
  headless: true,
  root,
};
const seeds = await loadSeeds(config);
const deps = { config, world: localWorldHandle(world), seeds, browser };

import { globFiles } from "../src/config.js";
const flowFiles = await globFiles(BASE, "fixtures/shop/e2e/flows/**/*.flow");
console.log(`gauntlet over ${flowFiles.length} flows × ${iterations} iterations`);

let failures = 0;
for (let i = 0; i < iterations; i++) {
  for (const flowFile of flowFiles) {
    const { trace } = await runFlowFile(flowFile, deps);
    if (trace.status === "pass") {
      const total = trace.steps.reduce((n, s) => n + s.durationMs, 0);
      console.log(`iter ${i + 1} ${trace.flow}: pass (${total}ms)`);
      continue;
    }
    failures++;
    console.log(`iter ${i + 1} ${trace.flow}: FAIL`);
    const failed = trace.steps.find((s) => s.status === "fail")!;
    console.log(`  step ${failed.index + 1}: ${failed.source}`);
    console.log(`  settle: ${JSON.stringify(failed.settle)}`);
    console.log(`  requests: ${failed.requests.map((r) => `${r.method} ${new URL(r.url).pathname} ${r.status ?? r.failure}`).join(" | ")}`);
    for (const e of failed.effects) if (!e.pass) console.log(`  ✗ ${e.rendered} — ${e.observed ?? ""}`);
    if (process.env.GAUNTLET_VERBOSE) console.log(renderReport(trace));
  }
}
console.log(`\nfailures: ${failures}/${iterations * flowFiles.length}`);
await browser.close();
await shop.close();
process.exit(failures ? 1 : 0);
