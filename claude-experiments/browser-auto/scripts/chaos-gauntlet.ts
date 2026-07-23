/** The conditions claim, executable: under injected latency (no failures),
 * flows must STILL pass every run — latency alone can never flake a bat flow.
 * Usage: npx tsx scripts/chaos-gauntlet.ts [iterations] [seed]
 */
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { startShopServer } from "../fixtures/shop/server.js";
import { world } from "../fixtures/shop/world.js";
import { runFlowFile } from "../src/runner/run.js";
import { localWorldHandle } from "../src/runner/world-handle.js";
import { globFiles, loadSeeds, type BatConfig } from "../src/config.js";
import { renderReport } from "../src/runner/trace.js";

const BASE = join(dirname(fileURLToPath(import.meta.url)), "..");
const iterations = Number(process.argv[2] ?? 10);
const seed = Number(process.argv[3] ?? 1);

process.env.BAT_TEST = "1";
const shop = await startShopServer();
const browser = await chromium.launch({ headless: true });
const root = await mkdtemp(join(tmpdir(), "bat-chaos-"));
const config: BatConfig = {
  baseUrl: shop.url,
  world: { module: join(BASE, "fixtures/shop/world.ts") },
  seeds: join(BASE, "fixtures/shop/e2e/world/*.seed.ts"),
  flows: join(BASE, "fixtures/shop/e2e/flows/**/*.flow"),
  stepBudgetMs: 20000, // physics scales with injected latency; semantics don't change
  headless: true,
  conditions: { latencyMs: [300, 1200], seed },
  root,
};
const seeds = await loadSeeds(config);
const deps = { config, world: localWorldHandle(world), seeds, browser };
const flowFiles = await globFiles(BASE, "fixtures/shop/e2e/flows/**/*.flow");

console.log(`chaos gauntlet: latency +300–1200ms (seed ${seed}), ${flowFiles.length} flows × ${iterations} iterations`);
let failures = 0;
for (let i = 0; i < iterations; i++) {
  for (const flowFile of flowFiles) {
    const { trace } = await runFlowFile(flowFile, deps);
    if (trace.status === "pass") {
      console.log(`iter ${i + 1} ${trace.flow}: pass`);
    } else {
      failures++;
      console.log(`iter ${i + 1} ${trace.flow}: FAIL`);
      console.log(renderReport(trace));
    }
  }
}
console.log(`\nfailures under latency-only chaos: ${failures}/${iterations * flowFiles.length} (must be 0)`);
await browser.close();
await shop.close();
process.exit(failures ? 1 : 0);
