/** Run the fixture shop standalone (for CLI demos): npx tsx fixtures/shop/serve.ts */
import { startShopServer } from "./server.js";
import { world } from "./world.js";
import { applyWorld } from "../../src/world/adapter.js";
import { composeWorld } from "../../src/world/algebra.js";
import catalog from "./e2e/world/catalog.seed.js";

process.env.BAT_TEST = "1";
const { url } = await startShopServer(Number(process.env.PORT ?? 4173));
await applyWorld(world, composeWorld([catalog]));
console.log(`fixture shop on ${url} (BAT_TEST=1, world seeded with catalog-basic)`);
