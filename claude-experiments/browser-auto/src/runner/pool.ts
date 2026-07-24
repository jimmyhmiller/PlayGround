import type { Browser } from "playwright";
import { loadSeeds, loadWorldHandle, resolveWorkerConfig, type BatConfig } from "../config.js";
import { launchApp, type AppEnvironment } from "./appenv.js";
import { runFlowFile, type RunResult } from "./run.js";
import type { WorldHandle } from "./world-handle.js";
import type { Seed } from "../world/types.js";

/**
 * Parallel flow execution. Each worker leases an ISOLATED environment —
 * bat launches one app instance per worker (own port, own world) from the
 * config's `app` spec — and workers pull flow files from a shared queue.
 * World isolation is structural: two workers can run the same flow at the
 * same time because they are never in the same world.
 *
 * Serial semantics are unchanged: with workers=1 and no `app` spec, nothing
 * here runs.
 */

export class PoolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PoolError";
  }
}

interface WorkerEnv {
  config: BatConfig;
  world: WorldHandle;
  seeds: Map<string, Seed>;
  app: AppEnvironment | null;
}

export interface PoolResultEntry extends RunResult {
  file: string;
  worker: number;
  startedMs: number;
  endedMs: number;
}

export async function runPool(
  files: string[],
  config: BatConfig,
  browser: Browser,
  workers: number,
  onResult?: (r: PoolResultEntry) => void,
): Promise<{ results: PoolResultEntry[]; ports: number[] }> {
  if (workers > 1 && !config.app) {
    throw new PoolError(
      `${workers} workers need isolated app instances, so bat must launch the app itself: add "app": {"command": "..."} ` +
        `to bat.config.json (with {port}/{index} placeholders in baseUrl / world.http / env as needed), or run with --workers 1.`,
    );
  }

  const envs: WorkerEnv[] = [];
  try {
    for (let i = 0; i < workers; i++) {
      let workerConfig = config;
      let app: AppEnvironment | null = null;
      if (config.app) {
        app = await launchApp(config.app, i, config.baseUrl, config.root);
        workerConfig = resolveWorkerConfig(config, { port: app.port, index: i });
      }
      const seeds = await loadSeeds(workerConfig);
      const world = await loadWorldHandle(workerConfig, {
        index: i,
        baseUrl: workerConfig.baseUrl,
        port: app?.port ?? null,
      });
      envs.push({ config: workerConfig, world, seeds, app });
    }

    const queue = files.map((file, order) => ({ file, order }));
    const results: PoolResultEntry[] = [];

    await Promise.all(
      envs.map(async (env, workerIndex) => {
        for (;;) {
          const next = queue.shift();
          if (!next) return;
          const startedMs = Date.now();
          const result = await runFlowFile(next.file, {
            config: env.config,
            world: env.world,
            seeds: env.seeds,
            browser,
          });
          const entry: PoolResultEntry = {
            ...result,
            file: next.file,
            worker: workerIndex,
            startedMs,
            endedMs: Date.now(),
          };
          results.push(entry);
          onResult?.(entry);
        }
      }),
    );

    return { results, ports: envs.map((e) => e.app?.port ?? 0) };
  } finally {
    await Promise.all(envs.map((e) => e.app?.stop()));
  }
}
