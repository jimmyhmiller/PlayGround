import { randomBytes } from "node:crypto";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { chromium, firefox, webkit, type Browser } from "playwright";
import { parseFlow } from "../dsl/parser.js";
import type { Flow } from "../dsl/ir.js";
import { explainFailure } from "./explain.js";
import type { Seed } from "../world/types.js";
import type { BatConfig } from "../config.js";
import { composeFlowWorld, prepareContext, runSteps } from "./executor.js";
import { NetworkTracker, settle } from "./settle.js";
import { renderReport, type Checkpoint, type FlowTrace } from "./trace.js";
import type { WorldHandle } from "./world-handle.js";

export interface RunDeps {
  config: BatConfig;
  world: WorldHandle;
  seeds: Map<string, Seed>;
  browser: Browser;
}

export async function launchBrowser(config: BatConfig): Promise<Browser> {
  const engine = config.browser === "firefox" ? firefox : config.browser === "webkit" ? webkit : chromium;
  return engine.launch({ headless: config.headless });
}

function slug(file: string): string {
  return basename(file).replace(/\.[^.]+$/, "").replace(/[^\w-]+/g, "_");
}

export interface RunResult {
  trace: FlowTrace;
  runDir: string;
  reportPath: string;
}

export interface RunFlowOptions {
  /** explain failures automatically (causal account + rerun evidence). Default true. */
  explain?: boolean;
  /** write trace/report/checkpoints under .bat/runs. Default true. */
  persist?: boolean;
}

export async function runFlowFile(file: string, deps: RunDeps, options: RunFlowOptions = {}): Promise<RunResult> {
  const source = await readFile(file, "utf8");
  const flow = parseFlow(source, file);
  return runFlow(flow, deps, options);
}

export async function runFlow(flow: Flow, deps: RunDeps, options: RunFlowOptions = {}): Promise<RunResult> {
  const { explain = true, persist = true } = options;
  const { config, world, seeds, browser } = deps;
  const description = composeFlowWorld(flow, seeds);
  let verification: FlowTrace["worldVerification"] = null;
  if (description) {
    const applied = await world.apply(description);
    verification = applied.verification;
  }

  // random suffix: parallel workers can start the same flow in the same ms
  const runId = `${new Date().toISOString().replace(/[:.]/g, "-")}-${randomBytes(3).toString("hex")}`;
  const runDir = join(config.root, ".bat", "runs", slug(flow.file), runId);
  if (persist) await mkdir(runDir, { recursive: true });

  const context = await browser.newContext({ baseURL: config.baseUrl });
  const page = await context.newPage();
  let trace: FlowTrace;
  try {
    const baseOpts = {
      baseUrl: config.baseUrl,
      stepBudgetMs: config.stepBudgetMs,
      seedRegistry: seeds,
      world,
      ...(config.conditions ? { conditions: config.conditions } : {}),
    };
    const env = await prepareContext(context, page, flow, baseOpts);
    if (persist) {
      env.session.downloads.onDownload = async (download, filename) => {
        const safe = filename.replace(/[^\w.-]+/g, "_") || "download";
        const dest = join(runDir, `download-${safe}`);
        try {
          await download.saveAs(dest);
          return dest;
        } catch (e) {
          if (process.env.BAT_DEBUG) console.error("download saveAs failed:", e instanceof Error ? e.message : e);
          return null; // failed to save — recorded as unsaved, not falsely claimed
        }
      };
    }

    trace = await runSteps(
      flow,
      context,
      page,
      {
        ...baseOpts,
        ...(persist
          ? {
              onCheckpoint: async (cp: Checkpoint) => {
                await writeFile(join(runDir, `checkpoint-${cp.step}.json`), JSON.stringify(cp), "utf8");
              },
              onScreenshot: async (stepIndex: number, png: Buffer) => {
                const name = `step-${stepIndex + 1}.png`;
                await writeFile(join(runDir, name), png);
                return name;
              },
            }
          : {}),
      },
      env,
      { fingerprint: description?.fingerprint ?? null, verification },
    );
  } finally {
    await context.close();
  }

  // Every failure gets an automatic causal explanation (no verdicts).
  const rerunTraces: FlowTrace[] = [];
  if (trace.status === "fail" && explain) {
    trace.explanation = await explainFailure(flow, trace, {
      ...deps,
      rerun: async (rerunDeps) => {
        const rerun = (await runFlow(flow, rerunDeps, { explain: false, persist: false })).trace;
        rerunTraces.push(rerun);
        return rerun;
      },
    });
  }

  if (persist) {
    const report = renderReport(trace);
    await writeFile(join(runDir, "trace.json"), JSON.stringify(trace, null, 2), "utf8");
    await writeFile(join(runDir, "report.txt"), report, "utf8");
    // rerun traces referenced by the explanation, for side-by-side comparison
    for (let i = 0; i < rerunTraces.length; i++) {
      await mkdir(join(runDir, "reruns"), { recursive: true });
      await writeFile(join(runDir, "reruns", `rerun-${i + 1}.json`), JSON.stringify(rerunTraces[i], null, 2), "utf8");
    }
    await writeFile(join(config.root, ".bat", "runs", slug(flow.file), "latest"), runId, "utf8");
  }
  return { trace, runDir, reportPath: join(runDir, "report.txt") };
}

export interface ReplayOptions {
  /** use the browser checkpoint (url + storage) instead of re-running prior steps.
   * Faster, but cannot restore transient UI state (open dialogs etc.). */
  fast: boolean;
}

export interface ReplayResult extends RunResult {
  tier: string;
}

export async function replayStep(
  file: string,
  stepOneBased: number,
  deps: RunDeps,
  opts: ReplayOptions,
): Promise<ReplayResult> {
  const { config, world, seeds, browser } = deps;
  const source = await readFile(file, "utf8");
  const flow = parseFlow(source, file);
  const target = stepOneBased - 1;
  if (target < 0 || target >= flow.steps.length) {
    throw new Error(`flow "${flow.name}" has ${flow.steps.length} steps; cannot replay step ${stepOneBased}`);
  }

  const description = composeFlowWorld(flow, seeds);
  let verification: FlowTrace["worldVerification"] = null;

  const runId = `replay-${new Date().toISOString().replace(/[:.]/g, "-")}`;
  const runDir = join(config.root, ".bat", "runs", slug(file), runId);
  await mkdir(runDir, { recursive: true });

  let tier: string;
  let checkpoint: Checkpoint | null = null;
  if (opts.fast && target > 0) {
    checkpoint = await loadCheckpoint(config, file, target - 1);
    if (!checkpoint) {
      throw new Error(
        `--fast replay needs checkpoint-${target - 1}.json from a previous run of ${file} — run 'bat run ${file}' first`,
      );
    }
  }

  if (checkpoint) {
    if (checkpoint.worldSnapshotId) {
      await world.restore(checkpoint.worldSnapshotId);
      tier = `snapshot (world restored from ${checkpoint.worldSnapshotId}, browser from checkpoint ${target - 1})`;
    } else {
      if (description) {
        const applied = await world.apply(description);
        verification = applied.verification;
      }
      tier =
        "checkpoint browser state + RESEEDED world (adapter has no snapshot/restore — world-side mutations from prior steps are NOT reproduced; implement snapshot/restore for L4)";
    }
  } else {
    if (description) {
      const applied = await world.apply(description);
      verification = applied.verification;
    }
    tier =
      target === 0
        ? "direct (first step needs no prior state)"
        : "fallback (reseed + re-run steps 1.." + target + " fully settled, then the target step)";
  }

  const contextOpts: Parameters<Browser["newContext"]>[0] = { baseURL: config.baseUrl };
  if (checkpoint) contextOpts!.storageState = checkpoint.storageState as never;
  const context = await browser.newContext(contextOpts);
  const page = await context.newPage();
  try {
    const baseOpts = {
      baseUrl: config.baseUrl,
      stepBudgetMs: config.stepBudgetMs,
      seedRegistry: seeds,
      world,
      ...(config.conditions ? { conditions: config.conditions } : {}),
    };
    const flowForPrepare = checkpoint ? { ...flow, givens: flow.givens.filter((g) => g.type !== "user") } : flow;
    const env = await prepareContext(context, page, flowForPrepare, baseOpts);

    let startAt = 0;
    if (checkpoint) {
      await page.goto(checkpoint.url, { waitUntil: "domcontentloaded", timeout: config.stepBudgetMs });
      const tracker = new NetworkTracker(page);
      await settle(page, tracker, { budgetMs: config.stepBudgetMs, clockInstalled: env.clockInstalled, matchers: [] });
      startAt = target;
    }

    const trace = await runSteps(
      flow,
      context,
      page,
      baseOpts,
      env,
      { fingerprint: description?.fingerprint ?? null, verification },
      startAt,
      target,
    );

    const report = `replay of step ${stepOneBased} — tier: ${tier}\n\n${renderReport(trace)}`;
    await writeFile(join(runDir, "trace.json"), JSON.stringify(trace, null, 2), "utf8");
    await writeFile(join(runDir, "report.txt"), report, "utf8");
    return { trace, runDir, reportPath: join(runDir, "report.txt"), tier };
  } finally {
    await context.close();
  }
}

async function loadCheckpoint(config: BatConfig, file: string, step: number): Promise<Checkpoint | null> {
  const base = join(config.root, ".bat", "runs", slug(file));
  let runId: string;
  try {
    runId = (await readFile(join(base, "latest"), "utf8")).trim();
  } catch {
    // fall back to newest run dir
    const dirs = (await readdir(base).catch(() => [] as string[])).filter((d) => d !== "latest").sort();
    const last = dirs[dirs.length - 1];
    if (!last) return null;
    runId = last;
  }
  try {
    return JSON.parse(await readFile(join(base, runId, `checkpoint-${step}.json`), "utf8")) as Checkpoint;
  } catch {
    return null;
  }
}
