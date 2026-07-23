#!/usr/bin/env node
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import process from "node:process";
import { chromium } from "playwright";
import { globFiles, loadConfig, loadSeeds, loadWorldHandle, ConfigError } from "./config.js";
import { parseFlow, FlowParseError } from "./dsl/parser.js";
import { composeFlowWorld, FlowSetupError } from "./runner/executor.js";
import { launchBrowser, replayStep, runFlowFile } from "./runner/run.js";
import { NetworkTracker, settle } from "./runner/settle.js";
import { renderReport } from "./runner/trace.js";
import { WorldError } from "./world/algebra.js";

const USAGE = `bat — browser auto tests

usage:
  bat check [flows...]        parse + static checks (no browser)
  bat run [flows...]          run flows; every failure is auto-triaged
                              (test-fault vs app-fault, with rerun evidence)
  bat replay <flow>:<step>    replay one step (add --fast to restore from checkpoint)
  bat inspect <url>           dump a page's semantic tree (write targets from ground truth)
  bat doctor                  world adapter capability level + the next rung
options:
  --headed                    show the browser
  --config <dir>              project root containing bat.config.json (default: cwd)
conditions (simulated bad conditions, always recorded and attributed):
  --latency <lo>-<hi>         inject lo..hi ms of latency per request
  --fail-rate <p>             fail p (0..1) of requests at the network level
  --seed <n>                  PRNG seed (required with any condition; reproducible)
`;

async function main(): Promise<number> {
  const args = process.argv.slice(2);
  const cmd = args.shift();
  if (!cmd || cmd === "--help" || cmd === "-h") {
    console.log(USAGE);
    return cmd ? 0 : 1;
  }

  const flags = new Set<string>();
  const values = new Map<string, string>();
  const VALUE_FLAGS = new Set(["config", "latency", "fail-rate", "seed"]);
  const positional: string[] = [];
  for (let i = 0; i < args.length; i++) {
    const a = args[i]!;
    if (a.startsWith("--") && VALUE_FLAGS.has(a.slice(2))) {
      const v = args[++i];
      if (v === undefined) {
        console.error(`${a} needs a value`);
        return 1;
      }
      values.set(a.slice(2), v);
    } else if (a.startsWith("--")) flags.add(a.slice(2));
    else positional.push(a);
  }
  const cwd = values.has("config") ? resolve(values.get("config")!) : process.cwd();

  const conditionsFromFlags = (): { latencyMs?: [number, number]; failRate?: number; seed: number } | null => {
    const latency = values.get("latency");
    const failRate = values.get("fail-rate");
    if (!latency && !failRate) return null;
    const seed = values.get("seed");
    if (seed === undefined) {
      throw new ConfigError("conditions require --seed <n> so runs are reproducible");
    }
    const profile: { latencyMs?: [number, number]; failRate?: number; seed: number } = { seed: Number(seed) };
    if (latency) {
      const m = /^(\d+)-(\d+)$/.exec(latency);
      if (!m) throw new ConfigError(`--latency must look like 200-1500, got '${latency}'`);
      profile.latencyMs = [Number(m[1]), Number(m[2])];
    }
    if (failRate) {
      const p = Number(failRate);
      if (!(p >= 0 && p <= 1)) throw new ConfigError(`--fail-rate must be 0..1, got '${failRate}'`);
      profile.failRate = p;
    }
    return profile;
  };

  switch (cmd) {
    case "check": {
      const config = await loadConfig(cwd);
      const seeds = await loadSeeds(config);
      const files = positional.length ? positional.map((f) => resolve(f)) : await globFiles(config.root, config.flows);
      if (files.length === 0) {
        console.error(`no flow files found (pattern: ${config.flows})`);
        return 1;
      }
      let failed = 0;
      for (const file of files) {
        try {
          const flow = parseFlow(await readFile(file, "utf8"), file);
          composeFlowWorld(flow, seeds); // seed existence, merge conflicts, closure, patches
          console.log(`✓ ${file} (${flow.steps.length} steps)`);
        } catch (e) {
          failed++;
          console.error(`✗ ${e instanceof Error ? e.message : String(e)}`);
        }
      }
      return failed ? 1 : 0;
    }

    case "run": {
      const cond = conditionsFromFlags();
      const config = await loadConfig(cwd, {
        ...(flags.has("headed") ? { headless: false } : {}),
        ...(cond ? { conditions: cond } : {}),
      });
      const seeds = await loadSeeds(config);
      const world = await loadWorldHandle(config);
      const files = positional.length ? positional.map((f) => resolve(f)) : await globFiles(config.root, config.flows);
      if (files.length === 0) {
        console.error(`no flow files found (pattern: ${config.flows})`);
        return 1;
      }
      const browser = await launchBrowser(config);
      let failed = 0;
      try {
        for (const file of files) {
          const { trace, reportPath } = await runFlowFile(file, { config, world, seeds, browser });
          if (trace.status === "pass") {
            console.log(`✓ ${trace.flow} (${trace.steps.length} steps)`);
          } else {
            failed++;
            console.error(renderReport(trace));
            console.error(`\nfull trace: ${reportPath.replace(/report\.txt$/, "trace.json")}`);
          }
        }
      } finally {
        await browser.close();
      }
      return failed ? 1 : 0;
    }

    case "replay": {
      const spec = positional[0];
      const m = spec ? /^(.+):(\d+)$/.exec(spec) : null;
      if (!m) {
        console.error("usage: bat replay <flow-file>:<step-number>");
        return 1;
      }
      const config = await loadConfig(cwd, flags.has("headed") ? { headless: false } : {});
      const seeds = await loadSeeds(config);
      const world = await loadWorldHandle(config);
      const browser = await launchBrowser(config);
      try {
        const result = await replayStep(resolve(m[1]!), Number(m[2]), { config, world, seeds, browser }, {
          fast: flags.has("fast"),
        });
        console.log(`tier: ${result.tier}\n`);
        console.log(renderReport(result.trace));
        return result.trace.status === "pass" ? 0 : 1;
      } finally {
        await browser.close();
      }
    }

    case "inspect": {
      const url = positional[0];
      if (!url) {
        console.error("usage: bat inspect <url>");
        return 1;
      }
      const browser = await chromium.launch({ headless: !flags.has("headed") });
      try {
        const page = await browser.newPage();
        const tracker = new NetworkTracker(page);
        await page.goto(url, { waitUntil: "domcontentloaded" });
        await settle(page, tracker, { budgetMs: 15000, clockInstalled: false, matchers: [] });
        console.log(`# semantic tree of ${page.url()} (settled)\n`);
        console.log(await page.locator("body").ariaSnapshot());
        const testids = await page.$$eval("[data-testid]", (els) =>
          els.map((el) => ({
            testid: el.getAttribute("data-testid"),
            tag: el.tagName.toLowerCase(),
            text: (el as HTMLElement).innerText?.slice(0, 60).replace(/\n/g, " ") ?? "",
          })),
        );
        if (testids.length) {
          console.log(`\n# testids\n`);
          for (const t of testids) console.log(`- testid "${t.testid}" (${t.tag}) ${JSON.stringify(t.text)}`);
        }
        return 0;
      } finally {
        await browser.close();
      }
    }

    case "doctor": {
      const config = await loadConfig(cwd);
      const world = await loadWorldHandle(config);
      const report = await world.doctor();
      console.log(`world adapter: ${report.levelName}\n`);
      console.log("what bat can prove for you today:");
      for (const p of report.proven) console.log(`  ✓ ${p}`);
      if (report.nextRungs.length) {
        console.log("\nthe next rungs (each is a small mechanical function):");
        for (const r of report.nextRungs) console.log(`  → ${r}`);
      } else {
        console.log("\nnothing left to climb — every guarantee is checked.");
      }
      return 0;
    }

    default:
      console.error(`unknown command '${cmd}'\n\n${USAGE}`);
      return 1;
  }
}

main().then(
  (code) => process.exit(code),
  (e) => {
    if (e instanceof FlowParseError || e instanceof WorldError || e instanceof ConfigError || e instanceof FlowSetupError) {
      console.error(e.message);
    } else {
      console.error(e);
    }
    process.exit(1);
  },
);
