import { mkdir, readFile, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { parseFlow } from "../dsl/parser.js";
import { describeConditions } from "./conditions.js";
import { runFlow, type RunDeps } from "./run.js";
import { completionOrder, type FlowTrace, type StepTrace } from "./trace.js";

/**
 * The flake hunter: run one flow many times and, when outcomes disagree,
 * DEMONSTRATE the app's inconsistency instead of just counting failures —
 * group failures by signature and cross-tabulate outcomes against response
 * completion order, the usual culprit behind "works on my machine".
 */

export interface HuntOptions {
  runs: number;
}

export interface FailureSignature {
  /** step index (0-based) + the failed expectations, rendered */
  step: number;
  stepSource: string;
  failedEffects: string[];
  /** distinct observed values per failed effect */
  observed: Record<string, string[]>;
  count: number;
}

export interface OrderOutcome {
  order: string;
  passes: number;
  fails: number;
}

export interface HuntReport {
  flow: string;
  file: string;
  runs: number;
  passes: number;
  fails: number;
  verdict: "STABLE" | "FLAKY" | "ALWAYS-FAILING";
  signatures: FailureSignature[];
  /** completion-order cross-tab at the (single most common) failing step */
  orderEvidence: OrderOutcome[] | null;
  /** true when every completion order maps to exactly one outcome */
  orderDeterminesOutcome: boolean;
  reportText: string;
}

export async function huntFlow(file: string, deps: RunDeps, opts: HuntOptions): Promise<HuntReport> {
  const source = await readFile(file, "utf8");
  const flow = parseFlow(source, file);

  const traces: FlowTrace[] = [];
  for (let i = 0; i < opts.runs; i++) {
    const { trace } = await runFlow(flow, deps);
    traces.push(trace);
  }

  const passes = traces.filter((t) => t.status === "pass").length;
  const fails = traces.length - passes;
  const verdict: HuntReport["verdict"] = fails === 0 ? "STABLE" : passes === 0 ? "ALWAYS-FAILING" : "FLAKY";

  // ---- group failures by signature
  const sigMap = new Map<string, FailureSignature>();
  for (const t of traces) {
    if (t.status !== "fail") continue;
    const failedStep = t.steps.find((s) => s.status === "fail");
    if (!failedStep) continue;
    const failedEffects = failedStep.effects.filter((e) => !e.pass).map((e) => e.rendered);
    const key = `${failedStep.index}|${failedEffects.join("|")}`;
    let sig = sigMap.get(key);
    if (!sig) {
      sig = {
        step: failedStep.index,
        stepSource: failedStep.source,
        failedEffects,
        observed: {},
        count: 0,
      };
      sigMap.set(key, sig);
    }
    sig.count++;
    for (const e of failedStep.effects) {
      if (e.pass || !e.observed) continue;
      const seen = (sig.observed[e.rendered] ??= []);
      if (!seen.includes(e.observed)) seen.push(e.observed);
    }
  }
  const signatures = [...sigMap.values()].sort((a, b) => b.count - a.count);

  // ---- completion-order evidence at the dominant failing step
  let orderEvidence: OrderOutcome[] | null = null;
  let orderDeterminesOutcome = false;
  const dominant = signatures[0];
  if (dominant) {
    const tab = new Map<string, { passes: number; fails: number }>();
    for (const t of traces) {
      const step: StepTrace | undefined = t.steps[dominant.step];
      if (!step || step.status === "not-run") continue;
      const order = completionOrder(step.requests).join(" → ") || "(no api traffic)";
      const cell = tab.get(order) ?? { passes: 0, fails: 0 };
      if (t.status === "pass") cell.passes++;
      else cell.fails++;
      tab.set(order, cell);
    }
    orderEvidence = [...tab.entries()]
      .map(([order, cell]) => ({ order, ...cell }))
      .sort((a, b) => b.passes + b.fails - (a.passes + a.fails));
    orderDeterminesOutcome =
      orderEvidence.length > 1 && orderEvidence.every((o) => o.passes === 0 || o.fails === 0);
  }

  const reportText = renderHuntReport({
    flow: flow.name,
    file,
    runs: opts.runs,
    passes,
    fails,
    verdict,
    signatures,
    orderEvidence,
    orderDeterminesOutcome,
    reportText: "",
  }, deps);

  return { flow: flow.name, file, runs: opts.runs, passes, fails, verdict, signatures, orderEvidence, orderDeterminesOutcome, reportText };
}

function renderHuntReport(r: HuntReport, deps: RunDeps): string {
  const out: string[] = [];
  out.push(`hunt: flow "${r.flow}" — ${r.runs} runs`);
  if (deps.config.conditions) {
    out.push(`conditions: SIMULATED BAD CONDITIONS ACTIVE — ${describeConditions(deps.config.conditions)}`);
  }
  out.push(`verdict: ${r.verdict} — ${r.passes} pass / ${r.fails} fail`);
  out.push("");

  if (r.fails === 0) {
    out.push(`no failures in ${r.runs} runs. If this flow flakes elsewhere, hunt it under conditions:`);
    out.push(`  bat hunt <flow> --runs ${r.runs} --latency 200-1500 --seed 1`);
    return out.join("\n");
  }

  for (const sig of r.signatures) {
    out.push(`failure signature (${sig.count}/${r.fails} failing runs): step ${sig.step + 1} '${sig.stepSource}'`);
    for (const eff of sig.failedEffects) {
      out.push(`  ✗ ${eff}`);
      for (const obs of sig.observed[eff] ?? []) out.push(`      observed: ${obs}`);
    }
  }

  if (r.orderEvidence && r.orderEvidence.length > 1) {
    out.push("");
    out.push(`evidence — response completion order at step ${r.signatures[0]!.step + 1} vs outcome:`);
    for (const o of r.orderEvidence) {
      out.push(`  ${o.order}   →   ${o.passes} pass, ${o.fails} fail`);
    }
    if (r.orderDeterminesOutcome) {
      out.push("");
      out.push(
        `  ⚑ the outcome is FULLY DETERMINED by response completion order. This is not a test problem:`,
      );
      out.push(
        `    the app renders different results depending on which response lands last — a race in the app.`,
      );
    }
  } else if (r.orderEvidence && r.orderEvidence.length === 1 && r.verdict === "FLAKY") {
    out.push("");
    out.push(
      `evidence: response completion order was identical across pass and fail runs — the inconsistency is not network-order-related. Check the failing step's observed values above for what actually varied.`,
    );
  }
  return out.join("\n");
}

export async function persistHunt(report: HuntReport, deps: RunDeps): Promise<string> {
  const slug = basename(report.file).replace(/\.[^.]+$/, "").replace(/[^\w-]+/g, "_");
  const dir = join(deps.config.root, ".bat", "hunts", slug, new Date().toISOString().replace(/[:.]/g, "-"));
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, "hunt.json"), JSON.stringify(report, null, 2), "utf8");
  await writeFile(join(dir, "report.txt"), report.reportText, "utf8");
  return dir;
}
