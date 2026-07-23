import type { Effect, Flow, Target } from "../dsl/ir.js";
import { formatTarget } from "../dsl/ir.js";
import type { RunDeps } from "./run.js";
import { completionOrder, type FlowTrace, type StepTrace } from "./trace.js";

/**
 * Automatic failure triage. Any time a run fails, bat answers the question a
 * human would otherwise spend an afternoon on: is the TEST faulty, or is the
 * APP faulty?
 *
 * The epistemics: bat holds the world (seeded, fingerprinted), the timing
 * (event-driven settlement) and the steps constant. So when identical reruns
 * disagree, the variance can only come from the app — that's a proof of app
 * nondeterminism, not a suspicion. When reruns agree, the failure is
 * deterministic and we classify it: a target that doesn't exist but nearly
 * matches something on the page is a test fault; a target that exists but
 * consistently carries the wrong state is the app behaving differently than
 * the flow expects. Chaos-induced failures are separated by rerunning once
 * without the injected conditions.
 */

export type DiagnosisVerdict =
  | "test-fault"
  | "app-inconsistent"
  | "app-behavior-mismatch"
  | "conditions-induced"
  | "inconclusive";

export interface OrderOutcome {
  order: string;
  passes: number;
  fails: number;
}

export interface Diagnosis {
  verdict: DiagnosisVerdict;
  /** the one-line answer: "THE TEST IS FAULTY: …" / "THE APP IS FAULTY: …" */
  headline: string;
  details: string[];
  reruns: { total: number; passed: number; failedSame: number; failedOther: number };
  orderEvidence?: OrderOutcome[];
}

export interface DiagnoseDeps extends RunDeps {
  /** rerun the flow once, without diagnosis or persistence */
  rerun: (deps: RunDeps) => Promise<FlowTrace>;
}

export async function diagnoseFailure(flow: Flow, original: FlowTrace, deps: DiagnoseDeps): Promise<Diagnosis> {
  const failedStep = original.steps.find((s) => s.status === "fail");
  if (!failedStep) {
    return {
      verdict: "inconclusive",
      headline: "no failing step found in the trace (runner bug — please report)",
      details: [],
      reruns: { total: 0, passed: 0, failedSame: 0, failedOther: 0 },
    };
  }

  // ---- fast path: ambiguity is always a test fault, no reruns needed
  if (failedStep.failure?.includes("is ambiguous:")) {
    return {
      verdict: "test-fault",
      headline: "THE TEST IS FAULTY: the target matches more than one element.",
      details: [
        "bat refuses to guess between multiple matches; scope the target ('in <container>') or use a testid.",
        "No reruns were needed — ambiguity is a property of the flow, not the app.",
      ],
      reruns: { total: 0, passed: 0, failedSame: 0, failedOther: 0 },
    };
  }

  const suggestions = nearMissSuggestions(failedStep);

  // ---- rerun phase: establish determinism under identical conditions
  const rerunCount = deps.config.diagnoseReruns ?? 4;
  const runs: FlowTrace[] = [original];
  for (let i = 0; i < rerunCount; i++) {
    runs.push(await deps.rerun(deps));
  }
  const sigOf = (t: FlowTrace): string | null => {
    const s = t.steps.find((st) => st.status === "fail");
    if (!s) return null;
    return `${s.index}|${s.effects.filter((e) => !e.pass).map((e) => e.rendered).join("|")}|${s.failure?.split("\n")[0] ?? ""}`;
  };
  const originalSig = sigOf(original);
  const rerunTraces = runs.slice(1);
  const passed = rerunTraces.filter((t) => t.status === "pass").length;
  const failedSame = rerunTraces.filter((t) => t.status === "fail" && sigOf(t) === originalSig).length;
  const failedOther = rerunTraces.filter((t) => t.status === "fail" && sigOf(t) !== originalSig).length;
  const rerunSummary = { total: rerunCount, passed, failedSame, failedOther };

  // ---- conditions check: does it pass without the injected chaos?
  if (deps.config.conditions && passed === 0) {
    const { conditions: _drop, ...cleanConfig } = deps.config;
    const clean = await deps.rerun({ ...deps, config: cleanConfig });
    if (clean.status === "pass") {
      return {
        verdict: "conditions-induced",
        headline:
          "THE FAILURE IS CHAOS-INDUCED: the flow passes without the injected conditions and fails under them.",
        details: [
          `Conditions active: seed ${deps.config.conditions.seed}` +
            (deps.config.conditions.latencyMs ? `, latency +${deps.config.conditions.latencyMs[0]}–${deps.config.conditions.latencyMs[1]}ms` : "") +
            (deps.config.conditions.failRate ? `, ${Math.round(deps.config.conditions.failRate * 100)}% request failures` : "") + ".",
          "A clean rerun (no injected conditions) passed. If the app should tolerate these conditions, this is an app resilience bug; if not, relax the profile.",
          "Injected requests are marked [injected …] in the network listing above.",
        ],
        reruns: rerunSummary,
      };
    }
  }

  // ---- verdicts
  if (passed > 0) {
    const evidence = orderCrossTab(runs, failedStep.index);
    const determined = evidence.length > 1 && evidence.every((o) => o.passes === 0 || o.fails === 0);
    const details = [
      `Reran the flow ${rerunCount}× under identical conditions — same world (${original.worldFingerprint ?? "n/a"}), same steps, event-driven timing: ${passed} passed, ${failedSame + failedOther} failed.`,
      "The test cannot cause this: bat holds world and timing constant, so run-to-run variance can only come from the app.",
    ];
    if (determined) {
      details.push("Response completion order at the failing step fully determines the outcome:");
      for (const o of evidence) details.push(`  ${o.order}   →   ${o.passes} pass, ${o.fails} fail`);
      details.push("The app renders different results depending on which response lands last — a race in the app.");
    } else if (evidence.length > 1) {
      details.push("Response completion order varied across runs but does not fully explain the outcome; compare the failing/passing traces for what else differed.");
    }
    const diag: Diagnosis = {
      verdict: "app-inconsistent",
      headline: "THE APP IS FAULTY (nondeterministic): identical runs produced different outcomes.",
      details,
      reruns: rerunSummary,
    };
    if (evidence.length > 0) diag.orderEvidence = evidence;
    return diag;
  }

  if (failedOther > 0) {
    return {
      verdict: "app-inconsistent",
      headline: "THE APP IS FAULTY (nondeterministic): every run failed, but not always the same way.",
      details: [
        `${failedSame + 1} run(s) failed like the original; ${failedOther} failed with a different signature.`,
        "Multiple failure modes under identical conditions means app behavior varies run to run.",
      ],
      reruns: rerunSummary,
    };
  }

  // fully deterministic failure
  if (suggestions.length > 0) {
    return {
      verdict: "test-fault",
      headline: "THE TEST IS LIKELY FAULTY: the target does not exist, but something close does.",
      details: [
        ...suggestions.map((s) => `did you mean ${s}?`),
        `The failure reproduced identically in all ${rerunCount} reruns — the page's content is stable; the flow's name for it appears wrong.`,
      ],
      reruns: rerunSummary,
    };
  }

  const observed = failedStep.effects
    .filter((e) => !e.pass && e.observed)
    .map((e) => `  ${e.rendered} — observed: ${e.observed}`);
  return {
    verdict: "app-behavior-mismatch",
    headline:
      "THE APP CONSISTENTLY BEHAVES DIFFERENTLY THAN THE FLOW EXPECTS — this is not flakiness.",
    details: [
      `The identical failure reproduced in all ${rerunCount} reruns (same world, same steps).`,
      ...(observed.length ? ["Every run observed the same thing:", ...observed] : []),
      "Two possibilities remain, and only intent can decide: the app regressed (fix the app), or the expectation is stale (update the flow). Either way, rerunning will not change the outcome.",
    ],
    reruns: rerunSummary,
  };
}

// ---------------------------------------------------------------------------
// near-miss detection: the target doesn't exist, but something similar does

function nearMissSuggestions(step: StepTrace): string[] {
  const wanted: Target[] = [];
  if (step.failedTarget) wanted.push(step.failedTarget);
  for (const v of step.effects) {
    if (v.pass) continue;
    const missing =
      v.observed !== undefined &&
      (v.observed.startsWith("no visible") || v.observed.includes("(element not found)") || v.observed.includes("no visible match"));
    if (!missing) continue;
    const eff = v.effect as Effect;
    const target =
      "target" in eff && eff.target ? eff.target : eff.type === "let" ? eff.from : undefined;
    if (target) wanted.push(target);
  }

  const candidates = snapshotCandidates(step.ariaSnapshot ?? "");
  const suggestions: string[] = [];
  for (const t of wanted) {
    if (t.name === undefined) continue;
    if (t.kind === "testid") {
      const best = bestMatch(t.name, (step.testids ?? []).map((id) => ({ kind: "testid", name: id })));
      if (best) suggestions.push(`testid "${best.name}" (instead of ${formatTarget(t)})`);
      continue;
    }
    const sameKind = candidates.filter((c) => c.kind === t.kind);
    const best = bestMatch(t.name, sameKind) ?? bestMatch(t.name, candidates);
    if (best) suggestions.push(`${best.kind} "${best.name}" (instead of ${formatTarget(t)})`);
  }
  return [...new Set(suggestions)];
}

function bestMatch(name: string, candidates: Array<{ kind: string; name: string }>): { kind: string; name: string } | null {
  let best: { kind: string; name: string } | null = null;
  let bestScore = 0.55; // threshold: below this, suggesting would mislead
  for (const c of candidates) {
    if (c.name === name) continue; // exact match means it's not a naming problem
    const s = similarity(name, c.name);
    if (s > bestScore) {
      best = c;
      bestScore = s;
    }
  }
  return best;
}

export function snapshotCandidates(snapshot: string): Array<{ kind: string; name: string }> {
  const out: Array<{ kind: string; name: string }> = [];
  for (const m of snapshot.matchAll(/-\s+([a-z]+)\s+"((?:[^"\\]|\\.)*)"/g)) {
    out.push({ kind: m[1]!, name: m[2]! });
  }
  return out;
}

export function similarity(a: string, b: string): number {
  const x = a.toLowerCase().trim();
  const y = b.toLowerCase().trim();
  if (x === y) return 1;
  if (x.includes(y) || y.includes(x)) return 0.85;
  const d = levenshtein(x, y);
  return 1 - d / Math.max(x.length, y.length);
}

function levenshtein(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;
  let prev = Array.from({ length: n + 1 }, (_, j) => j);
  for (let i = 1; i <= m; i++) {
    const cur = [i, ...new Array<number>(n).fill(0)];
    for (let j = 1; j <= n; j++) {
      cur[j] = Math.min(prev[j]! + 1, cur[j - 1]! + 1, prev[j - 1]! + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return prev[n]!;
}

// ---------------------------------------------------------------------------

function orderCrossTab(runs: FlowTrace[], stepIndex: number): OrderOutcome[] {
  const tab = new Map<string, { passes: number; fails: number }>();
  for (const t of runs) {
    const step = t.steps[stepIndex];
    if (!step || step.status === "not-run") continue;
    const order = completionOrder(step.requests).join(" → ") || "(no api traffic)";
    const cell = tab.get(order) ?? { passes: 0, fails: 0 };
    if (t.status === "pass") cell.passes++;
    else cell.fails++;
    tab.set(order, cell);
  }
  return [...tab.entries()]
    .map(([order, cell]) => ({ order, ...cell }))
    .sort((a, b) => b.passes + b.fails - (a.passes + a.fails));
}

export function renderDiagnosis(d: Diagnosis): string {
  const out: string[] = [];
  out.push("─".repeat(72));
  out.push(`diagnosis: ${d.headline}`);
  for (const line of d.details) out.push(`  ${line}`);
  if (d.reruns.total > 0) {
    out.push(
      `  (evidence base: ${d.reruns.total} automatic rerun(s) — ${d.reruns.passed} passed, ${d.reruns.failedSame} failed identically, ${d.reruns.failedOther} failed differently)`,
    );
  }
  return out.join("\n");
}
